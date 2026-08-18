"""Message normalization, debounce buffering, and turn dispatch."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from berangaria.core import state
from berangaria.core.state import (
    _buffer_lock,
    get_history_key,
    get_history_lock,
    get_turn_lock,
    histories,
    message_buffer,
    touch_activity,
)
from berangaria.core.utils import (
    escape_user_text,
    is_low_signal_user_text,
    now_local,
    strip_tiktok_urls,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueueRuntime:
    """Mutable settings and patchable boundaries used by the queue."""

    message_debounce_seconds: float
    max_media_items_in_context: int
    max_buffered_messages: int
    max_buffered_chars: int
    check_access_permissions: Callable[[int, int, bool], bool]
    truncate_at_sentence: Callable[[str, int], str]
    build_memory_text: Callable[..., str]
    extract_forward_info: Callable[[Any], str | None]
    extract_reply_context: Callable[[Any], tuple[str | None, str | None]]
    log_message_preview: Callable[[str], str]
    is_bot_mentioned: Callable[..., tuple[bool, str]]
    should_reply_randomly: Callable[[int], bool]
    enqueue_memory_source: Callable[..., int]
    release_memory_sources: Callable[[list[int | None]], Any]
    abandon_memory_sources: Callable[[list[int | None]], Any]
    send_llm_request: Callable[..., Awaitable[Any]]
    process_buffered_messages: Callable[..., Awaitable[None]]


async def process_buffered_messages(
    buffer_key: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    key: str,
    is_group: bool,
    user_id: int,
    user_name: str,
    mentioned: bool,
    random_reply: bool,
    runtime: QueueRuntime,
) -> None:
    """Commit one debounce buffer to history and optionally run an LLM turn."""
    async with _buffer_lock:
        data = message_buffer.get(buffer_key)
        if not data:
            return
        messages = data["messages"]
        del message_buffer[buffer_key]

    first_msg = messages[0]
    timestamp = first_msg["timestamp"]
    author_kind = first_msg.get("author_kind") or "User"
    if author_kind not in ("User", "Bot"):
        author_kind = "User"
    message_parts = [f"[{author_kind}: {user_name}] [Time: {timestamp}]"]

    if first_msg.get("forward_info"):
        message_parts.append(f"[{first_msg['forward_info']}]")
    if is_group and first_msg["reply_to_name"]:
        message_parts.append(f"[Reply to: {first_msg['reply_to_name']}]")
        message_parts.append(f"[Quoted message: {first_msg['reply_to_text']}]")

    combined_text = "\n".join(message["text"] for message in messages if message["text"])
    if combined_text:
        message_parts.append(f"[Message: {escape_user_text(combined_text)}]")
    else:
        empty_label = "сообщение без текста" if is_group else "без текста"
        message_parts.append(f"[Message: ({empty_label})]")

    max_description_chars = 1500
    media_items = [
        (message.get("media_kind", "image"), message["media_description"])
        for message in messages
        if message.get("media_description")
    ]
    for kind, description in media_items[: runtime.max_media_items_in_context]:
        if kind == "video":
            tag = "Video description"
        elif kind == "audio":
            tag = "Audio description"
        else:
            tag = "Image description"
        description = runtime.truncate_at_sentence(
            description, max_description_chars
        )
        message_parts.append(f"[{tag}: {escape_user_text(description)}]")
    if len(media_items) > runtime.max_media_items_in_context:
        overflow = len(media_items) - runtime.max_media_items_in_context
        message_parts.append(f"[+{overflow} more media items]")

    message_content = " ".join(message_parts)
    # The lock order is an invariant: whole turn first, short history mutation second.
    async with get_turn_lock(key):
        async with get_history_lock(key):
            if key not in histories:
                histories[key] = []
            history = histories[key]
            next_sid = max(
                (message.get("sid", 0) for message in history), default=0
            ) + 1
            last_mid = messages[-1].get("message_id")
            history.append(
                {
                    "role": "user",
                    "content": message_content,
                    "sid": next_sid,
                    "mid": last_mid,
                }
            )
            histories[key] = history
            touch_activity(key)
            state.save_history(key)

        if not (mentioned or random_reply):
            return
        if (
            random_reply
            and not mentioned
            and not media_items
            and is_low_signal_user_text(combined_text)
        ):
            logger.info(
                "🤫 [dim]Ambient пропущен (low-signal):[/] %r (ключ=%s)",
                (combined_text or "")[:60],
                key,
            )
            return
        if mentioned:
            await update.message.chat.send_action(action="typing")
        await runtime.send_llm_request(
            update, context, key, history, user_name, user_id, mentioned
        )


async def queue_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    media_description: str | None,
    media_kind: str | None,
    runtime: QueueRuntime,
) -> None:
    """Normalize one user update and append it to the debounce buffer."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    is_group = update.effective_chat.type in ["group", "supergroup"]
    key = get_history_key(chat_id, not is_group, user_id)
    buffer_key = f"{chat_id}_{user_id}"

    if not runtime.check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    original_text = text or ""
    text = strip_tiktok_urls(original_text)
    now = now_local()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"
    reply_to_name, reply_to_text = runtime.extract_reply_context(update.message)
    forward_info = runtime.extract_forward_info(update.message)

    mentioned, _ = runtime.is_bot_mentioned(update, context)
    random_reply = runtime.should_reply_randomly(chat_id) if is_group else False
    if not is_group:
        mentioned = True

    msg_data = {
        "text": text,
        "media_description": media_description,
        "media_kind": media_kind,
        "timestamp": timestamp,
        "reply_to_name": reply_to_name,
        "reply_to_text": reply_to_text,
        "forward_info": forward_info,
        "message_id": update.message.message_id,
        "created_at": (
            update.message.date.timestamp()
            if getattr(update.message, "date", None) is not None
            else time.time()
        ),
        "author_kind": "User",
    }
    memory_text = runtime.build_memory_text(
        original_text, is_forwarded=forward_info is not None
    )
    memory_source_id = None
    if memory_text:
        memory_source_id = runtime.enqueue_memory_source(
            scope=key,
            text=memory_text,
            author_name=user_name,
            author_id=str(user_id),
            message_id=update.message.message_id,
            created_at=msg_data["created_at"],
            ready=False,
        )
    msg_data["memory_source_id"] = memory_source_id

    if not text and not media_description:
        runtime.release_memory_sources([memory_source_id])
        return
    await enqueue_buffered(
        buffer_key=buffer_key,
        msg_data=msg_data,
        update=update,
        context=context,
        key=key,
        is_group=is_group,
        user_id=user_id,
        user_name=user_name,
        mentioned=mentioned,
        random_reply=random_reply,
        runtime=runtime,
    )


async def queue_bridge_bot_message(
    update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text: str,
    media_description: str | None,
    media_kind: str | None,
    reply_to_name: str | None,
    reply_to_text: str | None,
    reply_to_user_id: int | None,
    created_at: float | None,
    runtime: QueueRuntime,
) -> None:
    """Ingest an allowlisted group message from the read-only user bridge."""
    from berangaria.config import BOT_NAMES
    from berangaria.user_bridge.policy import message_mentions_bot

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "bot"
    chat_type = getattr(update.effective_chat, "type", "supergroup")
    if chat_type not in ("group", "supergroup"):
        return
    if not runtime.check_access_permissions(chat_id, user_id, True):
        logger.info("👀 [dim]User bridge: чат не в allowlist chat=%s[/]", chat_id)
        return

    text = strip_tiktok_urls(text or "")
    now = now_local()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"
    bot = context.bot
    bot_id = getattr(bot, "id", None)
    if bot_id is None:
        try:
            bot_id = (await bot.get_me()).id
        except Exception:
            bot_id = 0

    mentioned = message_mentions_bot(
        text,
        bot_id=int(bot_id or 0),
        bot_username=getattr(bot, "username", None),
        bot_first_name=getattr(bot, "first_name", None),
        reply_to_user_id=reply_to_user_id,
        bot_names=BOT_NAMES,
    )
    random_reply = runtime.should_reply_randomly(chat_id)
    if not text and not media_description:
        return

    key = get_history_key(chat_id, False, user_id)
    buffer_key = f"bridge_{chat_id}_{user_id}"
    msg_data = {
        "text": text,
        "media_description": media_description,
        "media_kind": media_kind,
        "timestamp": timestamp,
        "reply_to_name": reply_to_name,
        "reply_to_text": reply_to_text,
        "forward_info": None,
        "message_id": update.message.message_id,
        "created_at": created_at if created_at is not None else time.time(),
        "author_kind": "Bot",
        "memory_source_id": None,
    }
    logger.info(
        "👀 [[blue]bridge | %s[/]] [magenta]%s[/]: %s",
        chat_id,
        user_name,
        runtime.log_message_preview(text) or "(media)",
    )
    await enqueue_buffered(
        buffer_key=buffer_key,
        msg_data=msg_data,
        update=update,
        context=context,
        key=key,
        is_group=True,
        user_id=user_id,
        user_name=user_name,
        mentioned=mentioned,
        random_reply=random_reply,
        runtime=runtime,
    )


async def enqueue_buffered(
    *,
    buffer_key: str,
    msg_data: dict,
    update,
    context: ContextTypes.DEFAULT_TYPE,
    key: str,
    is_group: bool,
    user_id: int,
    user_name: str,
    mentioned: bool,
    random_reply: bool,
    runtime: QueueRuntime,
) -> None:
    """Atomically append to a debounce buffer and schedule its flush."""

    async def wait_and_process(debounce: float | None = None):
        source_ids: list[int | None] = []
        try:
            await asyncio.sleep(
                runtime.message_debounce_seconds if debounce is None else debounce
            )
            data = message_buffer.get(buffer_key)
            if data:
                source_ids = [
                    message.get("memory_source_id")
                    for message in list(data["messages"])
                ]
                await runtime.process_buffered_messages(
                    buffer_key,
                    update,
                    context,
                    key,
                    is_group,
                    user_id,
                    user_name,
                    data["mentioned"],
                    data["random_reply"],
                )
                runtime.release_memory_sources(source_ids)
        except asyncio.CancelledError:
            pass
        except Exception:
            runtime.abandon_memory_sources(source_ids)
            raise

    async with _buffer_lock:
        if buffer_key in message_buffer:
            message_buffer[buffer_key]["task"].cancel()
            message_buffer[buffer_key]["messages"].append(msg_data)
            if mentioned:
                message_buffer[buffer_key]["mentioned"] = True
            if random_reply:
                message_buffer[buffer_key]["random_reply"] = True

            buffered = message_buffer[buffer_key]["messages"]
            buffered_chars = sum(
                len(item.get("text") or "") for item in buffered
            )
            if (
                len(buffered) >= runtime.max_buffered_messages
                or buffered_chars >= runtime.max_buffered_chars
            ):
                logger.info(
                    "📦 [dim]Буфер '%s' достиг бюджета (%s сообщ., %s симв.) — флашим[/]",
                    buffer_key,
                    len(buffered),
                    buffered_chars,
                )
                message_buffer[buffer_key]["task"] = asyncio.create_task(
                    wait_and_process(debounce=0)
                )
                return
        else:
            message_buffer[buffer_key] = {
                "messages": [msg_data],
                "mentioned": mentioned,
                "random_reply": random_reply,
            }
        message_buffer[buffer_key]["task"] = asyncio.create_task(
            wait_and_process()
        )
