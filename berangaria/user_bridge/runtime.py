"""Isolated Telethon loop. Failures reconnect; they never stop Bot API polling."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from berangaria.config import (
    ALLOWED_GROUPS,
    TELEGRAM_API_HASH,
    TELEGRAM_API_ID,
    USER_BRIDGE_CHAT_IDS,
    USER_BRIDGE_DEDUP_TTL_SECONDS,
    USER_BRIDGE_ENABLED,
    USER_BRIDGE_MEDIA_TIMEOUT_SECONDS,
    USER_BRIDGE_PORT,
    USER_BRIDGE_RECONNECT_SECONDS,
    USER_BRIDGE_SESSION,
)
from berangaria.user_bridge.dedup import MessageDeduper
from berangaria.user_bridge.ingest import ingest_bridge_message
from berangaria.user_bridge.media import describe_bridge_media
from berangaria.user_bridge.models import BridgeEventMeta, BridgeMessage
from berangaria.user_bridge.policy import decide_bridge_event, resolve_bridge_chat_ids


logger = logging.getLogger(__name__)

_bridge_task: Optional[asyncio.Task] = None
_stop_event: Optional[asyncio.Event] = None
_CLIENT_START_TIMEOUT = 30.0


def _credentials_ok() -> bool:
    return bool(TELEGRAM_API_ID and TELEGRAM_API_HASH and USER_BRIDGE_SESSION)


async def start_user_bridge(application: Any) -> Optional[asyncio.Task]:
    """Start the background bridge task if enabled. Always fail-open."""
    global _bridge_task, _stop_event

    if not USER_BRIDGE_ENABLED:
        logger.info("👀 [dim]User bridge: выключен (user_bridge_enabled=false)[/]")
        return None
    if not _credentials_ok():
        logger.warning(
            "👀 [yellow]User bridge включён, но нет TELEGRAM_API_ID / "
            "TELEGRAM_API_HASH / USER_BRIDGE_SESSION — пропускаю[/]"
        )
        return None

    allowed = resolve_bridge_chat_ids(USER_BRIDGE_CHAT_IDS, ALLOWED_GROUPS)
    if not allowed:
        logger.warning(
            "👀 [yellow]User bridge: пустой allowlist чатов "
            "(user_bridge_chat_ids / allowed_groups) — пропускаю[/]"
        )
        return None

    if _bridge_task is not None and not _bridge_task.done():
        logger.warning("👀 [yellow]User bridge уже запущен[/]")
        return _bridge_task

    _stop_event = asyncio.Event()
    bot = application.bot
    _bridge_task = asyncio.create_task(
        _bridge_supervisor(bot, allowed_chat_ids=allowed, stop_event=_stop_event),
        name="user-bridge",
    )
    logger.info(
        "👀 [cyan]User bridge: старт (чаты=%s, reconnect=%ss)[/]",
        list(allowed),
        USER_BRIDGE_RECONNECT_SECONDS,
    )
    return _bridge_task


async def stop_user_bridge() -> None:
    """Signal the bridge to stop and wait briefly."""
    global _bridge_task, _stop_event
    if _stop_event is not None:
        _stop_event.set()
    task = _bridge_task
    _bridge_task = None
    if task is None:
        _stop_event = None
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
        pass
    _stop_event = None
    logger.info("👀 [dim]User bridge: остановлен[/]")


async def _bridge_supervisor(
    bot: Any,
    *,
    allowed_chat_ids: tuple[int, ...],
    stop_event: asyncio.Event,
) -> None:
    """Outer loop: never exits with an uncaught error into the bot process."""
    while not stop_event.is_set():
        try:
            await _run_client_once(bot, allowed_chat_ids=allowed_chat_ids, stop_event=stop_event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("👀 [red]User bridge: сессия упала, reconnect[/]")
        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=USER_BRIDGE_RECONNECT_SECONDS)
        except asyncio.TimeoutError:
            pass


async def _resolve_our_bot_id(bot: Any) -> Optional[int]:
    """Best-effort Bot API id for skipping our own messages.

    python-telegram-bot's ``Bot.id`` is a property that raises ``RuntimeError``
    until ``ExtBot.initialize`` has cached ``get_me``. ``getattr(bot, "id", None)``
    does not catch that, so treating it as a Telethon crash caused a spurious
    reconnect on every deploy that started the bridge before ``run_polling``.
    """
    try:
        bot_id = bot.id
    except Exception:
        bot_id = None
    if isinstance(bot_id, int):
        return bot_id
    try:
        me = await bot.get_me()
        return int(me.id)
    except Exception as exc:
        logger.warning("user_bridge: get_me failed (%s) — self-skip limited", exc)
        return None


async def _run_client_once(
    bot: Any,
    *,
    allowed_chat_ids: tuple[int, ...],
    stop_event: asyncio.Event,
) -> None:
    try:
        from telethon import TelegramClient, events
        from telethon.sessions import StringSession
    except ImportError:
        logger.error("👀 [red]User bridge: telethon не установлен[/]")
        # Sleep until stop so we do not tight-loop ImportError.
        await stop_event.wait()
        return

    deduper = MessageDeduper(ttl_seconds=USER_BRIDGE_DEDUP_TTL_SECONDS)
    session = StringSession(USER_BRIDGE_SESSION)
    if USER_BRIDGE_PORT:
        if not 1 <= USER_BRIDGE_PORT <= 65535:
            raise ValueError("USER_BRIDGE_PORT must be 0 (session default) or 1..65535")
        session.set_dc(session.dc_id, session.server_address, USER_BRIDGE_PORT)
    # The supervisor owns retries. Telethon's internal reconnect callback starts
    # an untracked get_me task on every successful TCP reconnect, even when the
    # peer immediately rejects MTProto. That can grow without bound on gateways.
    client = TelegramClient(
        session,
        TELEGRAM_API_ID,
        TELEGRAM_API_HASH,
        auto_reconnect=False,
        connection_retries=0,
        request_retries=0,
        timeout=10,
    )

    our_bot_id = await _resolve_our_bot_id(bot)

    @client.on(events.NewMessage(chats=list(allowed_chat_ids)))
    async def _on_new_message(event) -> None:  # type: ignore[no-untyped-def]
        if stop_event.is_set():
            return
        # Never let a single event kill the client loop.
        try:
            await _handle_event(
                event,
                bot=bot,
                client=client,
                allowed_chat_ids=allowed_chat_ids,
                our_bot_id=our_bot_id,
                deduper=deduper,
            )
        except Exception:
            logger.exception("user_bridge: event handler error")

    try:
        # Telethon's socket timeout doesn't bound its initialization RPCs.
        async with asyncio.timeout(_CLIENT_START_TIMEOUT):
            await client.connect()
            if not await client.is_user_authorized():
                logger.error(
                    "👀 [red]User bridge: session не авторизована "
                    "(перелогиньтесь через scripts/user_bridge_login.py)[/]"
                )
                return
            me = await client.get_me()
        logger.info(
            "👀 [green]User bridge подключён как %s (id=%s)[/]",
            getattr(me, "username", None) or getattr(me, "first_name", "?"),
            getattr(me, "id", "?"),
        )
        # Run until stop. disconnect on exit.
        wait_stop = asyncio.create_task(stop_event.wait())
        try:
            # Observe the disconnection future, including its network exception.
            # Polling is_connected alone loses those errors.
            disconnected = client.disconnected
            done, _pending = await asyncio.wait(
                {wait_stop, disconnected}, return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in done:
                await disconnected
        finally:
            if not wait_stop.done():
                wait_stop.cancel()
                try:
                    await wait_stop
                except asyncio.CancelledError:
                    pass
    finally:
        try:
            await client.disconnect()
        except Exception:
            logger.exception("user_bridge: disconnect failed")
        finally:
            # A failed connect can set this future before we start waiting on it.
            disconnected = client.disconnected
            if disconnected.done() and not disconnected.cancelled():
                disconnected.exception()


async def _handle_event(
    event: Any,
    *,
    bot: Any,
    client: Any,
    allowed_chat_ids: tuple[int, ...],
    our_bot_id: Optional[int],
    deduper: MessageDeduper,
) -> None:
    msg = event.message
    if msg is None:
        return

    chat_id = int(event.chat_id)
    message_id = int(msg.id)
    # Supergroups are channels in MTProto; treat allowlisted chats as groups.
    is_group = bool(event.is_group) or (
        bool(getattr(event, "is_channel", False)) and chat_id in allowed_chat_ids
    )

    sender = await event.get_sender()
    sender_is_bot = bool(getattr(sender, "bot", False)) if sender is not None else False
    sender_id = int(getattr(sender, "id", 0) or 0)

    decision = decide_bridge_event(
        BridgeEventMeta(
            chat_id=chat_id,
            message_id=message_id,
            is_group=is_group,
            sender_is_bot=sender_is_bot,
            sender_id=sender_id,
            our_bot_id=our_bot_id,
            allowed_chat_ids=allowed_chat_ids,
        )
    )
    if not decision.accept:
        return

    if deduper.seen_or_add(chat_id, message_id):
        logger.debug("user_bridge: dedup drop chat=%s mid=%s", chat_id, message_id)
        return

    sender_name = (
        getattr(sender, "first_name", None)
        or getattr(sender, "title", None)
        or getattr(sender, "username", None)
        or f"bot:{sender_id}"
    )
    sender_username = getattr(sender, "username", None)
    text = (msg.message or msg.text or "") if msg else ""
    # caption for media often lives in message text field in Telethon
    caption = text

    reply_to_name = None
    reply_to_text = None
    reply_quote_selected = False
    reply_quote_position = None
    reply_to_user_id = None
    if msg.is_reply:
        try:
            reply = await msg.get_reply_message()
            if reply is not None:
                r_sender = await reply.get_sender()
                if r_sender is not None:
                    reply_to_name = (
                        getattr(r_sender, "first_name", None)
                        or getattr(r_sender, "title", None)
                        or getattr(r_sender, "username", None)
                    )
                    reply_to_user_id = int(getattr(r_sender, "id", 0) or 0) or None
                reply_header = getattr(msg, "reply_to", None)
                selected_quote = getattr(reply_header, "quote_text", None)
                if selected_quote:
                    reply_to_text = str(selected_quote)[:1024]
                    reply_quote_selected = True
                    raw_offset = getattr(reply_header, "quote_offset", None)
                    if isinstance(raw_offset, int):
                        reply_quote_position = raw_offset
                else:
                    reply_to_text = (
                        reply.message or reply.text or "сообщение без текста"
                    )[:80]
        except Exception as exc:
            logger.debug("user_bridge: reply context failed: %s", exc)

    media_description = None
    media_kind = None
    try:
        media_description, media_kind = await asyncio.wait_for(
            describe_bridge_media(client, msg, caption=caption),
            timeout=USER_BRIDGE_MEDIA_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "user_bridge: media timeout chat=%s mid=%s", chat_id, message_id
        )
    except Exception:
        logger.exception("user_bridge: media path error")

    if not (text or "").strip() and not media_description:
        return

    created_at = None
    if getattr(msg, "date", None) is not None:
        try:
            created_at = msg.date.timestamp()
        except Exception:
            created_at = None

    bridge_msg = BridgeMessage(
        chat_id=chat_id,
        message_id=message_id,
        sender_id=sender_id,
        sender_name=str(sender_name),
        text=text or "",
        sender_username=sender_username,
        media_description=media_description,
        media_kind=media_kind,
        reply_to_name=reply_to_name,
        reply_to_text=reply_to_text,
        reply_quote_selected=reply_quote_selected,
        reply_quote_position=reply_quote_position,
        reply_to_user_id=reply_to_user_id,
        created_at=created_at,
        is_group=True,
    )
    await ingest_bridge_message(bot, bridge_msg)
