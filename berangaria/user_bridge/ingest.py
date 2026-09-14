"""Push normalized bridge messages into the existing debounce → LLM pipeline."""

from __future__ import annotations

import logging
from typing import Any

from berangaria.user_bridge.fake_update import build_bridge_context, build_bridge_update
from berangaria.user_bridge.models import BridgeMessage

logger = logging.getLogger(__name__)


async def ingest_bridge_message(bot: Any, message: BridgeMessage) -> None:
    """Hand off to handlers.queue_bridge_bot_message. Never raises to the caller."""
    try:
        from berangaria.chat import handlers

        update = build_bridge_update(
            bot=bot,
            chat_id=message.chat_id,
            message_id=message.message_id,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            text=message.text,
            sender_username=message.sender_username,
            created_at=message.created_at,
        )
        context = build_bridge_context(bot)
        await handlers.queue_bridge_bot_message(
            update,
            context,
            text=message.text,
            media_description=message.media_description,
            media_kind=message.media_kind,
            reply_to_name=message.reply_to_name,
            reply_to_text=message.reply_to_text,
            reply_quote_selected=message.reply_quote_selected,
            reply_quote_position=message.reply_quote_position,
            reply_to_user_id=message.reply_to_user_id,
            created_at=message.created_at,
        )
    except Exception:
        logger.exception(
            "user_bridge: ingest failed chat=%s mid=%s",
            message.chat_id,
            message.message_id,
        )



async def ingest_bridge_deletions(chat_id: int, message_ids: list[int]) -> None:
    """Drop deleted Telegram messages from debounce buffers and unfrozen history."""
    try:
        from berangaria.chat.history_mutations import apply_history_message_delete
        from berangaria.core import state
        from berangaria.core.state import (
            _buffer_lock,
            get_history_key,
            get_history_lock,
            histories,
            message_buffer,
            save_history,
            touch_activity,
        )

        ids = [int(mid) for mid in message_ids if int(mid) > 0]
        if not ids:
            return
        id_set = set(ids)

        async with _buffer_lock:
            prefix = f"{chat_id}_"
            for buffer_key in list(message_buffer.keys()):
                if not buffer_key.startswith(prefix):
                    continue
                data = message_buffer.get(buffer_key)
                if not data:
                    continue
                before = len(data["messages"])
                data["messages"] = [
                    message
                    for message in data["messages"]
                    if message.get("message_id") not in id_set
                ]
                removed = before - len(data["messages"])
                if removed:
                    logger.info(
                        "🗑️ [cyan]Удаление из буфера[/] chat=%s key=%s count=%s",
                        chat_id,
                        buffer_key,
                        removed,
                    )
                if not data["messages"]:
                    task = data.get("task")
                    if task is not None:
                        task.cancel()
                    del message_buffer[buffer_key]

        key = get_history_key(chat_id, False)
        changed = False
        async with get_history_lock(key):
            history = histories.get(key) or []
            for message_id in ids:
                result = apply_history_message_delete(
                    history, message_id=message_id, is_group=True
                )
                if result in {"removed", "updated"}:
                    changed = True
                    state.abandon_memory_source_by_message(
                        scope=key, message_id=message_id
                    )
                    logger.info(
                        "🗑️ [cyan]Удаление из истории[/] chat=%s mid=%s result=%s",
                        chat_id,
                        message_id,
                        result,
                    )
                elif result == "frozen":
                    logger.info(
                        "🗑️ [dim]Удаление проигнорировано (уже у провайдера)[/] "
                        "chat=%s mid=%s",
                        chat_id,
                        message_id,
                    )
            if changed:
                histories[key] = history
                touch_activity(key)
                save_history(key)
    except Exception:
        logger.exception(
            "user_bridge: delete ingest failed chat=%s ids=%s",
            chat_id,
            message_ids,
        )
