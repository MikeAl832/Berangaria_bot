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
            reply_to_user_id=message.reply_to_user_id,
            created_at=message.created_at,
        )
    except Exception:
        logger.exception(
            "user_bridge: ingest failed chat=%s mid=%s",
            message.chat_id,
            message.message_id,
        )
