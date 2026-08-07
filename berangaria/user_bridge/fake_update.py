"""Minimal PTB-shaped objects so send_llm_request can reply via Bot API.

The user bridge never receives a real Bot API Update for other bots' messages.
These stand-ins expose only the attributes the delivery path touches.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional


class BridgeChat:
    def __init__(self, chat_id: int, bot: Any, title: str = ""):
        self.id = chat_id
        self.type = "supergroup"
        self.title = title or ""
        self._bot = bot

    async def send_action(self, action: str = "typing"):
        await self._bot.send_chat_action(chat_id=self.id, action=action)


class BridgeUser:
    def __init__(
        self,
        user_id: int,
        first_name: str,
        username: Optional[str] = None,
        *,
        is_bot: bool = True,
    ):
        self.id = user_id
        self.first_name = first_name
        self.username = username
        self.is_bot = is_bot


class BridgeTgMessage:
    def __init__(
        self,
        *,
        chat: BridgeChat,
        user: BridgeUser,
        message_id: int,
        text: str,
        bot: Any,
        date: Optional[datetime] = None,
        message_thread_id: Optional[int] = None,
    ):
        self.chat = chat
        self.from_user = user
        self.message_id = message_id
        self.text = text or ""
        self.caption = None
        self.date = date or datetime.now(timezone.utc)
        self.message_thread_id = message_thread_id
        self.reply_to_message = None
        self._bot = bot

    async def reply_text(self, text: str, **kwargs):
        return await self._bot.send_message(
            chat_id=self.chat.id,
            text=text,
            reply_to_message_id=self.message_id,
            **kwargs,
        )


class BridgeUpdate:
    def __init__(self, message: BridgeTgMessage):
        self.message = message
        self.effective_chat = message.chat
        self.effective_user = message.from_user
        self.edited_message = None
        self.message_reaction = None


def build_bridge_update(
    *,
    bot: Any,
    chat_id: int,
    message_id: int,
    sender_id: int,
    sender_name: str,
    text: str = "",
    sender_username: Optional[str] = None,
    created_at: Optional[float] = None,
    chat_title: str = "",
) -> BridgeUpdate:
    chat = BridgeChat(chat_id, bot, title=chat_title)
    user = BridgeUser(sender_id, sender_name, sender_username, is_bot=True)
    date = None
    if created_at is not None:
        date = datetime.fromtimestamp(created_at, tz=timezone.utc)
    message = BridgeTgMessage(
        chat=chat,
        user=user,
        message_id=message_id,
        text=text,
        bot=bot,
        date=date,
    )
    return BridgeUpdate(message)


def build_bridge_context(bot: Any) -> SimpleNamespace:
    """ContextTypes-compatible namespace with a real Bot instance."""
    return SimpleNamespace(bot=bot, application=getattr(bot, "application", None))
