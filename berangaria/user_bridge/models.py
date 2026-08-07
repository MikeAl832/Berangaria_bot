"""Normalized bot messages produced by the user bridge (Telethon → internal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class BridgeMessage:
    """One group message from another bot, ready for ingest.

    Media is already resolved to an optional vision description — the bridge never
    hands raw Telethon objects to the LLM path.
    """

    chat_id: int
    message_id: int
    sender_id: int
    sender_name: str
    text: str = ""
    sender_username: Optional[str] = None
    media_description: Optional[str] = None
    media_kind: Optional[str] = None  # image | video | audio
    reply_to_name: Optional[str] = None
    reply_to_text: Optional[str] = None
    reply_to_user_id: Optional[int] = None
    created_at: Optional[float] = None
    is_group: bool = True


@dataclass(frozen=True, slots=True)
class BridgeEventMeta:
    """Raw fields used only by pure policy filters (no I/O)."""

    chat_id: int
    message_id: int
    is_group: bool
    sender_is_bot: bool
    sender_id: int
    our_bot_id: Optional[int] = None
    allowed_chat_ids: tuple[int, ...] = field(default_factory=tuple)
