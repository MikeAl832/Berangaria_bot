"""Validated Telegram messages and confirmed delivery receipts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OutgoingMessage:
    text: str
    reply_mid: int | None = None
    reply_sid: int | None = None


@dataclass(frozen=True)
class DeliveredMessage:
    text: str
    message_id: int
    reply_mid: int | None = None
    reply_sid: int | None = None
