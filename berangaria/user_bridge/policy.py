"""Pure accept/reject rules for user-bridge events (no I/O, easy to unit-test)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from berangaria.user_bridge.models import BridgeEventMeta


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    accept: bool
    reason: str = ""


def resolve_bridge_chat_ids(
    configured: list[int] | tuple[int, ...] | None,
    allowed_groups: list[int] | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Effective allowlist: explicit bridge list, else allowed_groups.

    Empty configured list means «inherit allowed_groups». Intersection is not
    applied when configured is non-empty — the operator listed chats on purpose —
    but every chat must still pass group-only + bot-only checks at event time.
    """
    if configured:
        return tuple(int(x) for x in configured)
    if allowed_groups:
        return tuple(int(x) for x in allowed_groups)
    return ()


def decide_bridge_event(meta: BridgeEventMeta) -> PolicyDecision:
    """Return whether a raw MTProto event should be ingested."""
    if not meta.is_group:
        return PolicyDecision(False, "not_group")
    if not meta.sender_is_bot:
        return PolicyDecision(False, "not_bot")
    if meta.our_bot_id is not None and meta.sender_id == meta.our_bot_id:
        return PolicyDecision(False, "self_bot")
    if meta.allowed_chat_ids and meta.chat_id not in meta.allowed_chat_ids:
        return PolicyDecision(False, "chat_not_allowed")
    if not meta.allowed_chat_ids:
        # No allowlist configured anywhere — refuse rather than watch the whole account.
        return PolicyDecision(False, "empty_allowlist")
    if meta.message_id <= 0:
        return PolicyDecision(False, "bad_message_id")
    return PolicyDecision(True, "ok")


def message_mentions_bot(
    text: str,
    *,
    bot_id: int,
    bot_username: Optional[str],
    bot_first_name: Optional[str],
    reply_to_user_id: Optional[int],
    bot_names: list[str] | tuple[str, ...] = (),
) -> bool:
    """Whether the bridge message is addressed to Berangaria (mention or reply)."""
    import re

    if reply_to_user_id is not None and reply_to_user_id == bot_id:
        return True

    message_text = text or ""
    if not message_text:
        return False

    if bot_username and f"@{bot_username}" in message_text:
        return True

    names: list[str] = []
    if bot_first_name:
        names.append(bot_first_name)
    names.extend(n for n in bot_names if n)

    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", message_text, re.IGNORECASE):
            return True
    return False
