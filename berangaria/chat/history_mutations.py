"""Mutate unsent user history rows before the first provider turn freezes them."""

from __future__ import annotations

import re
from typing import Any, Literal

from berangaria.core.utils import escape_user_text

EditResult = Literal["updated", "frozen", "missing"]
DeleteResult = Literal["updated", "removed", "frozen", "missing"]

_MESSAGE_TAG_RE = re.compile(r"\[Message: [^\]]*\]")


def is_history_row_mutable(entry: dict[str, Any] | None) -> bool:
    """Same freeze rule as incoming reactions: only ``provider_sent is False``."""
    if not entry:
        return False
    return entry.get("provider_sent") is False


def _empty_message_label(*, is_group: bool) -> str:
    return "сообщение без текста" if is_group else "без текста"


def _message_tag(text: str, *, is_group: bool) -> str:
    clean = (text or "").strip()
    if clean:
        return f"[Message: {escape_user_text(clean)}]"
    return f"[Message: ({_empty_message_label(is_group=is_group)})]"


def _replace_message_tag(content: str, new_tag: str) -> str:
    if _MESSAGE_TAG_RE.search(content or ""):
        return _MESSAGE_TAG_RE.sub(new_tag, content, count=1)
    if content:
        return f"{content} {new_tag}"
    return new_tag


def _combined_telegram_text(entry: dict[str, Any]) -> str:
    messages = entry.get("telegram_messages") or []
    parts = [
        (item.get("text") or "").strip()
        for item in messages
        if isinstance(item, dict)
    ]
    return "\n".join(part for part in parts if part)


def _entry_matches_message(entry: dict[str, Any], message_id: int) -> bool:
    if entry.get("role") != "user":
        return False
    if entry.get("mid") == message_id:
        return True
    messages = entry.get("telegram_messages") or []
    return any(
        isinstance(item, dict) and item.get("mid") == message_id
        for item in messages
    )


def find_user_history_entry(
    history: list[dict[str, Any]], message_id: int
) -> dict[str, Any] | None:
    for entry in reversed(history):
        if _entry_matches_message(entry, message_id):
            return entry
    return None


def apply_user_message_edit(
    history: list[dict[str, Any]],
    *,
    message_id: int,
    new_text: str,
    is_group: bool,
) -> EditResult:
    """Update a user row still waiting for the first provider send."""
    entry = find_user_history_entry(history, message_id)
    if entry is None:
        return "missing"
    if not is_history_row_mutable(entry):
        return "frozen"

    messages = entry.get("telegram_messages")
    if isinstance(messages, list) and messages:
        updated = False
        for item in messages:
            if isinstance(item, dict) and item.get("mid") == message_id:
                item["text"] = new_text
                updated = True
                break
        if not updated and entry.get("mid") == message_id and len(messages) == 1:
            messages[0]["text"] = new_text
            updated = True
        if not updated:
            return "missing"
        combined = _combined_telegram_text(entry)
    else:
        combined = new_text

    entry["content"] = _replace_message_tag(
        entry.get("content") or "",
        _message_tag(combined, is_group=is_group),
    )
    return "updated"


def apply_user_message_delete(
    history: list[dict[str, Any]],
    *,
    message_id: int,
    is_group: bool,
) -> DeleteResult:
    """Remove one Telegram message from an unfrozen user history row."""
    entry = find_user_history_entry(history, message_id)
    if entry is None:
        return "missing"
    if not is_history_row_mutable(entry):
        return "frozen"

    messages = entry.get("telegram_messages")
    if isinstance(messages, list) and messages:
        remaining = [
            item
            for item in messages
            if not (isinstance(item, dict) and item.get("mid") == message_id)
        ]
        if len(remaining) == len(messages):
            return "missing"
        if not remaining:
            history.remove(entry)
            return "removed"
        entry["telegram_messages"] = remaining
        entry["mid"] = remaining[-1].get("mid")
        entry["content"] = _replace_message_tag(
            entry.get("content") or "",
            _message_tag(_combined_telegram_text(entry), is_group=is_group),
        )
        return "updated"

    if entry.get("mid") == message_id:
        history.remove(entry)
        return "removed"
    return "missing"


def apply_history_message_delete(
    history: list[dict[str, Any]],
    *,
    message_id: int,
    is_group: bool,
) -> DeleteResult:
    """Remove one Telegram message from any unfrozen history row.

    User rows may be merged debounce batches (``telegram_messages``). Assistant
    and event rows are matched by ``mid`` only.
    """
    user_result = apply_user_message_delete(
        history, message_id=message_id, is_group=is_group
    )
    if user_result != "missing":
        return user_result

    for entry in reversed(history):
        if entry.get("mid") != message_id:
            continue
        if not is_history_row_mutable(entry):
            return "frozen"
        history.remove(entry)
        return "removed"
    return "missing"
