"""Persist the assistant side of a confirmed Telegram turn."""

from typing import Any

from berangaria.core.state import (
    get_history_lock,
    histories,
    save_history,
    touch_activity,
)


async def save_assistant_turn(
    text: str,
    *,
    turn: Any,
    key: str,
    history: list,
) -> dict | None:
    """Append text and structured chat actions to persisted history."""
    if (
        not text
        and not turn.reactions_made
        and not turn.stickers_made
        and not turn.voices_made
    ):
        return None

    entry = {"role": "assistant", "content": text}
    if turn.reactions_made:
        entry["reactions"] = list(turn.reactions_made)
    if turn.stickers_made:
        entry["stickers"] = list(turn.stickers_made)
    if turn.voices_made:
        entry["voices"] = list(turn.voices_made)

    async with get_history_lock(key):
        history.append(entry)
        histories[key] = history
        touch_activity(key)
        save_history(key)
    return entry


async def remember_bot_message_id(
    entry: dict | None,
    sent_mid: int | None,
    *,
    key: str,
) -> None:
    """Attach Telegram's ID so later incoming reactions resolve correctly."""
    if entry is None or not sent_mid:
        return
    async with get_history_lock(key):
        entry["mid"] = sent_mid
        save_history(key)
