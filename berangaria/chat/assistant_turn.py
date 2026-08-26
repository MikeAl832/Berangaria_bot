"""Persist the assistant side of a confirmed Telegram turn."""

import copy
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
    provider_message: dict | None = None,
) -> dict | None:
    """Append delivered text, actions, and opaque provider reasoning state."""
    if (
        not text
        and not turn.reactions_made
        and not turn.stickers_made
        and not turn.voices_made
    ):
        return None

    entry = {
        "role": "assistant",
        "content": text,
        # A freshly delivered Telegram reply has not appeared in a provider
        # request yet. Incoming reactions may still be attached to this row
        # without changing any prefix the provider could have cached.
        "provider_sent": False,
    }
    if turn.reactions_made:
        entry["reactions"] = list(turn.reactions_made)
    if turn.stickers_made:
        entry["stickers"] = list(turn.stickers_made)
    if turn.voices_made:
        entry["voices"] = list(turn.voices_made)

    if isinstance(provider_message, dict):
        reasoning_details = provider_message.get("reasoning_details")
        if isinstance(reasoning_details, list) and reasoning_details:
            entry["reasoning_details"] = copy.deepcopy(reasoning_details)
        else:
            reasoning_content = provider_message.get("reasoning_content")
            reasoning = provider_message.get("reasoning")
            if isinstance(reasoning_content, str) and reasoning_content:
                entry["reasoning_content"] = reasoning_content
            elif isinstance(reasoning, str) and reasoning:
                entry["reasoning"] = reasoning

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
