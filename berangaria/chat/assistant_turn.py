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
    provider_messages: list[dict] | None = None,
) -> dict | None:
    """Append delivered text plus the byte-stable provider side of the turn.

    ``content`` is the Telegram-visible text and may be cleaned for presentation.
    ``provider_messages`` is the exact assistant/tool transcript the provider saw
    or returned.  Keeping the two representations separate prevents Telegram-only
    cleanup (for example removing a final full stop) from invalidating the
    prompt prefix on the next turn.
    """
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

    exact_messages = [
        copy.deepcopy(message)
        for message in (provider_messages or [])
        if isinstance(message, dict)
    ]
    if not exact_messages and isinstance(provider_message, dict):
        exact_messages = [copy.deepcopy(provider_message)]
    for message in exact_messages:
        if not message.get("role"):
            message["role"] = "assistant"
    if exact_messages:
        entry["provider_messages"] = exact_messages

    reasoning_source = provider_message
    if not isinstance(reasoning_source, dict):
        reasoning_source = next(
            (
                message
                for message in reversed(exact_messages)
                if message.get("role") == "assistant"
            ),
            None,
        )
    if isinstance(reasoning_source, dict):
        reasoning_details = reasoning_source.get("reasoning_details")
        if isinstance(reasoning_details, list) and reasoning_details:
            entry["reasoning_details"] = copy.deepcopy(reasoning_details)
        else:
            reasoning_content = reasoning_source.get("reasoning_content")
            reasoning = reasoning_source.get("reasoning")
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
