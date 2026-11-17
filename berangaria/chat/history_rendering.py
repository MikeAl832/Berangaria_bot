"""Pure conversion between persisted chat history and provider messages."""

import copy
import re

from berangaria.core import state
from berangaria.core.utils import strip_tiktok_urls


def build_sid_map(history: list) -> dict:
    """Map stable reply handles to Telegram message IDs."""
    return {
        message["sid"]: message.get("mid")
        for message in history
        if message.get("role") == "user" and message.get("sid") is not None
    }


def _build_mid_to_sid(history: list) -> dict:
    """Map Telegram message IDs back to current reply handles."""
    result = {}
    for message in history or []:
        if (
            message.get("role") == "user"
            and message.get("mid") is not None
            and message.get("sid") is not None
        ):
            result[message["mid"]] = message["sid"]
    return result


def _format_reaction_note_part(reaction: dict, mid_to_sid: dict) -> str:
    emoji = reaction.get("emoji") or ""
    quote = (reaction.get("on") or "").strip()
    message_id = reaction.get("on_mid")
    sid = mid_to_sid.get(message_id) if message_id is not None else None
    if sid is not None:
        return f"{emoji} на [#{sid}] «{quote}»" if quote else f"{emoji} на [#{sid}]"
    return f"{emoji} на «{quote}»" if quote else emoji


def _render_assistant_message(message: dict, content: str) -> dict:
    """Rebuild one assistant message with provider state kept out of display text."""
    rendered = {"role": "assistant", "content": content}
    reasoning_details = message.get("reasoning_details")
    if isinstance(reasoning_details, list) and reasoning_details:
        rendered["reasoning_details"] = copy.deepcopy(reasoning_details)
        return rendered

    reasoning_content = message.get("reasoning_content")
    reasoning = message.get("reasoning")
    if isinstance(reasoning_content, str) and reasoning_content:
        rendered["reasoning_content"] = reasoning_content
    elif isinstance(reasoning, str) and reasoning:
        rendered["reasoning"] = reasoning
    return rendered


def _render_provider_messages(message: dict) -> list[dict]:
    """Return the exact provider transcript stored for a confirmed turn."""
    provider_messages = message.get("provider_messages")
    if not isinstance(provider_messages, list):
        return []
    return [
        copy.deepcopy(provider_message)
        for provider_message in provider_messages
        if isinstance(provider_message, dict) and provider_message.get("role")
    ]


def _render_history_event(message: dict) -> dict | None:
    """Convert one structured timeline event into ephemeral provider context."""
    if message.get("event_kind") != "incoming_reaction":
        return None

    actor_name = message.get("actor_name") or "кто-то"
    quote = (message.get("target_excerpt") or "").strip()
    target = f"твоё сообщение «{quote}»" if quote else "твоё сообщение"
    changes = []
    added = [str(emoji) for emoji in message.get("added") or [] if emoji]
    removed = [str(emoji) for emoji in message.get("removed") or [] if emoji]
    if added:
        changes.append("добавлены " + ", ".join(added))
    if removed:
        changes.append("убраны " + ", ".join(removed))
    if not changes:
        return None

    return {
        "role": "system",
        "content": (
            f"Реакции от {actor_name} на {target}: {'; '.join(changes)}. "
            "(это действие в чате, не текст)."
        ),
    }


def _resolve_native_image_parts(message: dict) -> list[dict]:
    """Build OpenAI-compatible image_url blocks from cached image references."""
    refs = message.get("media_images")
    if not isinstance(refs, list) or not refs:
        return []
    blocks: list[dict] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        uid = ref.get("id")
        if not isinstance(uid, str) or not uid:
            continue
        cached = state.get_cached_media_image(uid)
        if cached is None:
            continue
        b64_data, mime = cached
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64_data}"},
            }
        )
    return blocks


def render_history_for_api(
    history: list,
    *,
    native_images: bool = False,
    image_window: int | None = None,
) -> list:
    """Render persisted history into an ephemeral provider payload."""
    mid_to_sid = _build_mid_to_sid(history)
    eligible_image_indices: set[int] = set()
    if native_images:
        user_indices = [
            idx for idx, item in enumerate(history) if item.get("role") == "user"
        ]
        if image_window is None:
            eligible_image_indices = set(user_indices)
        elif image_window > 0:
            eligible_image_indices = set(user_indices[-image_window:])

    output = []
    for idx, message in enumerate(history):
        role = message.get("role")
        if role == "event":
            rendered_event = _render_history_event(message)
            if rendered_event is not None:
                output.append(rendered_event)
            continue

        content = message.get("content", "")
        if isinstance(content, str) and content:
            content = strip_tiktok_urls(content)
        sid = message.get("sid")
        if sid is not None and role == "user":
            content = f"[#{sid}] {content}"

        reactions = message.get("reactions") if role == "assistant" else None
        incoming = message.get("incoming_reactions") if role == "assistant" else None
        stickers = message.get("stickers") if role == "assistant" else None
        provider_messages = _render_provider_messages(message) if role == "assistant" else []
        if not (reactions or incoming or stickers):
            if role == "assistant":
                if provider_messages:
                    output.extend(provider_messages)
                else:
                    output.append(_render_assistant_message(message, content))
            elif role == "user" and idx in eligible_image_indices:
                image_blocks = _resolve_native_image_parts(message)
                if image_blocks:
                    output.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": content}, *image_blocks],
                        }
                    )
                else:
                    output.append({"role": role, "content": content})
            else:
                output.append({"role": role, "content": content})
            continue

        if provider_messages:
            output.extend(provider_messages)
        elif content:
            output.append(_render_assistant_message(message, content))

        notes = []
        if reactions:
            parts = [_format_reaction_note_part(item, mid_to_sid) for item in reactions]
            notes.append("Ты поставила реакцию " + ", ".join(parts) + ".")
        if stickers:
            parts = []
            for sticker in stickers:
                description = (sticker.get("desc") or "").strip()
                if len(description) > 80:
                    description = description[:80] + "…"
                emotion = sticker.get("emotion")
                parts.append(
                    f"[{emotion}] «{description}»" if emotion else f"«{description}»"
                )
            notes.append("Ты отправила стикер " + ", ".join(parts) + ".")
        if incoming:
            quote = content.strip()
            quote = quote[:40] + "…" if len(quote) > 40 else quote
            who = ", ".join(
                f"{item.get('emoji', '')} ({item.get('from', 'кто-то')})"
                for item in incoming
            )
            target = f"твоё сообщение «{quote}»" if quote else "твоё сообщение"
            notes.append(f"На {target} поставили реакции: {who}.")
        output.append(
            {
                "role": "system",
                "content": " ".join(notes) + " (это действия в чате, не текст).",
            }
        )
    return output


def renumber_sids(entries: list) -> None:
    """Renumber reply handles after history summarization."""
    sequence = 0
    for message in entries:
        if message.get("role") == "user" and message.get("sid") is not None:
            sequence += 1
            message["sid"] = sequence


def extract_plain_text(content) -> str:
    """Extract user text from a structured persisted history entry."""
    if isinstance(content, list):
        content = next(
            (
                part.get("text", "")
                for part in content
                if part.get("type") == "text"
            ),
            "",
        )
    if not isinstance(content, str):
        return ""

    message_match = re.search(r"\[Message:\s*(.*?)\]", content, flags=re.DOTALL)
    if message_match:
        return strip_tiktok_urls(message_match.group(1).strip())

    text = re.sub(
        r"\[(?:Image description|Video description|Context from memory|User|Time|"
        r"Reply to|Quoted message|Selected quote|Forwarded from [^]]+):"
        r"(?:[^\[\]]|\[(?!Message:))*?\]",
        "",
        content,
    )
    return strip_tiktok_urls(text.strip())
