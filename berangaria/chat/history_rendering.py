"""Pure conversion between persisted chat history and provider messages."""

import re

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


def render_history_for_api(history: list) -> list:
    """Render persisted history into an ephemeral provider payload."""
    mid_to_sid = _build_mid_to_sid(history)
    output = []
    for message in history:
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, str) and content:
            content = strip_tiktok_urls(content)
        sid = message.get("sid")
        if sid is not None and role == "user":
            content = f"[#{sid}] {content}"

        reactions = message.get("reactions") if role == "assistant" else None
        incoming = message.get("incoming_reactions") if role == "assistant" else None
        stickers = message.get("stickers") if role == "assistant" else None
        voices = message.get("voices") if role == "assistant" else None
        if not (reactions or incoming or stickers or voices):
            output.append({"role": role, "content": content})
            continue

        if content and not voices:
            output.append({"role": "assistant", "content": content})
        elif content and voices and (reactions or stickers or incoming):
            output.append({"role": "assistant", "content": content})

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
        if voices:
            parts = []
            for voice in voices:
                spoken = (voice.get("text") or "").strip()
                if not spoken and content:
                    spoken = content.strip()
                if len(spoken) > 80:
                    spoken = spoken[:80] + "…"
                emotion = voice.get("emotion")
                parts.append(f"[{emotion}] «{spoken}»" if emotion else f"«{spoken}»")
            notes.append("Ты отправила голосовое " + ", ".join(parts) + ".")
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
        r"Reply to|Quoted message|Forwarded from [^]]+):"
        r"(?:[^\[\]]|\[(?!Message:))*?\]",
        "",
        content,
    )
    return strip_tiktok_urls(text.strip())
