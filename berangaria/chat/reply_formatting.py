"""Pure Telegram reply formatting and cleanup helpers."""

import re

from berangaria.core.utils import strip_internal_tags


def markdown_to_html(text: str) -> str:
    """Convert the Markdown subset used by replies to Telegram HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"```(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Triple markers must be handled before double/single markers.
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"___(.+?)___", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def strip_markdown(text: str) -> str:
    """Remove reply Markdown while keeping readable fallback text."""
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"~~(.+?)~~", r"\1", text, flags=re.DOTALL)
    return re.sub(
        r"(?<!\w)_{1,3}(.+?)_{1,3}(?!\w)",
        r"\1",
        text,
        flags=re.DOTALL,
    )


def split_for_telegram(text: str, limit: int = 4096) -> list[str]:
    """Split text within Telegram's UTF-16 code-unit limit."""

    def utf16_len(value: str) -> int:
        return len(value.encode("utf-16-le")) // 2

    if utf16_len(text) <= limit:
        return [text]

    chunks: list[str] = []
    rest = text
    while rest:
        if utf16_len(rest) <= limit:
            chunks.append(rest)
            break
        low, high = 1, len(rest)
        while low < high:
            mid = (low + high + 1) // 2
            if utf16_len(rest[:mid]) <= limit:
                low = mid
            else:
                high = mid - 1
        cut = low
        window = rest[:cut]
        for separator in ("\n\n", "\n", " "):
            position = window.rfind(separator)
            if position > cut * 0.6:
                cut = position + len(separator)
                break
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return [chunk for chunk in chunks if chunk]


def is_parse_error(error: BaseException) -> bool:
    """Return whether Telegram rejected markup rather than delivery itself."""
    text = str(error).lower()
    return "parse" in text or "entity" in text or "entities" in text or "tag" in text


_SILENCE_RE = re.compile(
    r"^[\s.…\-—–·*\"'()]*"
    r"(?:молчу|молчит|молчание|промолч\w*|ничего\s+не\s+(?:скажу|отвечу)|"
    r"без\s+комментари\w*|воздержусь|пропущу)?"
    r"[\s.…\-—–·*\"'()!?]*$",
    re.IGNORECASE,
)


def clean_reply(reply: str) -> str:
    """Remove internal tokens and normalize model silence placeholders."""
    reply = strip_internal_tags(reply)
    if reply.endswith(".") and not reply.endswith("..."):
        reply = reply[:-1]
    if _SILENCE_RE.match(reply):
        return ""
    return reply
