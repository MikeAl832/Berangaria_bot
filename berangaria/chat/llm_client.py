import re
import logging
import asyncio
import copy
import random
import httpx
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from berangaria.config import (
    CHAT_API_URL, SUMMARY_INTERVAL, VISION_MODE, MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, MODEL, GENERATION_PARAMS, FACTUAL_TEMPERATURE, FULL_DEBUG_LOGS,
    PRICE_PROMPT_CACHE_MISS, PRICE_PROMPT_CACHE_HIT, PRICE_PROMPT_CACHE_WRITE,
    PRICE_COMPLETION,
    MEMORY_SEARCH_LIMIT, MEMORY_MIN_SCORE, MEMORY_MAX_CHARS,
    MEMORY_QUERY_MIN_CHARS, MEMORY_QUERY_RECENT_MESSAGES, MAX_API_RETRIES,
    MAX_TOOL_ROUNDS, STREAMING_ENABLED, STREAM_UPDATE_INTERVAL_SECONDS,
    STREAM_PREVIEW_MIN_CHARS,
    MULTI_MESSAGE_DELAY_MIN, MULTI_MESSAGE_DELAY_MAX, MULTI_MESSAGE_DELAY_TOTAL_CAP,
    MULTI_MESSAGE_CHARS_PER_SEC, chat_api_headers, apply_chat_routing,
)
from berangaria.prompts import SYSTEM_PROMPT, VISION_PROMPT_SUFFIX
from berangaria.core.state import histories, chat_tokens, api_call_count, get_history_lock, touch_activity, save_history
from berangaria.memory import store as memory_store
from berangaria.core import state
from berangaria.tools.schemas import TOOLS
from berangaria.tools.dispatch import ToolTurn, dispatch_tool_call
from berangaria.chat.streaming import TelegramStreamPreview, stream_chat_completion
from berangaria.core.utils import (
    now_local, is_low_signal_user_text, strip_tiktok_urls, strip_internal_tags,
)

logger = logging.getLogger(__name__)


def _message_reasoning_len(message: dict) -> int:
    """Length of provider reasoning, whether DeepSeek- or OpenAI-shaped."""
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        return len(reasoning_content)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return len(reasoning)
    if isinstance(reasoning, dict):
        text = reasoning.get("content") or reasoning.get("text") or ""
        return len(text) if isinstance(text, str) else 0
    return 0


def _estimate_request_cost(
    usage: dict,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
) -> float:
    """Prefer the provider's billed `usage.cost`; otherwise estimate from prices."""
    billed = usage.get("cost")
    if billed is not None:
        try:
            return float(billed)
        except (TypeError, ValueError):
            pass

    cached = max(0, int(cached_tokens or 0))
    written = max(0, int(cache_write_tokens or 0))
    uncached = max(0, int(prompt_tokens or 0) - cached - written)
    return (
        (uncached / 1_000_000) * PRICE_PROMPT_CACHE_MISS
        + (cached / 1_000_000) * PRICE_PROMPT_CACHE_HIT
        + (written / 1_000_000) * PRICE_PROMPT_CACHE_WRITE
        + (int(completion_tokens or 0) / 1_000_000) * PRICE_COMPLETION
    )


class ReplyDeliveryError(RuntimeError):
    """Финальный ответ не был подтверждён Telegram и ход нельзя коммитить."""


def markdown_to_html(text: str) -> str:
    """
    Конвертирует базовый Markdown в HTML для Telegram.
    Поддерживает: жирный, курсив, код, ссылки.
    """
    # Экранируем HTML символы
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Блоки кода ```code```
    text = re.sub(r'```(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    
    # Инлайн код `code`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Жирный курсив ***text*** или ___text___
    # ВАЖНО: тройные ДО двойных/одинарных, иначе ** «съест» *** и сломает разметку
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'___(.+?)___', r'<b><i>\1</i></b>', text)

    # Жирный текст **text** или __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Курсив *text* или _text_ (но не внутри слов)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<i>\1</i>', text)

    # Зачёркнутый ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    
    # Ссылки [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', text)

    return text


def strip_markdown(text: str) -> str:
    """
    Убирает markdown-разметку, оставляя читаемый текст.
    Используется как фолбэк, если HTML не распарсился Telegram'ом.
    """
    text = re.sub(r'```(.*?)```', r'\1', text, flags=re.DOTALL)        # блоки кода
    text = re.sub(r'`([^`]+)`', r'\1', text)                           # инлайн код
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1 (\2)', text)       # ссылки → текст (url)
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text, flags=re.DOTALL)  # *, **, ***
    text = re.sub(r'~~(.+?)~~', r'\1', text, flags=re.DOTALL)          # зачёркнутый
    text = re.sub(r'(?<!\w)_{1,3}(.+?)_{1,3}(?!\w)', r'\1', text, flags=re.DOTALL)  # _, __, ___
    return text


_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _current_time_str() -> str:
    """Формирует строку с текущей датой и временем суток для системного промпта (МСК)."""
    now = now_local()
    time_str = f"Today is {_DAYS[now.weekday()]}, {now.day} {_MONTHS[now.month-1]} {now.year} year. "

    if 5 <= now.hour < 12:
        time_of_day = "morning"
    elif 12 <= now.hour < 17:
        time_of_day = "daytime"
    elif 17 <= now.hour < 23:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    time_str += f"Times of Day: {time_of_day}."
    return time_str


def _build_system_prompt() -> str:
    """Собирает полный системный промпт: база + vision (если включён) + текущее время."""
    system_prompt = SYSTEM_PROMPT
    if VISION_MODE:
        system_prompt += VISION_PROMPT_SUFFIX
    system_prompt += f"\n\n=== CURRENT TIME ===\n{_current_time_str()}\n"
    return system_prompt


def _build_sid_map(history: list) -> dict:
    """Карта {sid -> telegram message_id} по текущей истории (для reply/react по [#N])."""
    return {
        m["sid"]: m.get("mid")
        for m in history
        if m.get("role") == "user" and m.get("sid") is not None
    }


def _build_mid_to_sid(history: list) -> dict:
    """Обратная карта telegram mid → актуальный [#sid] (sid после renumber всегда свежий)."""
    out = {}
    for m in history or []:
        if m.get("role") == "user" and m.get("mid") is not None and m.get("sid") is not None:
            out[m["mid"]] = m["sid"]
    return out


def _format_reaction_note_part(r: dict, mid_to_sid: dict) -> str:
    """
    Текст одной своей реакции для system-ноты.
    [#N] резолвим по on_mid из живой истории — после суммаризации/renumber
    номер всегда актуальный; если сообщения уже нет — только цитата.
    """
    emoji = r.get("emoji") or ""
    on = (r.get("on") or "").strip()
    mid = r.get("on_mid")
    sid = mid_to_sid.get(mid) if mid is not None else None
    if sid is not None:
        if on:
            return f"{emoji} на [#{sid}] «{on}»"
        return f"{emoji} на [#{sid}]"
    if on:
        return f"{emoji} на «{on}»"
    return emoji


def _render_history_for_api(history: list) -> list:
    """
    Готовит копию истории для отправки в API.
    - В начало каждого user-сообщения подставляет стабильный reply-хэндл [#sid].
    - Выкидывает служебные ключи (sid/mid), которых не должно быть в payload.
    Сам тег [#N] нигде не хранится — он живёт только в этой эфемерной копии,
    поэтому история, память и суммаризация остаются чистыми, а префикс стабилен (cache hit).
    """
    mid_to_sid = _build_mid_to_sid(history)
    out = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, str) and content:
            content = strip_tiktok_urls(content)
        sid = m.get("sid")
        if sid is not None and role == "user":
            content = f"[#{sid}] {content}"

        # Реакции (свои и входящие) отдаём ОТДЕЛЬНОЙ системной строкой, а не текстом ассистента:
        # так модель воспринимает это как факт-действие и не начинает печатать «(реакция…)»
        # в свои реплики. В историю/память/суммарайз попадают только структурные поля,
        # сама нота эфемерна — живёт лишь в этой копии (как тег [#N]).
        reactions = m.get("reactions") if role == "assistant" else None          # что бот поставил сам
        incoming = m.get("incoming_reactions") if role == "assistant" else None  # что поставили ему
        stickers = m.get("stickers") if role == "assistant" else None            # какие стикеры отправил
        voices = m.get("voices") if role == "assistant" else None                # голосовые этого хода
        if reactions or incoming or stickers or voices:
            # Typed assistant text (if any). Voice words live only in the action
            # note — same as stickers — so the model does not re-read them as a
            # normal typed reply plus a weird "you said this aloud" footer.
            if content and not voices:
                out.append({"role": "assistant", "content": content})
            elif content and voices and (reactions or stickers or incoming):
                # Rare hybrid row: keep non-voice prose if something else is
                # attached; pure voice turns keep content empty in storage.
                out.append({"role": "assistant", "content": content})
            notes = []
            if reactions:
                parts = [_format_reaction_note_part(r, mid_to_sid) for r in reactions]
                notes.append("Ты поставила реакцию " + ", ".join(parts) + ".")
            if stickers:
                parts = []
                for s in stickers:
                    d = (s.get('desc') or '').strip()
                    if len(d) > 80:
                        d = d[:80] + "…"
                    e = s.get('emotion')
                    parts.append(f"[{e}] «{d}»" if e else f"«{d}»")
                notes.append("Ты отправила стикер " + ", ".join(parts) + ".")
            if voices:
                # Parallel to stickers: "Ты отправила голосовое [emo] «слова»."
                parts = []
                for v in voices:
                    t = (v.get("text") or "").strip()
                    if not t and content:
                        t = content.strip()
                    if len(t) > 80:
                        t = t[:80] + "…"
                    e = v.get("emotion")
                    parts.append(f"[{e}] «{t}»" if e else f"«{t}»")
                notes.append("Ты отправила голосовое " + ", ".join(parts) + ".")
            if incoming:
                quote = content.strip()
                quote = (quote[:40] + "…") if len(quote) > 40 else quote
                who = ", ".join(f"{r.get('emoji', '')} ({r.get('from', 'кто-то')})" for r in incoming)
                target = f"твоё сообщение «{quote}»" if quote else "твоё сообщение"
                notes.append(f"На {target} поставили реакции: {who}.")
            out.append({
                "role": "system",
                "content": " ".join(notes) + " (это действия в чате, не текст).",
            })
        else:
            out.append({"role": role, "content": content})
    return out


def _renumber_sids(entries: list) -> None:
    """
    Перенумеровывает [#N] у user-сообщений с 1. Вызывается после суммаризации:
    старые сообщения ушли в резюме, оставшиеся свежие получают новые номера с #1.
    Бесплатно для кэша, т.к. суммаризация и так перестраивает префикс.
    """
    seq = 0
    for m in entries:
        if m.get("role") == "user" and m.get("sid") is not None:
            seq += 1
            m["sid"] = seq


# Плейсхолдеры, которыми модель «проговаривает» молчание вместо пустого ответа.
# Матчит сообщение целиком: только пунктуация/обёртки, либо короткая мета-фраза тишины.
_SILENCE_RE = re.compile(
    r"^[\s.…\-—–·*\"'()]*"
    r"(?:молчу|молчит|молчание|промолч\w*|ничего\s+не\s+(?:скажу|отвечу)|"
    r"без\s+комментари\w*|воздержусь|пропущу)?"
    r"[\s.…\-—–·*\"'()!?]*$",
    re.IGNORECASE,
)


def _split_for_telegram(text: str, limit: int = 4096) -> list[str]:
    """Режет текст на куски в пределах лимита Telegram.

    Лимит Telegram считается в UTF-16 code units, а не в символах Python:
    эмодзи вне BMP занимают две единицы, поэтому нарезка по len() может дать
    чанк, который Telegram отвергнет. Границы по возможности ставим по строкам
    и пробелам, чтобы не рвать слово пополам.
    """
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
        # Двоичный поиск максимального префикса, влезающего в лимит.
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


def _multi_message_delay_seconds(text: str, *, slept_total: float = 0.0) -> float:
    """Пауза перед следующим bubble: длина + jitter, с общим потолком на ход."""
    remaining = MULTI_MESSAGE_DELAY_TOTAL_CAP - slept_total
    if remaining <= 0:
        return 0.0
    base = len(text or "") / MULTI_MESSAGE_CHARS_PER_SEC
    delay = max(MULTI_MESSAGE_DELAY_MIN, min(MULTI_MESSAGE_DELAY_MAX, base))
    delay *= random.uniform(0.85, 1.15)
    return max(0.0, min(delay, remaining))


def _is_parse_error(error: BaseException) -> bool:
    """Отличает ошибку HTML-разметки от любой другой ошибки Telegram.

    Фолбэк на чистый текст осмыслен только когда Telegram отверг саму разметку.
    Сетевые и неоднозначные ошибки повторять нельзя: сообщение может быть уже
    создано, и повтор даст дубль.
    """
    text = str(error).lower()
    return "parse" in text or "entity" in text or "entities" in text or "tag" in text


def _clean_reply(reply: str) -> str:
    """Чистит ответ модели от служебных токенов и лишней финальной точки.

    Эмодзи намеренно НЕ вырезаются: промпт отговаривает модель от их использования,
    но когда эмодзи — сам ответ (огрызок, ответная реакция), он должен пройти.
    """
    # Срез служебных тегов общий с streaming-превью (см. utils).
    reply = strip_internal_tags(reply)
    if reply.endswith('.') and not reply.endswith('...'):
        reply = reply[:-1]
    # Модель иногда «проговаривает» молчание (… / — / «промолчу» / «(молчит)») вместо
    # пустого ответа. Сводим такие плейсхолдеры к пустой строке → уходит в ветку тишины.
    if _SILENCE_RE.match(reply):
        return ''
    return reply


def _extract_plain_text(content) -> str:
    """
    Извлекает чистый текст пользователя из сообщения для поиска по памяти.
    Убирает служебные теги, оставляя только содержимое [Message: ...].
    """
    if isinstance(content, list):
        content = next((p.get('text', '') for p in content if p.get('type') == 'text'), '')
    if not isinstance(content, str):
        return ''

    # Ищем [Message: ...] — самый частый случай
    msg_match = re.search(r'\[Message:\s*(.*?)\]', content, flags=re.DOTALL)
    if msg_match:
        return strip_tiktok_urls(msg_match.group(1).strip())

    # Если нет Message, убираем служебные блоки
    text = re.sub(
        r'\[(?:Image description|Video description|Context from memory|User|Time|Reply to|Quoted message|Forwarded from [^]]+):(?:[^\[\]]|\[(?!Message:))*?\]',
        '',
        content
    )

    return strip_tiktok_urls(text.strip())


def _is_meaningful_memory_query(text: str) -> bool:
    """Отсекает короткие/служебные/URL-only реплики, которые портят retrieval."""
    return not is_low_signal_user_text(text, min_alnum=MEMORY_QUERY_MIN_CHARS)


def _build_memory_search_query(history: list, user_name: str) -> str:
    """
    Берёт последние содержательные user-сообщения вместо слепого поиска по
    "Ладно" или "(сообщение без текста)".
    """
    candidates: list[str] = []
    for entry in reversed(history or []):
        if entry.get("role") != "user":
            continue
        plain = _extract_plain_text(entry.get("content", ""))
        if not _is_meaningful_memory_query(plain):
            continue
        candidates.append(plain)
        if len(candidates) >= MEMORY_QUERY_RECENT_MESSAGES:
            break

    if candidates:
        return "\n".join(reversed(candidates))[:1000]

    return user_name if _is_meaningful_memory_query(user_name) else ""


def _build_memory_relevance_query(history: list, user_name: str) -> str:
    """Возвращает только последнюю содержательную тему для fail-closed фильтра."""
    for entry in reversed(history or []):
        if entry.get("role") != "user":
            continue
        plain = _extract_plain_text(entry.get("content", ""))
        if _is_meaningful_memory_query(plain):
            return plain[:1000]
    return user_name if _is_meaningful_memory_query(user_name) else ""


_MEMORY_TERM_RE = re.compile(r"[^\W_]{4,}", flags=re.UNICODE)
_MEMORY_RECALL_RE = re.compile(
    r"\b(?:что|чего)\s+ты\s+(?:обо?\s+мне|про\s+меня)\s+помни\w*|"
    r"\bчто\s+ты\s+знаешь\s+(?:обо?\s+мне|про\s+меня)|"
    r"\b(?:расскажи|напомни)\w*(?:\s+мне)?\s+"
    r"(?:обо?\s+мне|про\s+меня)",
    flags=re.IGNORECASE,
)
_MEMORY_STOP_WORDS = {
    "какой", "какая", "какие", "который", "которая", "которые",
    "меня", "мне", "тебя", "тебе", "твой", "твоя", "свой", "своя",
    "пользователь", "использует", "сейчас", "сегодня", "просто",
    "скажи", "назови", "пожалуйста", "about", "what", "which", "user",
}


def _memory_terms(text: str) -> set[str]:
    return {
        token
        for token in _MEMORY_TERM_RE.findall((text or "").casefold())
        if token not in _MEMORY_STOP_WORDS
    }


def _is_general_memory_recall(query: str) -> bool:
    return bool(_MEMORY_RECALL_RE.search(query or ""))


def _memory_fact_matches_query(fact: str, query: str) -> bool:
    if not query or _is_general_memory_recall(query):
        return True
    fact_terms = _memory_terms(fact)
    query_terms = _memory_terms(query)
    return any(
        fact_term == query_term
        or (
            len(fact_term) >= 5
            and len(query_term) >= 5
            and fact_term[:5] == query_term[:5]
        )
        for fact_term in fact_terms
        for query_term in query_terms
    )


def _approved_memory_recall_results(scope: str) -> dict:
    """Возвращает только одобренный SQLite-реестр для общего recall-запроса."""
    facts = state.list_memory_facts(scope)[-MEMORY_SEARCH_LIMIT:]
    return {
        "results": [
            {"id": fact.mem0_id, "memory": fact.fact, "score": 1.0}
            for fact in facts
        ]
    }


def _format_memory_block(mem_results: dict, query: str = "") -> str:
    """
    Формирует компактный блок памяти с фильтрацией по релевантности.
    Возвращает готовый текст или пустую строку.
    """
    results = (mem_results or {}).get('results') or []
    if not results:
        return ''

    # Сортируем по релевантности
    results = sorted(results, key=lambda item: item.get('score') or 0.0, reverse=True)

    lines = []
    total = 0
    for item in results:
        if item.get('score', 0.0) < MEMORY_MIN_SCORE:
            continue
        fact = (item.get('memory') or '').strip()
        if not fact:
            continue
        if not _memory_fact_matches_query(fact, query):
            continue
        line = f"- {fact}"
        if total + len(line) > MEMORY_MAX_CHARS:
            break
        lines.append(line)
        total += len(line)
        if len(lines) >= MEMORY_SEARCH_LIMIT:
            break

    return "\n".join(lines)


def _count_memory_block_facts(mem_text: str) -> int:
    """Считает фактически отформатированные строки фактов для логов."""
    return sum(1 for line in mem_text.splitlines() if line.startswith("- "))


def _filter_approved_memory_results(mem_results: dict, scope: str) -> dict:
    """Fail-closed: сверяет scope, ID и точный текст с SQLite-реестром."""
    approved = {
        fact.mem0_id: fact.fact
        for fact in state.list_memory_facts(scope)
    }
    raw_results = (mem_results or {}).get("results") or []
    results = []
    seen_ids: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        memory_id = str(item.get("id") or "")
        memory_text = item.get("memory")
        if (
            not memory_id
            or memory_id in seen_ids
            or approved.get(memory_id) != memory_text
        ):
            continue
        seen_ids.add(memory_id)
        results.append(item)
    return {"results": results}


async def summarize_history(history: list) -> list:
    to_summarize = history[:-SUMMARY_INTERVAL]
    # SID и служебные поля меняем только в независимой копии. При ошибке API
    # исходная история должна остаться побитово неизменной.
    keep_recent = copy.deepcopy(history[-SUMMARY_INTERVAL:])

    if not to_summarize:
        return history

    # Старые сообщения уходят в резюме (их [#N] исчезают), у оставшихся свежих
    # сбрасываем нумерацию с #1, чтобы номера не росли бесконечно.
    _renumber_sids(keep_recent)

    # Содержательные реплики + слова из голосовых (content у voice-only пустой,
    # как у стикеров). Чистые реакции/стикеры без текста в резюме не нужны.
    summary_lines: list[str] = []
    for m in to_summarize:
        role = m.get("role") or "assistant"
        raw = m.get("content")
        if isinstance(raw, str) and strip_tiktok_urls(raw).strip():
            summary_lines.append(f"{role}: {strip_tiktok_urls(raw)}")
            continue
        for v in m.get("voices") or []:
            if not isinstance(v, dict):
                continue
            spoken = (v.get("text") or "").strip()
            if spoken:
                summary_lines.append(f"{role}: {strip_tiktok_urls(spoken)}")
    text_to_summarize = "\n".join(summary_lines)
    if not text_to_summarize.strip():
        logger.warning("📝 [yellow]Суммаризация пропущена:[/] нет текстового содержимого для сжатия")
        return history

    # Thinking on (default high effort) — quality of multi-turn compression depends on CoT.
    # Reasoning tokens share the max_tokens budget with content; too small a budget
    # yields content=null while reasoning_content is full. Keep a long client timeout
    # (was 30s → false timeouts under thinking).
    summary_payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Напиши ТЕХНИЧЕСКОЕ РЕЗЮМЕ диалога на русском:"
                    "Сожми этот диалог в КРАТКОЕ резюме на русском языке. "
                    "Пиши ТОЛЬКО суть, без вводных фраз. "
                    "Обязательно сохрани: имена, цифры, модели (например, RTX 5070 Ti), "
                    "технические характеристики, решения и важные факты. "
                    "НЕ пиши 'Пользователь сказал...', 'Собеседник ответил...' — просто перескажи факты."
                )
            },
            {
                "role": "user",
                "content": text_to_summarize
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.9,
        "reasoning": {"effort": "high"},
    }
    apply_chat_routing(summary_payload)

    try:
        headers = chat_api_headers()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(CHAT_API_URL, json=summary_payload, headers=headers)
            logger.info(f"Ответ сумморизации: [cyan]{response.status_code}[/]")
            response.raise_for_status()
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            # content nullable в API. При reasoning и исчерпанном max_tokens сюда
            # часто null — раньше re.sub(None) давал TypeError → «Ошибка суммаризации».
            raw = message.get("content")
            if not isinstance(raw, str) or not raw.strip():
                reasoning_len = _message_reasoning_len(message)
                raise ValueError(
                    "пустой content в ответе суммаризации"
                    + (f" (есть reasoning, {reasoning_len} символов)" if reasoning_len else "")
                )

            summary = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if not summary:
                raise ValueError("резюме пустое после очистки thinking-тегов")

            logger.info(f"📝 Резюме истории получено ({len(summary)} символов)")

            if FULL_DEBUG_LOGS:
                logger.debug(f"Содержание:\n{summary}")

            return [{"role": "user", "content": f"[Previous conversation summary: {summary}]"}] + keep_recent

    except Exception as e:
        logger.error(f"❌ [red]Ошибка суммаризации:[/] {e}")
        return history


async def send_llm_request(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: str,
    history: list, user_name: str, user_id: int, mentioned: bool = False):

    # Автосуммаризация при достижении 85% от лимита токенов
    context_threshold = int(MAX_CONTEXT_TOKENS * 0.85)
    if chat_tokens.get(key, 0) > context_threshold:
        logger.info(f"📝 [yellow]Автосуммаризация[/] для key={key}")
        history = await summarize_history(history)
        async with get_history_lock(key):
            histories[key] = history
            save_history(key)

    system_prompt = _build_system_prompt()
    # В payload подставляем reply-хэндлы [#N] (только в копию, история остаётся чистой)
    payload_messages = [{"role": "system", "content": system_prompt}] + _render_history_for_api(history)
    sid_to_mid = _build_sid_map(history)

    try:
        # Валидация ключа для безопасности
        if not state.is_valid_memory_scope(key):
            logger.warning(f"⚠️ [yellow]Невалидный ключ памяти:[/] {key}")
        else:
            query = _build_memory_search_query(history, user_name)
            if not query:
                if FULL_DEBUG_LOGS:
                    logger.debug(f"🔍 Mem0 поиск пропущен: нет содержательного query (scope={key})")
            else:
                relevance_query = _build_memory_relevance_query(history, user_name)
                if _is_general_memory_recall(relevance_query):
                    approved_results = _approved_memory_recall_results(key)
                    results_count = len(approved_results["results"])
                    if FULL_DEBUG_LOGS:
                        logger.debug(f"🔍 Память: общий recall из SQLite (scope={key})")
                elif memory_store.memory:
                    if FULL_DEBUG_LOGS:
                        logger.debug(f"🔍 Mem0 поиск: query='{query[:80]}', scope={key}")

                    # Уменьшен таймаут до 15 секунд для быстрого ответа
                    mem_results = await asyncio.wait_for(
                        asyncio.to_thread(
                            memory_store.memory.search,
                            query,
                            filters={"user_id": key},
                            limit=MEMORY_SEARCH_LIMIT
                        ),
                        timeout=15.0
                    )

                    results_count = len(mem_results.get('results', []))
                    approved_results = _filter_approved_memory_results(mem_results, key)
                else:
                    approved_results = {"results": []}
                    results_count = 0
                    if FULL_DEBUG_LOGS:
                        logger.debug(f"🔍 Mem0 поиск пропущен: хранилище недоступно (scope={key})")

                mem_text = _format_memory_block(
                    approved_results,
                    query=relevance_query,
                )

                if mem_text and payload_messages[-1]["role"] == "user":
                    last_content = payload_messages[-1]["content"]
                    payload_messages[-1] = {
                        "role": "user",
                        "content": f"{last_content}\n\n[Context from memory:\n{mem_text}\n]"
                    }
                    facts_count = _count_memory_block_facts(mem_text)

                    # Краткий лог для INFO, детальный для DEBUG
                    logger.info(f"🧠 Память: найдено {results_count} → загружено {facts_count} фактов ({len(mem_text)} символов)")

                    if FULL_DEBUG_LOGS:
                        logger.debug(f"📝 Факты:\n{mem_text}")

    except asyncio.TimeoutError:
        logger.warning(f"⚠️ [yellow]Память: таймаут поиска (15s), продолжаем без неё[/] scope={key}")
    except Exception as e:
        logger.error(f"⚠️ [red]Ошибка получения памяти:[/] {e}")

    # Мутируемое состояние хода (статусная плашка, реакции, стикеры, pending_reply) —
    # см. tool_handlers.ToolTurn. Живёт весь retry-цикл.
    turn = ToolTurn()
    used_tool = False  # после вызова инструмента (поиск/ссылка) отвечаем с пониженной температурой

    async def _request_completion(client, payload, headers):
        if not STREAMING_ENABLED:
            return await client.post(CHAT_API_URL, json=payload, headers=headers)

        preview = TelegramStreamPreview(
            update,
            context,
            mentioned=mentioned,
            status_message=turn.status_message,
            interval_seconds=STREAM_UPDATE_INTERVAL_SECONDS,
            min_chars=STREAM_PREVIEW_MIN_CHARS,
        )
        try:
            return await stream_chat_completion(
                client,
                CHAT_API_URL,
                payload=payload,
                headers=headers,
                on_content=preview.publish,
            )
        finally:
            # Если preview создал групповое сообщение, tool handlers и финальная
            # доставка должны переиспользовать именно его.
            turn.status_message = preview.status_message

    async def _delete_turn_status():
        if turn.status_message is None:
            return
        try:
            await turn.status_message.delete()
        except Exception:
            pass
        finally:
            turn.status_message = None

    async def _deliver(text: str, target_mid, status_msg):
        """
        Отправляет text в чат.
        target_mid is not None — реплаем на это сообщение; None — обычным сообщением без reply.
        Переиспользует статусную плашку поиска только если ответ идёт на текущее сообщение.
        HTML с фолбэком на чистый текст; длинное режет по 4096.
        Возвращает message_id отправленного ботом сообщения (для привязки входящих реакций)
        или None. Для длинного ответа — id первого чанка.
        """
        reply_html = markdown_to_html(text)
        reply_plain = strip_markdown(text)
        chat_id = update.effective_chat.id
        thread_id = getattr(update.message, "message_thread_id", None)

        # Статусную плашку (она висит реплаем на триггере) можно дописать только
        # если итоговый ответ адресован тому же триггерному сообщению.
        if status_msg is not None:
            if target_mid == update.message.message_id and len(reply_html) <= 4096:
                try:
                    await status_msg.edit_text(reply_html, parse_mode="HTML")
                    return status_msg.message_id  # отредактированная плашка и есть сообщение бота
                except BadRequest as e:
                    # Правка идемпотентна (тот же message_id), поэтому сетевую
                    # ошибку здесь пережить можно — но только не молча: при
                    # неоднозначном таймауте плашка могла уже стать ответом.
                    logger.warning(f"⚠️ [yellow]Правка статуса с HTML не прошла:[/] {e}")
                except Exception as e:
                    logger.error(
                        f"❌ [red]Правка статуса оборвалась неоднозначно:[/] {e}"
                    )
                    raise ReplyDeliveryError(
                        "Telegram не подтвердил правку статусного сообщения"
                    ) from e
            try:
                await status_msg.delete()
            except Exception:
                pass

        async def _raw(body: str, html: bool):
            kw = {"chat_id": chat_id, "text": body}
            if thread_id is not None:
                kw["message_thread_id"] = thread_id
            if target_mid is not None:
                kw["reply_to_message_id"] = target_mid
                kw["allow_sending_without_reply"] = True  # если целевое удалено — шлём без реплая
            if html:
                kw["parse_mode"] = "HTML"
            sent = await context.bot.send_message(**kw)
            return sent.message_id

        if len(reply_html) <= 4096:
            try:
                return await _raw(reply_html, True)
            except BadRequest as e:
                # Ловим ТОЛЬКО ошибку разметки. Раньше здесь стоял except
                # Exception, из-за чего TimedOut (дефолтный read_timeout PTB — 5 с)
                # приводил ко второй отправке уже созданного Telegram сообщения:
                # пользователь получал ответ дважды, а в историю попадал mid копии.
                if not _is_parse_error(e):
                    raise
                logger.warning(f"⚠️ [yellow]HTML не распарсился, отправляю как текст:[/] {e}")
                return await _raw(reply_plain, False)
        else:
            # Длинный ответ шлём чистым текстом, чтобы не порвать HTML-теги на границе чанка
            first_mid = None
            for chunk in _split_for_telegram(reply_plain):
                kw = {"chat_id": chat_id, "text": chunk}
                if thread_id is not None:
                    kw["message_thread_id"] = thread_id
                if first_mid is None and target_mid is not None:
                    # Реплай ставим только на первый чанк — остальные идут следом.
                    kw["reply_to_message_id"] = target_mid
                    kw["allow_sending_without_reply"] = True
                try:
                    sent = await context.bot.send_message(**kw)
                except Exception as e:
                    if first_mid is None:
                        raise
                    # Часть ответа уже в чате. Рвать ход нельзя: пользователь его
                    # видел, и повтор с начала дал бы дубль. Сохраняем то, что дошло.
                    logger.error(
                        f"❌ [red]Длинный ответ оборвался после первого чанка:[/] {e}"
                    )
                    return first_mid
                if first_mid is None:
                    first_mid = sent.message_id
            return first_mid

    async def _deliver_multi(messages: list[str], target_mid, status_msg):
        """
        Шлёт 2+ коротких сообщений с typing и sleep между ними.
        Reply (если есть) — только на первый bubble; mid для истории — первый.
        Возвращает (first_mid, delivered_texts) — delivered может быть короче
        при partial fail после первого send.
        """
        cleaned: list[str] = []
        for raw in messages or []:
            try:
                piece = _clean_reply(raw)
            except Exception:
                piece = (raw or "").strip()
            if piece:
                cleaned.append(piece)
        if not cleaned:
            if status_msg is not None:
                try:
                    await status_msg.delete()
                except Exception:
                    pass
            return None, []

        if len(cleaned) == 1:
            mid = await _deliver(cleaned[0], target_mid, status_msg)
            return mid, cleaned

        # Multi: status banner cannot become "the" reply for a whole series.
        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                pass

        chat_id = update.effective_chat.id
        thread_id = getattr(update.message, "message_thread_id", None)
        first_mid = None
        delivered: list[str] = []
        slept_total = 0.0

        for index, text in enumerate(cleaned):
            if index > 0:
                delay = _multi_message_delay_seconds(text, slept_total=slept_total)
                if delay > 0:
                    try:
                        await update.message.chat.send_action(action="typing")
                    except Exception:
                        pass
                    await asyncio.sleep(delay)
                    slept_total += delay

            reply_html = markdown_to_html(text)
            reply_plain = strip_markdown(text)
            use_html = len(reply_html) <= 4096
            body = reply_html if use_html else reply_plain

            async def _send(body: str, html: bool):
                kw = {"chat_id": chat_id, "text": body}
                if thread_id is not None:
                    kw["message_thread_id"] = thread_id
                if first_mid is None and target_mid is not None:
                    kw["reply_to_message_id"] = target_mid
                    kw["allow_sending_without_reply"] = True
                if html:
                    kw["parse_mode"] = "HTML"
                return await context.bot.send_message(**kw)

            try:
                try:
                    sent = await _send(body, use_html)
                except BadRequest as e:
                    if use_html and _is_parse_error(e):
                        logger.warning(
                            f"⚠️ [yellow]HTML multi-bubble не распарсился, plain:[/] {e}"
                        )
                        sent = await _send(reply_plain, False)
                    else:
                        raise
            except Exception as e:
                if first_mid is None:
                    raise
                logger.error(
                    f"❌ [red]Серия сообщений оборвалась после {len(delivered)}:[/] {e}"
                )
                return first_mid, delivered

            if first_mid is None:
                first_mid = sent.message_id
            delivered.append(text)

        return first_mid, delivered

    async def _save_assistant(text: str):
        """
        Пишет ход бота в историю. Если за ход были реакции — прикрепляет их
        к этой же записи (поле reactions), чтобы модель помнила, что среагировала.
        Реакция-без-текста сохраняется как пустой content + reactions.
        Дописывает в хвост — префикс не меняется, cache hit сохраняется.
        Возвращает созданную запись (чтобы потом проставить ей mid) или None.
        """
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
            # Like stickers: empty content + structured field. Spoken words are
            # rendered only in the system action note (and used in summarization).
            entry["voices"] = list(turn.voices_made)
        async with get_history_lock(key):
            history.append(entry)
            histories[key] = history
            touch_activity(key)
            save_history(key)
        return entry

    async def _remember_bot_mid(entry, sent_mid):
        """Проставляет mid отправленного сообщения на assistant-запись — чтобы потом
        привязать к ней входящие реакции. mid в payload не рендерится → кэш не трогает."""
        if entry is None or not sent_mid:
            return
        async with get_history_lock(key):
            entry["mid"] = sent_mid
            save_history(key)

    async with httpx.AsyncClient(timeout=600.0) as client:
        if FULL_DEBUG_LOGS:
            # В DEBUG режиме показываем полную структуру с содержимым
            logger.debug("[cyan]" + "=" * 80 + "[/]")
            logger.debug("[bright_green]📤 ЗАПРОС К МОДЕЛИ:[/]")
            logger.debug("[cyan]" + "=" * 80 + "[/]")
            for i, msg in enumerate(payload_messages, 1):
                role = msg['role']
                content = str(msg.get('content', ''))
                
                # Цвет в зависимости от роли
                role_color = {
                    'system': 'magenta',
                    'user': 'cyan',
                    'assistant': 'green'
                }.get(role, 'white')
                
                logger.debug(f"\n[yellow][{i}][/] Role: [{role_color}]{role.upper()}[/]")
                logger.debug(f"Length: [dim]{len(content)} символов[/]")
                logger.debug(f"[{role_color}]Content:[/]")
                logger.debug(f"[dim]{content[:2000]}{'...' if len(content) > 2000 else ''}[/]")
                logger.debug("[dim]" + "-" * 80 + "[/]")
            logger.debug("[cyan]" + "=" * 80 + "[/]")

        forced_answer_nudge = False  # один раз подтолкнём ответить, если промолчала при прямом обращении

        api_failures = 0
        tool_rounds = 0
        while True:
            gen_params = dict(GENERATION_PARAMS)
            if used_tool:
                # Факты после поиска/чтения ссылки — холоднее, меньше выдумок
                gen_params["temperature"] = FACTUAL_TEMPERATURE

            payload = apply_chat_routing({
                "model": MODEL,
                "messages": payload_messages,
                "max_tokens": MAX_REPLY_TOKENS,
                "tools": TOOLS,
                **gen_params
            })

            try:
                headers = chat_api_headers()
                
                response = await _request_completion(client, payload, headers)

                if response.status_code == 400:
                    await _delete_turn_status()
                    async with get_history_lock(key):
                        histories[key] = []
                        save_history(key)
                    logger.error(f"[red]400:[/] {response.text}")
                    await update.message.reply_text("⚠️ История сброшена. Напишите ещё раз.")
                    return

                # Обработка rate limiting
                if response.status_code == 429:
                    api_failures += 1
                    if api_failures >= MAX_API_RETRIES:
                        await _delete_turn_status()
                        await update.message.reply_text("❌ API временно перегружен. Попробуйте позже.")
                        return
                    try:
                        retry_after = min(
                            60.0,
                            max(1.0, float(response.headers.get("Retry-After", 5))),
                        )
                    except (TypeError, ValueError):
                        retry_after = 5.0
                    logger.warning(
                        f"⚠️ [yellow]Rate limit (429), ждём {retry_after:g}s перед retry "
                        f"{api_failures}/{MAX_API_RETRIES}[/]"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code != 200:
                    logger.error(f"❌ [red]API error {response.status_code}:[/] {response.text[:200]}")
                    api_failures += 1
                    if api_failures < MAX_API_RETRIES:
                        await asyncio.sleep(2 ** (api_failures - 1))
                        continue
                    await _delete_turn_status()
                    await update.message.reply_text(f"❌ Ошибка API: {response.status_code}")
                    return

                data = response.json()
                choice = data['choices'][0]
                finish_reason = choice.get('finish_reason', '')
                message = choice['message']
                usage = data.get('usage', {})

                if usage:
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    prompt_details = usage.get('prompt_tokens_details', {})
                    cached_tokens = prompt_details.get('cached_tokens', 0)
                    cache_write_tokens = prompt_details.get('cache_write_tokens', 0)
                    
                    chat_tokens[key] = total_tokens
                    
                    logger.info(f"📊 Токены: запрос=[cyan]{prompt_tokens}[/] (кэш=[cyan]{cached_tokens}[/]), "
                                f"ответ=[cyan]{completion_tokens}[/], всего=[bright_green]{total_tokens}[/]")

                    total_cost = _estimate_request_cost(
                        usage,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached_tokens=cached_tokens,
                        cache_write_tokens=cache_write_tokens,
                    )
                    logger.info(f"💰 Стоимость запроса: [bright_green]${total_cost:.6f}[/]")
                
                if finish_reason == 'tool_calls' and message.get('tool_calls'):
                    tool_rounds += 1
                    if tool_rounds > MAX_TOOL_ROUNDS:
                        logger.error(f"❌ Превышен лимит tool-call раундов ({MAX_TOOL_ROUNDS})")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        await update.message.reply_text(
                            "❌ Не удалось завершить обработку инструментов. Попробуйте переформулировать запрос."
                        )
                        return
                    payload_messages.append(message)
                    # The cold temperature is only right where the reply retells
                    # facts that were looked up. It used to be switched on by ANY
                    # tool, so a sticker search or a reaction silently flattened the
                    # rest of the turn — punishing the model for exactly the
                    # behaviour the prompt is trying to encourage.
                    if any(
                        (tc.get("function") or {}).get("name") in ("web_search", "read_url")
                        for tc in message["tool_calls"]
                    ):
                        used_tool = True
                    turn.pending_reply = None  # (target_mid, text, sid) если модель выбрала reply_to_message
                    turn.pending_messages = None  # list[str] если send_messages

                    for tool_call in message['tool_calls']:
                        try:
                            await dispatch_tool_call(
                                turn, payload_messages, update, context,
                                tool_call, sid_to_mid, history,
                            )
                        except Exception as exc:
                            logger.error(f"❌ Ошибка инструмента: {exc}", exc_info=True)
                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", ""),
                                "content": f"Инструмент завершился ошибкой: {exc}",
                            })

                    # reply_to_message терминальный и приоритетный: если модель его вызвала,
                    # отправляем выбранный ответ и завершаем — без ещё одного витка к API
                    # и без дефолтного реплая ниже (двойной отправки не будет).
                    if turn.pending_reply is not None:
                        reply_mid, reply_text, reply_sid = turn.pending_reply
                        try:
                            reply_text = _clean_reply(reply_text)
                        except Exception as exc:
                            # Ход терминальный: payload_messages уже содержит
                            # assistant-сообщение с tool_calls, на которые нет
                            # ответов. Уронить это в общий retry-обработчик значит
                            # переотправить такой payload, получить 400 и стереть
                            # историю чата. Завершаем ход здесь.
                            logger.error(
                                f"❌ Не удалось подготовить текст реплая: {exc}",
                                exc_info=True,
                            )
                            await _delete_turn_status()
                            if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                await _save_assistant("")
                            return
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        if reply_text:
                            logger.info(f"↩️ [magenta]Ответ реплаем на[/] [#{reply_sid}]")
                            try:
                                sent_mid = await _deliver(reply_text, reply_mid, turn.status_message)
                            except Exception as exc:
                                logger.error(f"❌ Не удалось доставить ответ: {exc}", exc_info=True)
                                if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                    await _save_assistant("")
                                raise ReplyDeliveryError(
                                    "Telegram не подтвердил доставку ответа"
                                ) from exc
                            saved = await _save_assistant(reply_text)
                            await _remember_bot_mid(saved, sent_mid)
                        else:
                            # reply_to_message без текста — отправлять нечего (пустых сообщений не шлём)
                            if turn.reacted or turn.sticker_sent or turn.voice_sent:
                                await _save_assistant("")  # реакция/стикер/голос, текста нет
                            elif not mentioned:
                                logger.info("🤫 [dim]Промолчала (ambient, reply без текста)[/]")
                            else:
                                logger.warning("⚠️ [yellow]reply_to_message без текста при прямом обращении[/]")
                            if turn.status_message:
                                try:
                                    await turn.status_message.delete()
                                except Exception:
                                    pass
                        return

                    # send_messages — terminal burst (2–3 short bubbles + typing pauses).
                    if turn.pending_messages:
                        messages = list(turn.pending_messages)
                        turn.pending_messages = None
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        target_mid = update.message.message_id if mentioned else None
                        logger.info(
                            f"💬 [magenta]Серия из {len(messages)} сообщений[/]"
                        )
                        try:
                            sent_mid, delivered = await _deliver_multi(
                                messages, target_mid, turn.status_message
                            )
                        except Exception as exc:
                            logger.error(
                                f"❌ Не удалось доставить серию сообщений: {exc}",
                                exc_info=True,
                            )
                            if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                await _save_assistant("")
                            raise ReplyDeliveryError(
                                "Telegram не подтвердил доставку серии сообщений"
                            ) from exc
                        if delivered:
                            # One history row: joined bubbles (prefix cache / summarizer).
                            history_text = "\n".join(delivered)
                            saved = await _save_assistant(history_text)
                            await _remember_bot_mid(saved, sent_mid)
                        elif turn.reactions_made or turn.stickers_made or turn.voices_made:
                            await _save_assistant("")
                        return

                    # Стикер уже в чате — это полный ответ, лишний round-trip к API не нужен.
                    if turn.sticker_sent:
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        await _save_assistant("")
                        logger.info("🎨 [dim]Ход завершён стикером (без доп. текста)[/]")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return

                    # Голосовое уже в чате — терминальный ответ, без ещё одного round-trip.
                    if turn.voice_sent:
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        await _save_assistant("")
                        logger.info("🔊 [dim]Ход завершён голосовым (без доп. текста)[/]")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return

                    continue

                reply = message.get('content', '')
                
                # В DEBUG показываем полный ответ модели
                if FULL_DEBUG_LOGS:
                    logger.debug("[blue]" + "=" * 80 + "[/]")
                    logger.debug("[bright_green]📥 ОТВЕТ ОТ МОДЕЛИ:[/]")
                    logger.debug("[blue]" + "=" * 80 + "[/]")
                    
                    # Цвет finish_reason
                    finish_color = {
                        'stop': 'green',
                        'length': 'yellow',
                        'tool_calls': 'cyan'
                    }.get(finish_reason, 'white')
                    
                    logger.debug(f"Finish reason: [{finish_color}]{finish_reason}[/]")
                    logger.debug(f"Content length: [dim]{len(reply)} символов[/]")
                    logger.debug("[green]Content:[/]")
                    logger.debug(f"[bright_green]{reply}[/]")
                    logger.debug("[blue]" + "=" * 80 + "[/]")
                
                # Увеличиваем счётчик вызовов API
                api_call_count[key] = api_call_count.get(key, 0) + 1

                reply = _clean_reply(reply)

                if not reply:
                    if turn.reacted or turn.sticker_sent or turn.voice_sent:
                        # Ограничилась реакцией/стикером/голосом — валидный ответ.
                        await _save_assistant("")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return
                    if not mentioned:
                        # Ambient-пинг: к Ber не обращались — осознанное молчание, это норма.
                        logger.info(f"🤫 [dim]Промолчала (ambient)[/] (ключ={key})")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return
                    # Прямое обращение / личка / событие — молчать нельзя. Один раз подталкиваем ответить.
                    if not forced_answer_nudge:
                        forced_answer_nudge = True
                        payload_messages.append({
                            "role": "system",
                            "content": "Тебе адресовали сообщение напрямую — нельзя молчать. Дай короткий ответ в своём стиле."
                        })
                        logger.info(f"↩️ [yellow]Пустой ответ при прямом обращении — подталкиваю ответить[/] (ключ={key})")
                        continue
                    logger.warning(f"⚠️ [yellow]Пустой ответ при прямом обращении даже после напоминания[/] (ключ={key})")
                    if turn.status_message:
                        try:
                            await turn.status_message.delete()
                        except Exception:
                            pass
                    return

                if finish_reason == 'length':
                    logger.warning(f"⚠️ [yellow]Ответ обрезан по лимиту токенов[/] (ключ={key})")
                    reply += "\n\n_(ответ обрезан)_"

                # Долговременная память обрабатывается отдельно из SQLite-очереди;
                # доставка ответа не зависит от extractor/verifier.

                # Модель не выбрала инструмент reply_to_message:
                # адресное обращение → реплай на триггер (как раньше);
                # ambient (случайный пинг) → обычное сообщение без reply.
                target_mid = update.message.message_id if mentioned else None
                try:
                    sent_mid = await _deliver(reply, target_mid, turn.status_message)
                except Exception as exc:
                    logger.error(f"❌ Не удалось доставить ответ: {exc}", exc_info=True)
                    if turn.reactions_made or turn.stickers_made or turn.voices_made:
                        await _save_assistant("")
                    raise ReplyDeliveryError(
                        "Telegram не подтвердил доставку ответа"
                    ) from exc
                saved = await _save_assistant(reply)
                await _remember_bot_mid(saved, sent_mid)
                return

            except ReplyDeliveryError:
                # Доставка — часть логической транзакции хода. Не повторяем LLM/tools
                # и даём debounce-слою оставить memory sources в waiting.
                raise
            except httpx.ConnectError:
                logger.error("❌ [bright_red]API недоступен![/]")
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await update.message.reply_text("❌ API недоступен!")
                return
            except httpx.TimeoutException:
                logger.error("❌ [bright_red]Таймаут запроса к API[/]")
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await update.message.reply_text("❌ Таймаут.")
                return
            except Exception as e:
                logger.error(f"❌ [bright_red]Ошибка в обработке запроса:[/] {e}", exc_info=True)
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await update.message.reply_text("❌ Ошибка при обработке.")
                return
