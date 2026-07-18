import re
import logging
import asyncio
import copy
import httpx
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, SUMMARY_INTERVAL, VISION_MODE, MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, MODEL, GENERATION_PARAMS, FACTUAL_TEMPERATURE, FULL_DEBUG_LOGS, PRICE_PROMPT_CACHE_MISS,
    PRICE_PROMPT_CACHE_HIT, PRICE_COMPLETION, SYSTEM_PROMPT,
    MEMORY_SEARCH_LIMIT, MEMORY_MIN_SCORE, MEMORY_MAX_CHARS,
    MEMORY_QUERY_MIN_CHARS, MEMORY_QUERY_RECENT_MESSAGES, MAX_API_RETRIES,
    MAX_TOOL_ROUNDS, STREAMING_ENABLED, STREAM_UPDATE_INTERVAL_SECONDS,
    STREAM_PREVIEW_MIN_CHARS,
)
from state import histories, chat_tokens, api_call_count, get_history_lock, touch_activity, save_history
import memory_store
import state
from tools import TOOLS
from tool_handlers import ToolTurn, dispatch_tool_call
from streaming import TelegramStreamPreview, stream_chat_completion
from utils import now_local, is_low_signal_user_text, strip_tiktok_urls

logger = logging.getLogger(__name__)


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


# Дополнение к системному промпту, когда включён vision-режим
VISION_PROMPT_SUFFIX = """
=== IMAGES, VIDEO, AND AUDIO ===
When a user sends media, you receive it as [Image description: ...], [Video description: ...], or [Audio description: ...] inside their message.
These descriptions come from a vision/audio model that processed the media and described it naturally — like a friend telling you what they saw or heard.

The description includes:
- **Images**: what's visible (people, objects, text, logos, setting, colors), recognized characters/memes/brands, mood
- **Video**: what's happening, how the scene evolves over time, recognized content
- **Audio**: transcribed speech or description of sounds/music

How to use it:
✓ React naturally as if you experienced it yourself — joke, tease, or comment on interesting details
✓ Reference recognized characters/memes/brands by name — this is your advantage
✓ For audio: respond to what was said as if you heard it directly
✓ If the description says "похоже на..." (looks like) — you can mention it with slight uncertainty
✓ If it says the model didn't recognize something — don't invent names

What NOT to do:
✗ NEVER write "visible in the picture", "judging by the description", "according to the text", "you said in the audio"
✗ Don't say "the description mentions..." or "the transcript shows..." — you're supposed to have experienced it directly
✗ Don't quote the description structure or format

Treat the description as your own observation. The user doesn't know you didn't process the media directly.
"""

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
        if reactions or incoming or stickers:
            if content:
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


def _clean_reply(reply: str) -> str:
    """Чистит ответ модели от служебных токенов и лишней финальной точки.

    Эмодзи намеренно НЕ вырезаются: промпт отговаривает модель от их использования,
    но когда эмодзи — сам ответ (огрызок, ответная реакция), он должен пройти.
    """
    reply = re.sub(r'<\|channel\>.*?<channel\|>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<\|.*?\|>', '', reply).strip()

    reply = re.sub(
        r'\[Context from memory(?:\s*:[^\]]*)?\]',
        'долгосрочной памяти',
        reply,
        flags=re.IGNORECASE,
    )

    # [#N] — внутренние reply-хэндлы для инструментов. Модель иногда всё же
    # цитирует их вопреки системному промпту, поэтому не выпускаем их в Telegram.
    reply = re.sub(r'\[#\d+\](?:\s*(?:,|и|или)\s*\[#\d+\])*', '', reply)
    reply = re.sub(r'\s+([,.;:!?])', r'\1', reply)
    reply = re.sub(r'[,;:]+([.!?])', r'\1', reply)
    reply = re.sub(r'[ \t]{2,}', ' ', reply)

    reply = reply.strip()
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


def _memory_fact_matches_query(fact: str, query: str) -> bool:
    if not query or _MEMORY_RECALL_RE.search(query):
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
    
    # Формируем текст для локальной модели
    # Берём только содержательные реплики. Реакция-без-текста (content="") сюда не идёт:
    # её эмодзи живут в поле reactions, которое в резюме не нужно — старые реакции забываются.
    text_to_summarize = "\n".join([
        f"{m['role']}: {strip_tiktok_urls(m['content'])}"
        for m in to_summarize
        if isinstance(m.get('content'), str) and strip_tiktok_urls(m.get('content', '')).strip()
    ])
    
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
        "max_tokens": 2000,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40 
    }
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(DEEPSEEK_API_URL, json=summary_payload, headers=headers)
            logger.info(f"Ответ сумморизации: [cyan]{response.status_code}[/]")
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content']
            
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
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

    if memory_store.memory:
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
            return await client.post(DEEPSEEK_API_URL, json=payload, headers=headers)

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
                DEEPSEEK_API_URL,
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
                except Exception as e:
                    logger.warning(f"⚠️ [yellow]Правка статуса с HTML не прошла:[/] {e}")
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
            except Exception as e:
                logger.warning(f"⚠️ [yellow]HTML не распарсился, отправляю как текст:[/] {e}")
                return await _raw(reply_plain, False)
        else:
            # Длинный ответ шлём чистым текстом, чтобы не порвать HTML-теги на границе чанка
            first_mid = None
            for i in range(0, len(reply_plain), 4096):
                chunk = reply_plain[i:i + 4096]
                if thread_id is not None:
                    sent = await context.bot.send_message(chat_id=chat_id, text=chunk, message_thread_id=thread_id)
                else:
                    sent = await context.bot.send_message(chat_id=chat_id, text=chunk)
                if first_mid is None:
                    first_mid = sent.message_id
            return first_mid

    async def _save_assistant(text: str):
        """
        Пишет ход бота в историю. Если за ход были реакции — прикрепляет их
        к этой же записи (поле reactions), чтобы модель помнила, что среагировала.
        Реакция-без-текста сохраняется как пустой content + reactions.
        Дописывает в хвост — префикс не меняется, cache hit сохраняется.
        Возвращает созданную запись (чтобы потом проставить ей mid) или None.
        """
        if not text and not turn.reactions_made and not turn.stickers_made:
            return None
        entry = {"role": "assistant", "content": text}
        if turn.reactions_made:
            entry["reactions"] = list(turn.reactions_made)
        if turn.stickers_made:
            entry["stickers"] = list(turn.stickers_made)
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

            payload = {
                "model": MODEL,
                "messages": payload_messages,
                "max_tokens": MAX_REPLY_TOKENS,
                "tools": TOOLS,
                **gen_params
            }

            try:
                headers = {
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                }
                
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
                    
                    chat_tokens[key] = total_tokens
                    
                    logger.info(f"📊 Токены: запрос=[cyan]{prompt_tokens}[/] (кэш=[cyan]{cached_tokens}[/]), "
                                f"ответ=[cyan]{completion_tokens}[/], всего=[bright_green]{total_tokens}[/]")
                
                    prompt_not_cached = prompt_tokens - cached_tokens
                    cost_prompt = (prompt_not_cached / 1_000_000) * PRICE_PROMPT_CACHE_MISS
                    cost_cached = (cached_tokens / 1_000_000) * PRICE_PROMPT_CACHE_HIT
                    cost_completion = (completion_tokens / 1_000_000) * PRICE_COMPLETION

                    total_cost = cost_prompt + cost_cached + cost_completion
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
                    used_tool = True  # дальше отвечаем на холодной температуре
                    turn.pending_reply = None  # (target_mid, text, sid) если модель выбрала reply_to_message

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
                        reply_text = _clean_reply(reply_text)
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        if reply_text:
                            logger.info(f"↩️ [magenta]Ответ реплаем на[/] [#{reply_sid}]")
                            try:
                                sent_mid = await _deliver(reply_text, reply_mid, turn.status_message)
                            except Exception as exc:
                                logger.error(f"❌ Не удалось доставить ответ: {exc}", exc_info=True)
                                if turn.reactions_made or turn.stickers_made:
                                    await _save_assistant("")
                                raise ReplyDeliveryError(
                                    "Telegram не подтвердил доставку ответа"
                                ) from exc
                            saved = await _save_assistant(reply_text)
                            await _remember_bot_mid(saved, sent_mid)
                        else:
                            # reply_to_message без текста — отправлять нечего (пустых сообщений не шлём)
                            if turn.reacted or turn.sticker_sent:
                                await _save_assistant("")  # запоминаем реакцию/стикер, текста нет
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
                    if turn.reacted or turn.sticker_sent:
                        # Ограничилась реакцией/стикером — валидный ответ. Запоминаем в истории.
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
                    if turn.reactions_made or turn.stickers_made:
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
