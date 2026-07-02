import re
import json
import random
import logging
import asyncio
import httpx
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, SUMMARY_INTERVAL, VISION_MODE, MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, MODEL, GENERATION_PARAMS, FACTUAL_TEMPERATURE, DEBUG, PRICE_PROMPT_CACHE_MISS,
    PRICE_PROMPT_CACHE_HIT, PRICE_COMPLETION, SYSTEM_PROMPT,
    MEMORY_SEARCH_LIMIT, MEMORY_MIN_SCORE, MEMORY_MAX_CHARS, MEMORY_FLUSH_MAX_CHARS, MAX_API_RETRIES,
)
import state
from state import histories, chat_tokens, api_call_count, get_history_lock, touch_activity, save_history
from memory_store import memory
from tools import web_search, read_url, TOOLS, ALLOWED_REACTIONS
from sticker_store import search_stickers
from config import STICKER_ENABLED

logger = logging.getLogger(__name__)


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
    """Формирует строку с текущей датой и временем суток для системного промпта."""
    now = datetime.now()
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


def _render_history_for_api(history: list) -> list:
    """
    Готовит копию истории для отправки в API.
    - В начало каждого user-сообщения подставляет стабильный reply-хэндл [#sid].
    - Выкидывает служебные ключи (sid/mid), которых не должно быть в payload.
    Сам тег [#N] нигде не хранится — он живёт только в этой эфемерной копии,
    поэтому история, память и суммаризация остаются чистыми, а префикс стабилен (cache hit).
    """
    out = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
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
                parts = []
                for r in reactions:
                    on = (r.get("on") or "").strip()
                    parts.append(f"{r.get('emoji', '')} на «{on}»" if on else r.get("emoji", ""))
                notes.append("Ты поставила реакцию " + ", ".join(parts) + ".")
            if stickers:
                sp = ", ".join(f"«{(s.get('desc') or '')[:40]}»" for s in stickers)
                notes.append("Ты отправила стикер " + sp + ".")
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


def _build_sid_map(history: list) -> dict:
    """Карта {sid -> telegram message_id} по текущей истории (для reply/react по [#N])."""
    return {
        m["sid"]: m.get("mid")
        for m in history
        if m.get("role") == "user" and m.get("sid") is not None
    }


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
    """Чистит ответ модели от служебных токенов, эмодзи и лишней финальной точки."""
    reply = re.sub(r'<\|channel\>.*?<channel\|>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<\|.*?\|>', '', reply).strip()
    
    # Удаляем все эмодзи из текста (диапазоны Unicode для эмодзи)
    # Охватывает основные блоки эмодзи
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # эмотиконы
        "\U0001F300-\U0001F5FF"  # символы и пиктограммы
        "\U0001F680-\U0001F6FF"  # транспорт и карты
        "\U0001F1E0-\U0001F1FF"  # флаги
        "\U00002702-\U000027B0"  # дингбаты
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # дополнительные эмодзи
        "\U0001FA70-\U0001FAFF"  # расширенные эмодзи
        "\U00002600-\U000026FF"  # разное
        "\U00002700-\U000027BF"  # дингбаты
        "]+", 
        flags=re.UNICODE
    )
    reply = emoji_pattern.sub('', reply).strip()
    
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
        return msg_match.group(1).strip()

    # Если нет Message, убираем служебные блоки
    text = re.sub(
        r'\[(?:Image description|Video description|Context from memory|User|Time|Reply to|Quoted message|Forwarded from [^]]+):(?:[^\[\]]|\[(?!Message:))*?\]',
        '',
        content
    )

    return text.strip()


def _format_memory_block(mem_results: dict) -> str:
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
        line = f"- {fact}"
        if total + len(line) > MEMORY_MAX_CHARS:
            break
        lines.append(line)
        total += len(line)
        if len(lines) >= MEMORY_SEARCH_LIMIT:
            break

    return "\n".join(lines)


# ========================================
# 💾 ДОЛГОВРЕМЕННАЯ ПАМЯТЬ v2 (батчи под лимит эмбеддера)
# ========================================

def _chunk_lines_by_chars(lines: list, max_chars: int) -> list:
    """Режет список строк на под-батчи, каждый не длиннее max_chars символов."""
    chunks, cur, cur_len = [], [], 0
    for line in lines:
        # Если одна строка сама по себе больше лимита — режем её жёстко
        if len(line) > max_chars:
            if cur:
                chunks.append(cur)
                cur, cur_len = [], 0
            for i in range(0, len(line), max_chars):
                chunks.append([line[i:i + max_chars]])
            continue
        if cur and cur_len + len(line) > max_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append(cur)
    return chunks


async def _add_memory_chunks(key: str, lines: list) -> None:
    """Отправляет накопленные реплики в Mem0 батчами под лимит эмбеддера."""
    if not memory or not lines:
        return
    for chunk in _chunk_lines_by_chars(lines, MEMORY_FLUSH_MAX_CHARS):
        text = "\n".join(chunk)
        try:
            result = await asyncio.to_thread(
                memory.add,
                [{"role": "user", "content": text}],
                user_id=key,
            )
            added = len(result.get('results', [])) if isinstance(result, dict) else 0
            logger.info(f"✅ Память: батч {len(text)} симв. → [bright_green]{added}[/] фактов (scope={key})")
        except Exception as e:
            logger.error(f"⚠️ Ошибка сохранения памяти (scope={key}): {e}")


def record_user_memory(key: str, text: str, user_name: str, is_group: bool) -> None:
    """
    Кладёт реплику юзера в буфер памяти. При достижении лимита символов —
    запускает фоновый флаш батчем. Вызывается из обработчика сообщений.
    """
    if not memory:
        return
    text = (text or "").strip()
    if not text:
        return

    line = f"{user_name}: {text}" if is_group else text
    buf = state.pending_memory.setdefault(key, [])
    buf.append(line)

    if sum(len(l) for l in buf) >= MEMORY_FLUSH_MAX_CHARS:
        # Снимаем снимок и очищаем синхронно (без await) — без гонок в одном loop
        lines = list(buf)
        state.pending_memory[key] = []
        asyncio.create_task(_add_memory_chunks(key, lines))


def flush_pending_memory_blocking() -> None:
    """Синхронно сохраняет остатки буфера при остановке бота (loop уже не крутится)."""
    if not memory:
        return
    for key in list(state.pending_memory.keys()):
        lines = state.pending_memory.get(key) or []
        if not lines:
            continue
        state.pending_memory[key] = []
        for chunk in _chunk_lines_by_chars(lines, MEMORY_FLUSH_MAX_CHARS):
            try:
                memory.add([{"role": "user", "content": "\n".join(chunk)}], user_id=key)
            except Exception as e:
                logger.error(f"⚠️ Ошибка финального сохранения памяти '{key}': {e}")


async def summarize_history(history: list) -> list:
    to_summarize = history[:-SUMMARY_INTERVAL]
    keep_recent = history[-SUMMARY_INTERVAL:]

    if not to_summarize:
        return history

    # Старые сообщения уходят в резюме (их [#N] исчезают), у оставшихся свежих
    # сбрасываем нумерацию с #1, чтобы номера не росли бесконечно.
    _renumber_sids(keep_recent)
    
    # Формируем текст для локальной модели
    # Берём только содержательные реплики. Реакция-без-текста (content="") сюда не идёт:
    # её эмодзи живут в поле reactions, которое в резюме не нужно — старые реакции забываются.
    text_to_summarize = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in to_summarize
        if isinstance(m.get('content'), str) and m.get('content').strip()
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
            
            if DEBUG:
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
        async with get_history_lock(key):
            history = await summarize_history(history)
            histories[key] = history
            save_history(key)

    system_prompt = _build_system_prompt()
    # В payload подставляем reply-хэндлы [#N] (только в копию, история остаётся чистой)
    payload_messages = [{"role": "system", "content": system_prompt}] + _render_history_for_api(history)
    sid_to_mid = _build_sid_map(history)

    if memory:
        try:
            # Валидация ключа для безопасности
            if not re.match(r'^(private|group)_-?\d+$', key):
                logger.warning(f"⚠️ [yellow]Невалидный ключ памяти:[/] {key}")
            else:
                # Ищем по чистому тексту пользователя
                query = _extract_plain_text(history[-1].get('content', '')) if history else user_name
                if not query:
                    query = user_name

                if DEBUG:
                    logger.debug(f"🔍 Mem0 поиск: query='{query[:80]}', scope={key}")

                # Уменьшен таймаут до 15 секунд для быстрого ответа
                mem_results = await asyncio.wait_for(
                    asyncio.to_thread(
                        memory.search,
                        query,
                        filters={"user_id": key},
                        limit=MEMORY_SEARCH_LIMIT
                    ),
                    timeout=15.0
                )

                results_count = len(mem_results.get('results', []))
                mem_text = _format_memory_block(mem_results)
                
                if mem_text and payload_messages[-1]["role"] == "user":
                    last_content = payload_messages[-1]["content"]
                    payload_messages[-1] = {
                        "role": "user",
                        "content": f"{last_content}\n\n[Context from memory:\n{mem_text}\n]"
                    }
                    facts_count = mem_text.count('\n') if mem_text else 0
                    
                    # Краткий лог для INFO, детальный для DEBUG
                    logger.info(f"🧠 Память: найдено {results_count} → загружено {facts_count} фактов ({len(mem_text)} символов)")
                    
                    if DEBUG:
                        logger.debug(f"📝 Факты:\n{mem_text}")

        except asyncio.TimeoutError:
            logger.warning(f"⚠️ [yellow]Память: таймаут поиска (15s), продолжаем без неё[/] scope={key}")
        except Exception as e:
            logger.error(f"⚠️ [red]Ошибка получения памяти:[/] {e}")

    status_message = None
    used_tool = False  # после вызова инструмента (поиск/ссылка) отвечаем с пониженной температурой
    reacted = False    # бот поставил реакцию — допускаем ответ без текста
    reactions_made = []  # [{"emoji", "on"}] — реакции этого хода, чтобы записать их в историю
    sticker_sent = False  # бот отправил стикер — тоже допускаем ответ без текста
    stickers_made = []  # [{"desc"}] — стикеры этого хода, чтобы модель помнила, что отправила

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
        if not text and not reactions_made and not stickers_made:
            return None
        entry = {"role": "assistant", "content": text}
        if reactions_made:
            entry["reactions"] = list(reactions_made)
        if stickers_made:
            entry["stickers"] = list(stickers_made)
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
        if DEBUG:
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

        for attempt in range(MAX_API_RETRIES):
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
                
                response = await client.post(DEEPSEEK_API_URL, json=payload, headers=headers)

                if response.status_code == 400:
                    histories[key] = []
                    logger.error(f"[red]400:[/] {response.text}")
                    await update.message.reply_text("⚠️ История сброшена. Напишите ещё раз.")
                    return

                # Обработка rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"⚠️ [yellow]Rate limit (429), ждём {retry_after}s перед retry {attempt+1}/{MAX_API_RETRIES}[/]")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code != 200:
                    logger.error(f"❌ [red]API error {response.status_code}:[/] {response.text[:200]}")
                    if attempt < MAX_API_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
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
                    payload_messages.append(message)
                    used_tool = True  # дальше отвечаем на холодной температуре
                    pending_reply = None  # (target_mid, text, sid) если модель выбрала reply_to_message

                    for tool_call in message['tool_calls']:
                        func_name = tool_call['function']['name']
                        args = json.loads(tool_call['function']['arguments'])

                        if func_name == 'web_search':
                            await update.message.chat.send_action(action="typing")
                            status_text = f"🔍 Выполняю поиск: *{args['query']}*..."

                            if status_message is None:
                                status_message = await update.message.reply_text(
                                    status_text,
                                    parse_mode="Markdown"
                                )
                            else:
                                try:
                                    await status_message.edit_text(
                                        status_text,
                                        parse_mode="Markdown"
                                    )
                                except Exception as e:
                                    logger.warning(f"⚠️ [yellow]Не удалось отредактировать статусное сообщение:[/] {e}")

                            logger.info(f"🔍 [blue]Поиск:[/] {args['query']}")

                            search_result = web_search(
                                query=args['query'],
                                max_results=args.get('max_results', 5),
                                timelimit=args.get('timelimit', None),
                                region=args.get('region', 'ru-ru')
                            )

                            if status_message:
                                try:
                                    await status_message.edit_text("🔍 Поиск завершён, обрабатываю результаты...")
                                except Exception:
                                    pass

                            logger.debug(f"📄 Результат: {repr(search_result[:200])}")

                            if not search_result:
                                search_result = "Поиск не дал результатов."

                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": search_result
                            })

                        elif func_name == 'read_url':
                            url = args.get('url', '')
                            await update.message.chat.send_action(action="typing")
                            status_text = "🔗 Читаю ссылку..."
                            if status_message is None:
                                status_message = await update.message.reply_text(status_text)
                            else:
                                try:
                                    await status_message.edit_text(status_text)
                                except Exception:
                                    pass

                            logger.info(f"🔗 [blue]Чтение ссылки:[/] {url}")
                            page_text = read_url(url)
                            logger.debug(f"📄 Страница: {repr(page_text[:200])}")

                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": page_text
                            })

                        elif func_name == 'send_sticker':
                            query = (args.get('query') or '').strip()
                            if not STICKER_ENABLED:
                                tool_result = "Стикеры отключены. Ответь текстом."
                            elif not query:
                                tool_result = "Пустой запрос стикера. Ответь текстом."
                            else:
                                try:
                                    await update.message.chat.send_action(action="choose_sticker")
                                except Exception:
                                    pass
                                # Поиск синхронный (эмбеддинг + Qdrant) — уводим в поток,
                                # чтобы не блокировать event loop.
                                candidates = await asyncio.to_thread(search_stickers, query)
                                if not candidates:
                                    logger.info(f"🎨 [dim]Стикер под '{query}' не найден (ниже порога)[/]")
                                    tool_result = "Подходящего стикера не нашлось. Ответь обычным способом."
                                else:
                                    # Берём лучший; рандомим ТОЛЬКО среди near-ties (скор в пределах
                                    # 0.03 от топа) — так не теряем качество, но даём разнообразие
                                    # на реальных ничьих, а не на явно худших кандидатах.
                                    best_score = candidates[0].get('score') or 0.0
                                    pool = [c for c in candidates if best_score - (c.get('score') or 0.0) <= 0.03]
                                    chosen = random.choice(pool or candidates[:1])
                                    file_id = chosen.get('file_id')
                                    thread_id = getattr(update.message, "message_thread_id", None)
                                    try:
                                        kw = {"chat_id": update.effective_chat.id, "sticker": file_id}
                                        if thread_id is not None:
                                            kw["message_thread_id"] = thread_id
                                        await context.bot.send_sticker(**kw)
                                        sticker_sent = True
                                        stickers_made.append({"desc": chosen.get('description') or query})
                                        logger.info(
                                            f"🎨 [magenta]Стикер отправлен[/] под '{query}' "
                                            f"(score={chosen.get('score'):.3f}, «{(chosen.get('description') or '')[:40]}»)"
                                        )
                                        tool_result = "Стикер отправлен. Если добавить нечего — можешь обойтись без текста."
                                    except Exception as e:
                                        logger.warning(f"⚠️ [yellow]Не удалось отправить стикер:[/] {e}")
                                        tool_result = "Стикер отправить не удалось. Ответь текстом."

                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": tool_result
                            })

                        elif func_name == 'react_to_message':
                            # Модель часто шлёт эмодзи с вариативным селектором U+FE0F (❤️),
                            # а Telegram и ALLOWED_REACTIONS хранят каноничную форму без него (❤).
                            # Срезаем FE0F, иначе сердце/☃/✍/🕊/❤‍🔥/🤷‍♂ молча не проходят валидацию.
                            emoji = (args.get('emoji') or '').strip().replace(chr(0xFE0F), '')
                            try:
                                react_sid = int(args.get('id'))
                            except (TypeError, ValueError):
                                react_sid = None
                            react_mid = sid_to_mid.get(react_sid) if react_sid is not None else None
                            if react_mid is None:
                                react_mid = update.message.message_id
                            if emoji not in ALLOWED_REACTIONS:
                                tool_result = f"Эмодзи '{emoji}' не разрешён Telegram. Ответь текстом или выбери из списка."
                            else:
                                try:
                                    await context.bot.set_message_reaction(
                                        chat_id=update.effective_chat.id,
                                        message_id=react_mid,
                                        reaction=emoji
                                    )
                                    reacted = True
                                    # Привязку храним как короткую цитату, БЕЗ числового [#N]:
                                    # хэндлы перенумеровываются → старый номер протух бы и провоцировал
                                    # галлюцинации. Цитата самодостаточна и не теряет смысл со временем.
                                    on_quote = None
                                    if react_mid != update.message.message_id:
                                        for _m in history:
                                            if _m.get("role") == "user" and _m.get("mid") == react_mid:
                                                _t = (_m.get("content") or "").strip()
                                                on_quote = (_t[:40] + "…") if len(_t) > 40 else _t
                                                break
                                    reactions_made.append({"emoji": emoji, "on": on_quote})
                                    logger.info(f"😀 [magenta]Реакция:[/] {emoji} → [#{react_sid if react_sid is not None else 'текущее'}]")
                                    tool_result = f"Реакция {emoji} поставлена. Если добавить нечего — можешь обойтись без текста."
                                except Exception as e:
                                    logger.warning(f"⚠️ [yellow]Не удалось поставить реакцию {emoji}:[/] {e}")
                                    tool_result = "Не удалось поставить реакцию, ответь текстом."

                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": tool_result
                            })

                        elif func_name == 'reply_to_message':
                            try:
                                reply_sid = int(args.get('id'))
                            except (TypeError, ValueError):
                                reply_sid = None
                            reply_text = args.get('text', '') or ''
                            reply_mid = sid_to_mid.get(reply_sid)
                            if reply_mid is None:
                                # Невалидный/устаревший [#N] — отвечаем на текущее сообщение
                                reply_mid = update.message.message_id
                            pending_reply = (reply_mid, reply_text, reply_sid)
                            # Терминальный инструмент: ниже делаем return, нового запроса к API
                            # не будет — поэтому tool-result в payload не добавляем.

                        else:
                            # Неизвестный инструмент — всё равно отвечаем, иначе API упадёт
                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": f"Инструмент '{func_name}' не поддерживается."
                            })

                    # reply_to_message терминальный и приоритетный: если модель его вызвала,
                    # отправляем выбранный ответ и завершаем — без ещё одного витка к API
                    # и без дефолтного реплая ниже (двойной отправки не будет).
                    if pending_reply is not None:
                        reply_mid, reply_text, reply_sid = pending_reply
                        reply_text = _clean_reply(reply_text)
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        if reply_text:
                            saved = await _save_assistant(reply_text)
                            logger.info(f"↩️ [magenta]Ответ реплаем на[/] [#{reply_sid}]")
                            sent_mid = await _deliver(reply_text, reply_mid, status_message)
                            await _remember_bot_mid(saved, sent_mid)
                        else:
                            # reply_to_message без текста — отправлять нечего (пустых сообщений не шлём)
                            if reacted or sticker_sent:
                                await _save_assistant("")  # запоминаем реакцию/стикер, текста нет
                            elif not mentioned:
                                logger.info("🤫 [dim]Промолчала (ambient, reply без текста)[/]")
                            else:
                                logger.warning("⚠️ [yellow]reply_to_message без текста при прямом обращении[/]")
                            if status_message:
                                try:
                                    await status_message.delete()
                                except Exception:
                                    pass
                        return

                    continue

                reply = message.get('content', '')
                
                # В DEBUG показываем полный ответ модели
                if DEBUG:
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
                    logger.debug(f"[green]Content:[/]")
                    logger.debug(f"[bright_green]{reply}[/]")
                    logger.debug("[blue]" + "=" * 80 + "[/]")
                
                # Увеличиваем счётчик вызовов API
                api_call_count[key] = api_call_count.get(key, 0) + 1

                reply = _clean_reply(reply)

                if not reply:
                    if reacted or sticker_sent:
                        # Ограничилась реакцией/стикером — валидный ответ. Запоминаем в истории.
                        await _save_assistant("")
                        if status_message:
                            try:
                                await status_message.delete()
                            except Exception:
                                pass
                        return
                    if not mentioned:
                        # Ambient-пинг: к Ber не обращались — осознанное молчание, это норма.
                        logger.info(f"🤫 [dim]Промолчала (ambient)[/] (ключ={key})")
                        if status_message:
                            try:
                                await status_message.delete()
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
                    if status_message:
                        try:
                            await status_message.delete()
                        except Exception:
                            pass
                    return

                if finish_reason == 'length':
                    logger.warning(f"⚠️ [yellow]Ответ обрезан по лимиту токенов[/] (ключ={key})")
                    reply += "\n\n_(ответ обрезан)_"

                saved = await _save_assistant(reply)  # пишет ход + прикрепляет реакции этого хода

                # Сохранение в долговременную память происходит не здесь, а батчами:
                # реплики юзеров копятся в state.pending_memory (см. record_user_memory)
                # и флашатся по достижении лимита символов (memory v2).

                # Модель не выбрала инструмент reply_to_message:
                # адресное обращение → реплай на триггер (как раньше);
                # ambient (случайный пинг) → обычное сообщение без reply.
                target_mid = update.message.message_id if mentioned else None
                sent_mid = await _deliver(reply, target_mid, status_message)
                await _remember_bot_mid(saved, sent_mid)
                return

            except httpx.ConnectError:
                logger.error("❌ [bright_red]API недоступен![/]")
                await update.message.reply_text("❌ API недоступен!")
                return
            except httpx.TimeoutException:
                logger.error("❌ [bright_red]Таймаут запроса к API[/]")
                await update.message.reply_text("❌ Таймаут.")
                return
            except Exception as e:
                logger.error(f"❌ [bright_red]Ошибка в обработке запроса:[/] {e}", exc_info=True)
                if attempt < MAX_API_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                await update.message.reply_text("❌ Ошибка при обработке.")
                return


async def generate_and_send_tiktok_review(bot, chat_id, message_id, key, history, message_thread_id=None):
    """Generates a review for a TikTok video using DeepSeek and replies to the Telegram message."""
    system_prompt = _build_system_prompt()
    payload_messages = [{"role": "system", "content": system_prompt}] + _render_history_for_api(history)

    payload = {
        "model": MODEL,
        "messages": payload_messages,
        "max_tokens": MAX_REPLY_TOKENS,
        **GENERATION_PARAMS
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(DEEPSEEK_API_URL, json=payload, headers=headers)
            if response.status_code != 200:
                logger.error(f"❌ [red]DeepSeek review API error {response.status_code}:[/] {response.text}")
                return

            data = response.json()
            reply = data['choices'][0]['message']['content']

            reply = _clean_reply(reply)

            if not reply:
                return

            reply_html = markdown_to_html(reply)
            
            logger.info(f"📤 Отправка рецензии в чат {chat_id} на сообщение {message_id}...")
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    reply_to_message_id=message_id,
                    text=reply_html,
                    parse_mode="HTML"
                )
            except Exception as e:
                if "not found" in str(e).lower() or "reply" in str(e).lower():
                    logger.warning(f"⚠️ Не удалось отправить реплай к сообщению {message_id}: {e}. Пробуем отправить без реплая...")
                    await bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=reply_html,
                        parse_mode="HTML"
                    )
                else:
                    raise

            async with get_history_lock(key):
                # Reload history just in case
                if key in histories:
                    histories[key].append({"role": "assistant", "content": reply})
                else:
                    histories[key] = history + [{"role": "assistant", "content": reply}]
                touch_activity(key)
                save_history(key)
                
    except Exception as e:
        logger.error(f"❌ [red]Ошибка при генерации рецензии TikTok:[/] {e}", exc_info=True)
