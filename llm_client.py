import re
import json
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


# Дополнение к системному промпту, когда включён vision-режим (общее для обычных ответов и TikTok-рецензий)
VISION_PROMPT_SUFFIX = """
            === IMAGES AND VIDEO ===
            When a user sends a photo, you receive it as [Image description: ...] inside the message.
            When a user sends a video, you receive it as [Video description: ...] — this is a description of several frames distributed along the timeline.
            The description arrives in a structured format: sections «DETAILS» (what is visible), «RECOGNITION» (recognized characters/memes/brands), and «SUMMARY» (brief retelling).
            Use the RECOGNITION section to mention the character/meme/brand by name — this is your main advantage. The SUMMARY sets the mood for the joke. DETAILS is raw material; do not read it out.
            Consider that you saw the picture or video yourself. React naturally: joke, tease, or comment on interesting details.
            NEVER write "visible in the picture", "visible in the video", "judging by the description", "according to the text", "in the details section" — this destroys the illusion.
            DO NOT read the sections verbatim and do not quote the format «DETAILS/RECOGNITION/SUMMARY». Use the description only as context for a witty remark.
            If the RECOGNITION section says «possibly this is …» — you may mention it with slight uncertainty. If it says «I don't recognize» — do not invent names.
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


def _clean_reply(reply: str) -> str:
    """Чистит ответ модели от служебных токенов и лишней финальной точки."""
    reply = re.sub(r'<\|channel\>.*?<channel\|>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<\|.*?\|>', '', reply).strip()
    reply = reply.strip()
    if reply.endswith('.') and not reply.endswith('...'):
        reply = reply[:-1]
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
    
    # Формируем текст для локальной модели
    text_to_summarize = "\n".join([
        f"{m['role']}: {m['content']}" 
        for m in to_summarize 
        if isinstance(m.get('content'), str)
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
    history: list, user_name: str, user_id: int, is_group: bool, 
    random_reply: bool, reason: str):

    # Автосуммаризация при достижении 85% от лимита токенов
    context_threshold = int(MAX_CONTEXT_TOKENS * 0.85)
    if chat_tokens.get(key, 0) > context_threshold:
        logger.info(f"📝 [yellow]Автосуммаризация[/] для key={key}")
        async with get_history_lock(key):
            history = await summarize_history(history)
            histories[key] = history
            save_history(key)

    system_prompt = _build_system_prompt()
    payload_messages = [{"role": "system", "content": system_prompt}] + history

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

    if random_reply and is_group and payload_messages[-1]["role"] == "user":
        random_instruction = (
            "\n\n=== IMPORTANT: YOU ARE RESPONDING RANDOMLY ===\n"
            "You don't have to respond to every message. Several messages came in without your response. "
            "You decided to respond now.\n"
            "RULES FOR THIS CASE:\n"
            "- Reply ONLY to the LAST message in history\n"
            "- Ignore older messages — nobody expects a reply to them anymore\n"
            "- If the last message wasn't addressed to you, comment on it as you wish\n"
            "- DO NOT list all the messages you didn't reply to earlier\n"
            "- Just give a short remark on the current moment\n"
            "=== END OF RULES ==="
        )
        last_content = payload_messages[-1]["content"]
        payload_messages[-1] = {
            "role": "user",
            "content": f"{last_content}{random_instruction}"
        }
            
    status_message = None
    used_tool = False  # после вызова инструмента (поиск/ссылка) отвечаем с пониженной температурой
    reacted = False    # бот поставил реакцию — допускаем пустой текстовый ответ

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

                        elif func_name == 'react_to_message':
                            emoji = (args.get('emoji') or '').strip()
                            if emoji not in ALLOWED_REACTIONS:
                                tool_result = f"Эмодзи '{emoji}' не разрешён Telegram. Ответь текстом или выбери из списка."
                            else:
                                try:
                                    await update.message.set_reaction(reaction=emoji)
                                    reacted = True
                                    logger.info(f"😀 [magenta]Реакция:[/] {emoji}")
                                    tool_result = f"Реакция {emoji} поставлена. Если добавить нечего — верни пустой ответ."
                                except Exception as e:
                                    logger.warning(f"⚠️ [yellow]Не удалось поставить реакцию {emoji}:[/] {e}")
                                    tool_result = "Не удалось поставить реакцию, ответь текстом."

                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": tool_result
                            })

                        else:
                            # Неизвестный инструмент — всё равно отвечаем, иначе API упадёт
                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": f"Инструмент '{func_name}' не поддерживается."
                            })

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
                    if reacted:
                        # Бот ограничился реакцией — это валидный ответ, текст не нужен
                        if status_message:
                            try:
                                await status_message.delete()
                            except Exception:
                                pass
                        return
                    await update.message.reply_text("❌ Пустой ответ от модели.")
                    return

                if finish_reason == 'length':
                    logger.warning(f"⚠️ [yellow]Ответ обрезан по лимиту токенов[/] (ключ={key})")
                    reply += "\n\n_(ответ обрезан)_"

                async with get_history_lock(key):
                    history.append({"role": "assistant", "content": reply})
                    histories[key] = history
                    touch_activity(key)  # Обновляем время активности
                    save_history(key)    # Персистим историю на диск

                # Сохранение в долговременную память происходит не здесь, а батчами:
                # реплики юзеров копятся в state.pending_memory (см. record_user_memory)
                # и флашатся по достижении лимита символов (memory v2).

                if is_group and not random_reply and reason != "reply":
                    reply = f"{reply}"

                # Готовим HTML-версию и чистый текст на случай, если HTML не распарсится
                reply_html = markdown_to_html(reply)
                reply_plain = strip_markdown(reply)

                async def _send_reply_as_new():
                    """Новым сообщением: HTML с фолбэком на чистый текст; длинное режем по 4096."""
                    if len(reply_html) <= 4096:
                        try:
                            await update.message.reply_text(reply_html, parse_mode="HTML")
                        except Exception as e:
                            logger.warning(f"⚠️ [yellow]HTML не распарсился, отправляю как текст:[/] {e}")
                            await update.message.reply_text(reply_plain)
                    else:
                        # Длинный ответ шлём чистым текстом, чтобы не порвать HTML-теги на границе чанка
                        for i in range(0, len(reply_plain), 4096):
                            await update.message.reply_text(reply_plain[i:i + 4096])

                if status_message:
                    edited = False
                    if len(reply_html) <= 4096:
                        try:
                            await status_message.edit_text(reply_html, parse_mode="HTML")
                            edited = True
                        except Exception as e:
                            logger.warning(f"⚠️ [yellow]Правка статуса с HTML не прошла:[/] {e}")
                    if not edited:
                        try:
                            await status_message.delete()
                        except Exception:
                            pass
                        await _send_reply_as_new()
                else:
                    await _send_reply_as_new()
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
    payload_messages = [{"role": "system", "content": system_prompt}] + history

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
