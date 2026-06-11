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
    MAX_REPLY_TOKENS, MODEL, GENERATION_PARAMS, DEBUG, PRICE_PROMPT_CACHE_MISS,
    PRICE_PROMPT_CACHE_HIT, PRICE_COMPLETION, SYSTEM_PROMPT,
    MEMORY_SEARCH_LIMIT, MEMORY_MIN_SCORE, MEMORY_MAX_CHARS, MAX_API_RETRIES,
)
from state import histories, chat_tokens, api_call_count, get_history_lock, touch_activity
from memory_store import memory
from tools import web_search, TOOLS

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
    
    # Жирный текст **text** или __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    
    # Курсив *text* или _text_ (но не внутри слов)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<i>\1</i>', text)
    
    # Жирный курсив ***text*** или ___text___
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'___(.+?)___', r'<b><i>\1</i></b>', text)
    
    # Зачёркнутый ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    
    # Ссылки [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', text)
    
    return text


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

    system_prompt = SYSTEM_PROMPT

    if VISION_MODE:
        system_prompt += ("""
            === IMAGES AND VIDEO ===
            When a user sends a photo, you receive it as [Image description: ...] inside the message.
            When a user sends a video, you receive it as [Video description: ...] — this is a description of several frames distributed along the timeline.
            The description arrives in a structured format: sections «DETAILS» (what is visible), «RECOGNITION» (recognized characters/memes/brands), and «SUMMARY» (brief retelling).
            Use the RECOGNITION section to mention the character/meme/brand by name — this is your main advantage. The SUMMARY sets the mood for the joke. DETAILS is raw material; do not read it out.
            Consider that you saw the picture or video yourself. React naturally: joke, tease, or comment on interesting details.
            NEVER write "visible in the picture", "visible in the video", "judging by the description", "according to the text", "in the details section" — this destroys the illusion.
            DO NOT read the sections verbatim and do not quote the format «DETAILS/RECOGNITION/SUMMARY». Use the description only as context for a witty remark.
            If the RECOGNITION section says «possibly this is …» — you may mention it with slight uncertainty. If it says «I don't recognize» — do not invent names.        
        """)

    now = datetime.now()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"]

    time_str = (
        f"Today is {days[now.weekday()]}, {now.day} {months[now.month-1]} {now.year} year. "
    )

    if 5 <= now.hour < 12:
        time_of_day = "morning"
    elif 12 <= now.hour < 17:
        time_of_day = "daytime"
    elif 17 <= now.hour < 23:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    time_str += f"Times of Day: {time_of_day}."

    # Автосуммаризация при достижении 85% от лимита токенов
    context_threshold = int(MAX_CONTEXT_TOKENS * 0.85)
    if chat_tokens.get(key, 0) > context_threshold:
        logger.info(f"📝 [yellow]Автосуммаризация[/] для key={key}")
        async with get_history_lock(key):
            history = await summarize_history(history)
            histories[key] = history
        
    system_prompt += f"\n\n=== CURRENT TIME ===\n{time_str}\n"
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
            payload = {
                "model": MODEL,
                "messages": payload_messages,
                "max_tokens": MAX_REPLY_TOKENS,
                "tools": TOOLS,
                **GENERATION_PARAMS 
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

                reply = re.sub(r'<\|channel\>.*?<channel\|>', '', reply, flags=re.DOTALL).strip()
                reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
                reply = re.sub(r'<\|.*?\|>', '', reply).strip()
                reply = reply.strip()
                if reply.endswith('.') and not reply.endswith('...'):
                    reply = reply[:-1]

                if not reply:
                    await update.message.reply_text("❌ Пустой ответ от модели.")
                    return

                if finish_reason == 'length':
                    logger.warning(f"⚠️ [yellow]Ответ обрезан по лимиту токенов[/] (ключ={key})")
                    reply += "\n\n_(ответ обрезан)_"

                async with get_history_lock(key):
                    history.append({"role": "assistant", "content": reply})
                    histories[key] = history
                    touch_activity(key)  # Обновляем время активности

                if memory:
                    async def save_memory_background():
                        try:
                            # В память кладём только чистую реплику пользователя
                            raw_user_msg = history[-2].get('content', '') if len(history) >= 2 else ''
                            last_user_msg = _extract_plain_text(raw_user_msg)

                            if not last_user_msg:
                                if DEBUG:
                                    logger.info(f"🧠 [magenta]Память: нет текста для сохранения[/] (scope={key}), пропуск")
                                return

                            # В группе подставляем имя для корректной атрибуции фактов
                            if is_group:
                                speaker_msg = f"{user_name}: {last_user_msg}"
                            else:
                                speaker_msg = last_user_msg

                            if DEBUG:
                                logger.debug(f"💾 Mem0 сохранение: '{speaker_msg[:60]}...' (scope={key})")

                            result = await asyncio.to_thread(
                                memory.add,
                                [
                                    {"role": "user", "content": speaker_msg}
                                ],
                                user_id=key
                            )
                            
                            added = len(result.get('results', []))
                            logger.info(f"✅ Память сохранена: {added} фактов")
                        except json.JSONDecodeError as e:
                            logger.error(f"⚠️ Ошибка парсинга JSON от mem0: {e}")
                            if DEBUG:
                                logger.debug(f"Попытка сохранить: '{speaker_msg[:200]}'")
                        except Exception as e:
                            logger.error(f"⚠️ Ошибка сохранения памяти: {e}")
                    
                    asyncio.create_task(save_memory_background()) 

                if is_group and not random_reply and reason != "reply":
                    reply = f"{reply}"

                # Конвертируем Markdown в HTML для Telegram
                reply_html = markdown_to_html(reply)
                
                if status_message:
                    try:
                        if len(reply_html) <= 4096:
                            await status_message.edit_text(reply_html, parse_mode="HTML")
                        else:
                            await status_message.delete()
                            for i in range(0, len(reply_html), 4096):
                                await update.message.reply_text(reply_html[i:i+4096], parse_mode="HTML")
                    except Exception as e:
                        logger.warning(f"⚠️ [yellow]Не удалось отредактировать статусное сообщение:[/] {e}")
                        await update.message.reply_text(reply_html, parse_mode="HTML")
                else:
                    if len(reply_html) <= 4096:
                        await update.message.reply_text(reply_html, parse_mode="HTML")
                    else:
                        for i in range(0, len(reply_html), 4096):
                            await update.message.reply_text(reply_html[i:i+4096], parse_mode="HTML")
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
    system_prompt = SYSTEM_PROMPT
    if VISION_MODE:
        system_prompt += ("""
            === IMAGES AND VIDEO ===
            When a user sends a photo, you receive it as [Image description: ...] inside the message.
            When a user sends a video, you receive it as [Video description: ...] — this is a description of several frames distributed along the timeline.
            The description arrives in a structured format: sections «DETAILS» (what is visible), «RECOGNITION» (recognized characters/memes/brands), and «SUMMARY» (brief retelling).
            Use the RECOGNITION section to mention the character/meme/brand by name — this is your main advantage. The SUMMARY sets the mood for the joke. DETAILS is raw material; do not read it out.
            Consider that you saw the picture or video yourself. React naturally: joke, tease, or comment on interesting details.
            NEVER write "visible in the picture", "visible in the video", "judging by the description", "according to the text", "in the details section" — this destroys the illusion.
            DO NOT read the sections verbatim and do not quote the format «DETAILS/RECOGNITION/SUMMARY». Use the description only as context for a witty remark.
        """)

    now = datetime.now()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]

    time_str = f"Today is {days[now.weekday()]}, {now.day} {months[now.month-1]} {now.year} year. "
    if 5 <= now.hour < 12:
        time_of_day = "morning"
    elif 12 <= now.hour < 17:
        time_of_day = "daytime"
    elif 17 <= now.hour < 23:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    time_str += f"Times of Day: {time_of_day}."

    system_prompt += f"\n\n=== CURRENT TIME ===\n{time_str}\n"
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

            reply = re.sub(r'<\|channel\>.*?<channel\|>', '', reply, flags=re.DOTALL).strip()
            reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
            reply = re.sub(r'<\|.*?\|>', '', reply).strip()
            reply = reply.strip()
            if reply.endswith('.') and not reply.endswith('...'):
                reply = reply[:-1]

            if not reply:
                return

            reply_html = markdown_to_html(reply)
            
            logger.info(f"📤 Отправка рецензии в чат {chat_id} на сообщение {message_id}...")
            await bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                reply_to_message_id=message_id,
                text=reply_html,
                parse_mode="HTML"
            )

            async with get_history_lock(key):
                # Reload history just in case
                if key in histories:
                    histories[key].append({"role": "assistant", "content": reply})
                else:
                    histories[key] = history + [{"role": "assistant", "content": reply}]
                touch_activity(key)
                
    except Exception as e:
        logger.error(f"❌ [red]Ошибка при генерации рецензии TikTok:[/] {e}", exc_info=True)
