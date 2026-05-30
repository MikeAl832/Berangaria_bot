import re
import json
import logging
import asyncio
import httpx
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    LM_STUDIO_URL, SUMMARY_INTERVAL, VISION_MODE, MAX_CONTEXT_TOKENS, 
    MAX_REPLY_TOKENS, MODEL, MEM0_MODEL, GENERATION_PARAMS, TOOLS, API_KEY, API_URL,
    DEBUG, PRICE_PROMPT_CACHE_MISS, PRICE_PROMPT_CACHE_HIT, PRICE_COMPLETION, SYSTEM_PROMPT,
    MEMORY_SEARCH_LIMIT, MEMORY_MIN_SCORE, MEMORY_MAX_CHARS,
)
from state import histories, chat_tokens, api_call_count
from memory_store import memory
from tools import web_search

logger = logging.getLogger(__name__)


def _extract_plain_text(content) -> str:
    """
    Достаёт чистый текст пользователя из сообщения для поиска по памяти:
    отбрасывает [Image description: ...] / [Video description: ...] / [Context from memory: ...]
    и служебные теги, оставляя только содержимое [Message: ...] (или сырой текст).
    """
    if isinstance(content, list):
        content = next((p.get('text', '') for p in content if p.get('type') == 'text'), '')
    if not isinstance(content, str):
        return ''

    # Убираем тяжёлые блоки описаний медиа и контекста памяти
    text = re.sub(r'\[Image description:.*?\]', '', content, flags=re.DOTALL)
    text = re.sub(r'\[Video description:.*?\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Context from memory:.*?\]', '', text, flags=re.DOTALL)

    # Если есть [Message: ...] — берём только его содержимое
    msg_match = re.search(r'\[Message:\s*(.*?)\]', text, flags=re.DOTALL)
    if msg_match:
        return msg_match.group(1).strip()

    # Иначе чистим оставшиеся служебные теги
    text = re.sub(r'\[(?:User|Time|Reply to|Quoted message):.*?\]', '', text, flags=re.DOTALL)
    return text.strip()


def _format_memory_block(mem_results: dict) -> str:
    """
    Формирует компактный блок памяти:
    - фильтрует по порогу релевантности MEMORY_MIN_SCORE
    - берёт топ MEMORY_SEARCH_LIMIT
    - ограничивает суммарную длину MEMORY_MAX_CHARS
    Возвращает готовый текст (без обёртки) или '' если ничего релевантного.
    """
    results = (mem_results or {}).get('results') or []
    if not results:
        return ''

    # Отсортируем по score по убыванию (если score есть)
    def _score(item):
        return item.get('score', 0.0) or 0.0

    results = sorted(results, key=_score, reverse=True)

    lines = []
    total = 0
    for item in results:
        # Порог релевантности применяем только если score реально присутствует
        if 'score' in item and item['score'] is not None and item['score'] < MEMORY_MIN_SCORE:
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
        "model": MEM0_MODEL or MODEL,
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(LM_STUDIO_URL, json=summary_payload)
            logger.info(f"Ответ сумморизации: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content']
            
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
            if DEBUG:
                logger.info(f"📝 Резюме истории:\n{summary}")
            else:
                logger.info(f"📝 Резюме истории получено ({len(summary)} символов)")
            
            return [{"role": "user", "content": f"[Previous conversation summary: {summary}]"}] + keep_recent
            
    except Exception as e:
        logger.error(f"❌ Ошибка суммаризации: {e}")
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

    if chat_tokens.get(key, 0) > MAX_CONTEXT_TOKENS * 0.85:
        logger.info(f"📝 Автосуммаризация для key={key}")
        history = await summarize_history(history)
        histories[key] = history
        
    system_prompt += f"\n\n=== CURRENT TIME ===\n{time_str}\n"
    payload_messages = [{"role": "system", "content": system_prompt}] + history

    if memory:
        try:
            # C: ищем по чистому тексту пользователя, без описаний картинок/видео и тегов
            query = _extract_plain_text(history[-1].get('content', '')) if history else user_name
            if not query:
                query = user_name

            # Память партиционируется по чату (key): в группе — общая на весь чат
            # (group_<chat_id>), в личке — на пользователя (private_<user_id>).
            mem_results = await asyncio.wait_for(
                asyncio.to_thread(
                    memory.search,
                    query,
                    filters={"user_id": key},
                    limit=MEMORY_SEARCH_LIMIT
                ),
                timeout=30.0
            )

            # A + D: фильтрация по релевантности, ограничение количества и длины, аккуратный формат
            mem_text = _format_memory_block(mem_results)
            if mem_text and payload_messages[-1]["role"] == "user":
                last_content = payload_messages[-1]["content"]
                payload_messages[-1] = {
                    "role": "user",
                    "content": f"{last_content}\n\n[Context from memory:\n{mem_text}\n]"
                }
                if DEBUG:
                    logger.info(f"🧠 Память ({mem_text.count(chr(10)) + 1} фактов, {len(mem_text)} символов) scope={key} по запросу: {query[:80]}")

        except asyncio.TimeoutError:
            logger.warning("⚠️ Память: таймаут поиска, продолжаем без неё")
        except Exception as e:
            logger.error(f"⚠️ Ошибка получения памяти: {e}")

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
            messages_structure = []
            for msg in payload_messages:
                content_full = str(msg.get('content', ''))
                messages_structure.append({
                    "role": msg['role'],
                    "length": len(content_full),
                    "content": content_full
                })
            logger.info(f"📤 Структура запроса: {json.dumps(messages_structure, indent=2, ensure_ascii=False)}")

        for _ in range(5): 
            payload = {
                "model": MODEL,
                "messages": payload_messages,
                "max_tokens": MAX_REPLY_TOKENS,
                "tools": TOOLS,
                **GENERATION_PARAMS 
            }

            try:
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(API_URL, json=payload, headers=headers)

                if response.status_code == 400:
                    histories[key] = []
                    logger.error(f"400: {response.text}")
                    await update.message.reply_text("⚠️ История сброшена. Напишите ещё раз.")
                    return

                if response.status_code != 200:
                    await update.message.reply_text(f"❌ Ошибка API: {response.status_code}")
                    return

                data = response.json()
                choice = data['choices'][0]
                finish_reason = choice.get('finish_reason', '')
                message = choice['message']
                usage = data.get('usage', {})
                prompt_tokens = 0
                completion_tokens = 0
                cached_tokens = 0

                if usage:
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    prompt_details = usage.get('prompt_tokens_details', {})
                    cached_tokens = prompt_details.get('cached_tokens', 0)
                    
                    chat_tokens[key] = total_tokens
                    
                    logger.info(f"📊 Токены: запрос={prompt_tokens} (кэш={cached_tokens}), "
                                f"ответ={completion_tokens}, всего={total_tokens}")
                
                prompt_not_cached = prompt_tokens - cached_tokens
                cost_prompt = (prompt_not_cached / 1_000_000) * PRICE_PROMPT_CACHE_MISS
                cost_cached = (cached_tokens / 1_000_000) * PRICE_PROMPT_CACHE_HIT
                cost_completion = (completion_tokens / 1_000_000) * PRICE_COMPLETION

                total_cost = cost_prompt + cost_cached + cost_completion
                logger.info(f"💰 Стоимость запроса: ${total_cost:.6f}")
                
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
                                    logger.warning(f"Не удалось отредактировать статусное сообщение: {e}")

                            logger.info(f"🔍 Поиск: {args['query']}")

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
                    reply += "\n\n_(ответ обрезан)_"

                history.append({"role": "assistant", "content": reply})
                histories[key] = history

                if memory:
                    async def save_memory_background():
                        try:
                            # B1 + B3: в память кладём ТОЛЬКО чистую реплику пользователя
                            # (без описаний картинок/видео и без реплик ассистента).
                            raw_user_msg = history[-2].get('content', '') if len(history) >= 2 else ''
                            last_user_msg = _extract_plain_text(raw_user_msg)

                            if not last_user_msg:
                                # Нечего извлекать (например, было только фото без подписи) — пропускаем
                                logger.info(f"🧠 Память: нет текста для сохранения (scope={key}), пропуск")
                                return

                            # В группе подставляем имя говорящего, чтобы mem0 корректно
                            # атрибутировал факт нужному человеку (в общей памяти чата).
                            if is_group:
                                speaker_msg = f"{user_name}: {last_user_msg}"
                            else:
                                speaker_msg = last_user_msg

                            await asyncio.to_thread(
                                memory.add,
                                [
                                    {"role": "user", "content": speaker_msg}
                                ],
                                user_id=key
                            )
                            logger.info(f"✅ Память сохранена (scope={key})")
                        except Exception as e:
                            logger.error(f"⚠️ Ошибка сохранения памяти: {e}")
                    
                    asyncio.create_task(save_memory_background()) 

                if is_group and not random_reply and reason != "reply":
                    reply = f"{reply}"

                if status_message:
                    try:
                        if len(reply) <= 4096:
                            await status_message.edit_text(reply)
                        else:
                            await status_message.delete()
                            for i in range(0, len(reply), 4096):
                                await update.message.reply_text(reply[i:i+4096])
                    except Exception as e:
                        logger.warning(f"Не удалось отредактировать статусное сообщение: {e}")
                        await update.message.reply_text(reply)
                else:
                    if len(reply) <= 4096:
                        await update.message.reply_text(reply)
                    else:
                        for i in range(0, len(reply), 4096):
                            await update.message.reply_text(reply[i:i+4096])
                return

            except httpx.ConnectError:
                logger.error("❌ API недоступен!")
                await update.message.reply_text("❌ API недоступен!")
                return
            except httpx.TimeoutException:
                logger.error("❌ Таймаут запроса к API")
                await update.message.reply_text("❌ Таймаут.")
                return
            except Exception as e:
                logger.error(f"❌ Ошибка в обработке запроса: {e}")
                await update.message.reply_text("❌ Ошибка при обработке.")
                return
