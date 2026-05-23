import re
import json
import logging
import asyncio
import httpx
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    LM_STUDIO_URL, LOCAL_MODELS, SUMMARY_INTERVAL, VISION_MODE, MAX_CONTEXT_TOKENS, 
    MAX_REPLY_TOKENS, MODEL, GENERATION_PARAMS, TOOLS, API_KEY, API_URL,
    DEBUG, PRICE_PROMPT_CACHE_MISS, PRICE_PROMPT_CACHE_HIT, PRICE_COMPLETION, SYSTEM_PROMPT
)
from state import histories, chat_tokens
from memory_store import memory
from tools import web_search

logger = logging.getLogger(__name__)

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
        "model": LOCAL_MODELS[0] if LOCAL_MODELS else MODEL,
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
        "max_tokens": 1500,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50 
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(LM_STUDIO_URL, json=summary_payload)
            logger.info(f"Ответ сумморизации: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content']
            
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
            logger.info(f"📝 Резюме истории: {summary[:100]}...")
            
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
            === VISION CAPABILITIES ===
            If the user sends a photo, you can see it. Describe what you notice, make witty observations about it, but never sound like a technical image analysis tool. 
            Your comments should be natural and humorous — like a friend looking at a picture over your shoulder.
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
            query = history[-1].get('content', '') if history else user_name
            if isinstance(query, list):
                query = next((p.get('text', '') for p in query if p.get('type') == 'text'), '')

            mem_results = await asyncio.wait_for(
                asyncio.to_thread(
                    memory.search,
                    query,
                    filters={"user_id": str(user_id)},
                    limit=5
                ),
                timeout=30.0
            )

            if mem_results and mem_results.get('results'):
                mem_text = "\n".join([f"- {m['memory']}" for m in mem_results['results']])
                
                if payload_messages[-1]["role"] == "user":
                    last_content = payload_messages[-1]["content"]
                    payload_messages[-1] = {
                        "role": "user",
                        "content": f"{last_content}\n\n[Context from memory:{mem_text}]"
                    }

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
                content_preview = str(msg.get('content', ''))[:100]
                messages_structure.append({
                    "role": msg['role'],
                    "length": len(str(msg.get('content', ''))),
                    "preview": content_preview
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
                            status_message = await update.message.reply_text(
                                f"🔍 Выполняю поиск: *{args['query']}*...",
                                parse_mode="Markdown"
                            )
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
                            last_user_msg = history[-2].get('content', '') if len(history) >= 2 else ''
                            if isinstance(last_user_msg, list):
                                last_user_msg = next((p.get('text', '') for p in last_user_msg if p.get('type') == 'text'), '')
                            
                            await asyncio.to_thread(
                                memory.add,
                                [
                                    {"role": "user", "content": last_user_msg},
                                    {"role": "assistant", "content": reply}
                                ],
                                user_id=str(user_id)
                            )
                            logger.info(f"✅ Память сохранена для user_id={user_id}")
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
