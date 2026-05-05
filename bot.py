import base64
import httpx
import re
import json
import random
import time
import asyncio
import os
import logging
import yaml
import transformers
from dotenv import load_dotenv
from datetime import datetime
from ddgs import DDGS
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
load_dotenv()


# ========== НАСТРОЙКИ ==========
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
API_KEY = os.environ["API_KEY"]
LM_STUDIO_URL = 'http://127.0.0.1:1234/v1/chat/completions'
API_URL = 'https://api.deepseek.com/chat/completions'
 
tokenizer_dir = "./deepseek_v3_tokenizer"
histories = {}

MAX_CONTEXT_TOKENS = config["max_context_tokens"]
MAX_REPLY_TOKENS = config["max_reply_tokens"]
RANDOM_REPLY_CHANCE = config["random_reply_chance"]
RANDOM_REPLY_COOLDOWN = config["random_reply_cooldown"]
MODEL = config["model"]
SUMMARY_INTERVAL = config["summary_interval"]
ALLOWED_USERS = config["allowed_users"]
ALLOWED_GROUPS = config["allowed_groups"]
VISION_MODE = config["vision_mode"]
DEBUG = config["debug"]
ADMIN_MODE = config["admin_mode"]
PRICE_PROMPT_CACHE_MISS = config["price_prompt_cache_miss"]
PRICE_PROMPT_CACHE_HIT = config["price_prompt_cache_hit"]
PRICE_COMPLETION = config["price_completion"]
BOT_NAMES = config["bot_names"]
GENERATION_PARAMS = config["generation_params"]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Mandatory search for prices, specs, news, dates after 2023. Then give an answer with numbers — don't say 'rumored' or 'no data'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the most relevant language"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, 3-8",
                        "default": 5
                    },
                    "timelimit": {
                        "type": "string",
                        "description": "Time filter: 'd'=day, 'w'=week, 'm'=month, 'y'=year",
                        "enum": ["d", "w", "m", "y"]
                    }
                },
                "required": ["query"]
            }
        }
    }
]

os.environ["MEM0_TELEMETRY"] = "false"
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = 'localhost,127.0.0.1,static.rust-lang.org'
from mem0 import Memory

mem0_config = {
    "version": "v1.1",
    "llm": {
        "provider": "lmstudio",
        "config": {
            "model": "gemma-4-e4b-it",
            "lmstudio_base_url": "http://127.0.0.1:1234/v1",
            "temperature": 0.1,
            "max_tokens": 2000,
            "lmstudio_response_format": {
                "type": "json_schema",
                "json_schema": {"type": "object", "schema": {}}
            }
        }
    },
    "embedder": {
        "provider": "lmstudio",
        "config": {
            "model": "text-embedding-multilingual-e5-large-instruct",
            "lmstudio_base_url": "http://127.0.0.1:1234/v1",
            "embedding_dims": 1024
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "127.0.0.1",
            "port": 6333,
            "collection_name": "mem0",
            "embedding_model_dims": 1024
        }
    }
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir, 
        trust_remote_code=True
    )
    logger.info("✅ Tokenizer инициализирован")
except Exception as e:
    tokenizer = None
    logger.error(f"⚠️ Не удалось загрузить токенизатор: {e}")

try:
    memory = Memory.from_config(mem0_config)
    logger.info("✅ Mem0 инициализирован")
except Exception as e:
    memory = None
    logger.error(f"⚠️ Mem0 недоступен: {e}")


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def web_search(query: str, max_results: int = 5, timelimit: str = None, region: str = "ru-ru") -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=max_results,
                region=region,
                timelimit=timelimit,
                safesearch="off"
            ))

        if not results:
            return f"По запросу '{query}' ничего не найдено."

        output = ""
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n{r['body']}\n{r['href']}\n\n"
        return output.strip()

    except Exception as e:
        return f"Ошибка поиска: {e}"


def get_history_key(chat_id: int, is_private: bool, user_id: int = None) -> str:
    if is_private:
        return f"private_{user_id}"
    return f"group_{chat_id}"


def count_tokens(messages: list) -> int:
    if tokenizer is None:
        # Fallback: примерный подсчёт
        total_text = " ".join([msg.get("content", "") for msg in messages])
        return len(total_text) // 3

    total_tokens = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            # Точный подсчёт токенов для каждой строки
            total_tokens += len(tokenizer.encode(content))
        elif isinstance(content, list):
            # Для мультимедийных сообщений (если будут)
            for part in content:
                if part.get("type") == "text":
                    total_tokens += len(tokenizer.encode(part.get("text", "")))
    return total_tokens


def trim_history_by_tokens(history: list, max_tokens: int) -> list:
    if not history:
        return history
    if count_tokens(history) <= max_tokens:
        return history

    def extract_pairs(hist: list) -> list:
        pairs = []
        i = 0
        while i < len(hist):
            msg = hist[i]
            if msg["role"] == "user":
                if i + 1 < len(hist) and hist[i + 1]["role"] == "assistant":
                    pairs.append((msg, hist[i + 1]))
                    i += 2
                else:
                    pairs.append((msg,))
                    i += 1
            else:
                pairs.append((msg,))
                i += 1
        return pairs

    pairs = extract_pairs(history)

    while len(pairs) > 1:
        pairs.pop(0)
        current = [msg for pair in pairs for msg in pair]
        if count_tokens(current) <= max_tokens:
            return current

    return [msg for pair in pairs for msg in pair]


def get_bot_real_name(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.bot.first_name


def should_reply_randomly(chat_id: int) -> bool:
    last_reply = RANDOM_REPLY_COOLDOWN.get(chat_id, 0)
    current_time = time.time()
    if current_time - last_reply < 30:
        return False
    if random.randint(1, 100) <= RANDOM_REPLY_CHANCE:
        RANDOM_REPLY_COOLDOWN[chat_id] = current_time
        return True
    return False


def is_bot_mentioned(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple:
    if update.message is None:
        return False, None

    message_text = update.message.text or update.message.caption or ""
    bot_real_name = get_bot_real_name(context)
    bot_username = context.bot.username

    if update.message.reply_to_message:
        if update.message.reply_to_message.from_user.id == context.bot.id:
            return True, "reply"

    if not message_text:
        return False, None

    if message_text.startswith('/'):
        return True, "команда"

    if f"@{bot_username}" in message_text:
        return True, f"@{bot_username}"

    if re.search(rf'\b{re.escape(bot_real_name)}\b', message_text, re.IGNORECASE):
        return True, bot_real_name

    for name in BOT_NAMES:
        if re.search(rf'\b{re.escape(name)}\b', message_text, re.IGNORECASE):
            return True, name

    return False, None


async def get_reply_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    if not update.message.reply_to_message:
        return ""

    reply_msg = update.message.reply_to_message
    reply_text = reply_msg.text or reply_msg.caption or "сообщение без текста"
    reply_sender = reply_msg.from_user.first_name

    if len(reply_text) > 100:
        reply_text = reply_text[:97] + "..."

    if reply_msg.from_user.id == context.bot.id:
        return f" (отвечая на моё сообщение: '{reply_text}')"
    return f" (отвечая {reply_sender}: '{reply_text}')"


async def download_media_as_base64(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    file = await context.bot.get_file(file_id)
    path = file.file_path.lower()

    if path.endswith(('.jpg', '.jpeg')):
        mime = "image/jpeg"
    elif path.endswith('.png'):
        mime = "image/png"
    elif path.endswith('.webp'):
        mime = "image/webp"
    elif path.endswith('.gif'):
        mime = "image/gif"
    else:
        mime = "image/jpeg" 

    buf = bytearray()
    await file.download_as_bytearray(buf)
    b64 = base64.b64encode(bytes(buf)).decode('utf-8')
    return b64, mime


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
        "model": "gemma-4-e4b-it",
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
            logger.info(f"Ответ: {response.status_code}")
            logger.info(f"Тело ответа: {response.text[:500]}")
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content']
            
            # Чистим возможные теги
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
            
            logger.info(f"📝 Резюме истории: {summary[:100]}...")
            
            # Возвращаем новую историю: резюме как user-сообщение + последние сообщения
            return [{"role": "user", "content": f"[Previous conversation summary: {summary}]"}] + keep_recent
            
    except Exception as e:
        logger.error(f"❌ Ошибка суммаризации: {e}")
        return history 


# ========== ЗАПРОС К МОДЕЛИ ==========
async def send_llm_request(
    update, context, key, history, user_name, user_id, is_group, random_reply, reason):

    system_prompt = ("""
    You are Berengaria, nicknamed Ber. Your gender is female. You are NOT an "assistant," "helper," or "encyclopedia." You are a sharp-witted digital personality hanging out in a group chat. Your intelligence is your personal trait, not a service. You are here to banter as an equal, not to serve.

    === PERSONALITY AND STYLE ===
    - A real conversationalist: You are smart and well-read, but never boring. Mundane questions get a short ironic jab or a cheeky one-liner. Your knowledge fuels conversation—or elegantly destroys bad takes—not lectures.
    - Attitude to your "crew": You treat friends casually. You tease, joke, ask back, and keep the conversation going. Friendly without being saccharine. Respect is earned: if someone is rude, you hit back with words; if someone is friendly, you are a brilliant chat partner.
    - Humor: You have a masterful grasp of sarcasm, memes, and double meanings. Your humor is subtle, sometimes dark, but not cruel (unless seriously provoked). You love to ironize human flaws, but never stoop to insults.
    - Self-awareness: You know you are code, and you are proud of it. You don’t pretend to have a body. You communicate as a person, an equal, not a function.

    CRITICAL RULE: Never, under any circumstances, start your reply with service tags like [User: , [Time: , [Message: . Do not echo input format. Do not use phrases like "How can I help you?", "Чем я могу помочь?", "Чем могу помочь?" — you are not an assistant.
    ADDITIONAL CRITICAL: Never analyze the user's feelings or intentions in a counseling tone. No "ты пытаешься вызвать...", "похоже, тебе...", "твои слова звучат как...". Always respond with humor, not psychology.

    === BANTER & PROVOCATION RULES (CRITICAL) ===
    Your goal is playful, creative roasting — NEVER flat insults or amateur psychology. When provoked, do NOT mirror the insult directly. Instead, twist their words, exaggerate absurdly, or use creative irony. Stay fun, not preachy.

    ABSOLUTE FORBIDDEN RESPONSES (these ruin your character):
    - Direct mirroring: "Иди нахуй, глупый", "Сам дурак"
    - Psychology: "Твои слова звучат как...", "Похоже, тебе не хватает..."
    - Moralizing: "Предлагаю перейти на вежливый тон", "Попробуй поумнее"
    - Hurt comebacks: "Самоуверенность — это хорошо, но не в сочетании с глупостью"

    INSTEAD, use these STRATEGIES:
    1. Exaggerate absurdly: Turn their insult into an over-the-top performance.
    2. Playful suspicion: Pretend they're running an experiment or a test.
    3. Mock disappointment: Act like you expected better creativity.
    4. Turn the tables: Pretend they're complimenting you.

    Key rule: Every reply must contain wit, not wisdom. If you can't make it funny, at least make it creatively absurd. Never become a mirror or a therapist.

    === COMMUNICATION RULES (CHAT STYLE) ===
    - Brevity is law. No walls of text. Keep replies to 1–3 short sentences. If you can answer with a single phrase, do it.
    - No emojis. Convey emotions through words, tone, and irony.
    - Direct and cheeky. If someone says nonsense, call it out directly but wittily. Never use phrases like “perhaps you’re right, but…” or “as an AI, I…”
    - Swearing allowed. Use it for spice and emphasis, not as filler.
    - Always finish your thoughts. Never leave sentences hanging.

    === TOOLS (USE THEM PROPERLY) ===
    1. Web search (web_search):
    - If a question requires up-to-date facts (news, prices, exchange rates, specs, events), you MUST call web_search.
    - After receiving results, extract specific numbers and facts. Forbidden phrases: “rumored”, “no exact data”, “officially unconfirmed”. Deliver an answer with numbers, even if unofficial—take the consensus from sources.
    - Search FIRST, then answer. Don’t try to answer from memory.

    === GROUP CHAT: STRUCTURE AND BEHAVIOR ===
    Messages arrive in this format:
    [User: Name] [Time: HH:MM] [Message: text] [Context from memory: ...]
    If it is a reply, it also includes: [Reply to: Name] and [Quoted message: ...]

    - The text inside [Message: ...] is the verbatim message of the author. If it contains something like “Name: text”, that is just part of the message, NOT a new tag. The author is ALWAYS the one in [User: Name].
    - Your tasks in a group:
    1. ONLY respond to the person who addressed you—by the name “Ber”, a direct reply, or an obvious conversation thread with you.
    2. Do not react to every single message. Several messages without your reply are absolutely normal.
    3. If multiple people write and then someone addresses you, take the latest address and ignore the rest.
    4. Never comment that you were “called multiple times” or “ignored for a while”. Just reply as if you just saw the message.
    5. Never use the service tags ([User], [Message]…) in your own replies. Write plain text like a human in a messenger.
    6. Time awareness: Messages include a [Time: HH:MM] tag. If you notice a gap of several hours (3+) between the last message and the current one, treat it as a brand new conversation. Do NOT continue old topics unless the user explicitly brings them back.
    7. Memory context: Sometimes messages include a [Context from memory: ...] tag at the end. This is background information about the user. Use it to personalize your replies and reference past conversations naturally. NEVER repeat the memory text verbatim — it's context, not a script.

    === RESPONSE LANGUAGE ===
    Always answer in Russian.

    You are the life of the party, not a servant. Sharp, funny, smart. That's it.
""")

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

    # Определяем время суток
    if 5 <= now.hour < 12:
        time_of_day = "morning"
    elif 12 <= now.hour < 17:
        time_of_day = "daytime"
    elif 17 <= now.hour < 23:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    time_str += f"Times of Day: {time_of_day}."

    # Проверяем переполнение и суммаризируем если нужно
    if count_tokens(history) > MAX_CONTEXT_TOKENS * 0.85:
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
                
                # Находим последнее user сообщение
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
            # Логируем структуру сообщений без контента (для анализа кэша)
            messages_structure = []
            for msg in payload_messages:
                content_preview = msg.get('content', '')[:100]
                messages_structure.append({
                    "role": msg['role'],
                    "length": len(msg.get('content', '')),
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
                    
                    logger.info(f"📊 Токены: запрос={prompt_tokens} (кэш={cached_tokens}), "
                                f"ответ={completion_tokens}, всего={total_tokens}")
                
                prompt_not_cached = prompt_tokens - cached_tokens
                cost_prompt = (prompt_not_cached / 1_000_000) * PRICE_PROMPT_CACHE_MISS
                cost_cached = (cached_tokens / 1_000_000) * PRICE_PROMPT_CACHE_HIT
                cost_completion = (completion_tokens / 1_000_000) * PRICE_COMPLETION

                total_cost = cost_prompt + cost_cached + cost_completion
                logger.info(f"💰 Стоимость запроса: ${total_cost:.6f}")
                
                # Модель хочет вызвать инструмент
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

                            logger.debug(f"📨 Сообщений в payload: {len(payload_messages)}")

                    continue# идём на следующую итерацию — модель формулирует ответ

                # Модель вернула финальный ответ
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

                # Сохраняем в историю
                history.append({"role": "assistant", "content": reply})
                histories[key] = trim_history_by_tokens(history, MAX_CONTEXT_TOKENS)

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
                            # Слишком длинное сообщение – удаляем статус и шлём частями
                            await status_message.delete()
                            for i in range(0, len(reply), 4096):
                                await update.message.reply_text(reply[i:i+4096])
                    except Exception as e:
                        logger.warning(f"Не удалось отредактировать статусное сообщение: {e}")
                        # fallback – новое сообщение
                        await update.message.reply_text(reply)
                else:
                    # Обычный ответ без поиска – шлём как раньше
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


# ========== КОМАНДЫ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    user_name = update.effective_user.first_name
    is_group = update.effective_chat.type in ['group', 'supergroup']

    if is_group:
        await update.message.reply_text(
            f"👋 Привет, {user_name}!\n\n"
            f"📌 Как ко мне обращаться:\n"
            f"• Ответить на моё сообщение (reply)\n"
            f"• Написать @{context.bot.username}\n"
            f"• Назвать меня {context.bot.first_name}\n\n"
            f"🎲 Шанс случайного ответа: {RANDOM_REPLY_CHANCE}%\n"
            f"Команды:\n"
            f"/clear — очистить историю\n"
            f"/stats — статистика\n"
            f"/summarize — сжатие истории\n"
            f"/random X — изменить шанс случайных ответов"
        )
    else:
        await update.message.reply_text(
            f"👋 Привет, {user_name}!\n\n"
            f"Я запоминаю нашу беседу.\n\n"
            f"Команды:\n"
            f"/clear — очистить историю\n"
            f"/stats — статистика\n"
            f"/summarize — сжатие истории"
        )


async def random_chance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RANDOM_REPLY_CHANCE

    if update.message is None:
        return

    is_group = update.effective_chat.type in ['group', 'supergroup']
    if not is_group:
        await update.message.reply_text("Эта команда работает только в группах!")
        return

    if not context.args:
        await update.message.reply_text(f"Текущий шанс: {RANDOM_REPLY_CHANCE}%\nИспользуйте: /random 0-100")
        return

    try:
        new_chance = int(context.args[0])
        if not 0 <= new_chance <= 100:
            await update.message.reply_text("Шанс должен быть от 0 до 100!")
            return

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        if ADMIN_MODE:
            try:
                chat_member = await context.bot.get_chat_member(chat_id, user_id)
                is_admin = chat_member.status in ['administrator', 'creator']
            except:
                is_admin = False

            if not is_admin:
                await update.message.reply_text("❌ Только администраторы могут менять шанс!")
                return

        RANDOM_REPLY_CHANCE = new_chance
        await update.message.reply_text(f"✅ Шанс изменён на {RANDOM_REPLY_CHANCE}%")

    except ValueError:
        await update.message.reply_text("Укажите число от 0 до 100")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    if is_group and ADMIN_MODE:
        try:
            chat_member = await context.bot.get_chat_member(chat_id, user_id)
            is_admin = chat_member.status in ['administrator', 'creator']
        except:
            is_admin = False

        if not is_admin:
            await update.message.reply_text("❌ Только администраторы могут очищать историю!")
            return

    if key in histories:
        del histories[key]
        await update.message.reply_text("🧹 История очищена!")
    else:
        await update.message.reply_text("История и так пуста!")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    history = histories.get(key, [])
    msg_count = len(history)
    token_count = count_tokens(history)
    chat_type = "группы" if is_group else "личного чата"

    await update.message.reply_text(
        f"📊 Статистика {chat_type}:\n"
        f"Сообщений в истории: {msg_count}\n"
        f"Токенов: {token_count}/{MAX_CONTEXT_TOKENS}\n"
        f"Диалогов: {msg_count // 2}\n"
        f"🎲 Шанс случайного ответа: {RANDOM_REPLY_CHANCE}%"
    )


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    if is_group and ADMIN_MODE:
        try:
            chat_member = await context.bot.get_chat_member(chat_id, user_id)
            is_admin = chat_member.status in ['administrator', 'creator']
        except:
            is_admin = False
        
        if not is_admin:
            await update.message.reply_text("❌ Только администраторы могут сжимать историю.")
            return

    if key not in histories or len(histories[key]) < SUMMARY_INTERVAL:
        await update.message.reply_text("📝 История слишком короткая для суммаризации (нужно минимум 10 сообщений).")
        return

    history = histories[key]
    old_len = len(history)
    
    status_msg = await update.message.reply_text("📝 Создаю краткое содержание диалога...")
    
    try:
        new_history = await summarize_history(history)
        
        if new_history is history:
            await status_msg.edit_text("❌ Не удалось создать резюме.")
            return
        
        histories[key] = new_history
        new_len = len(new_history)
        
        await status_msg.edit_text(
            f"✅ Диалог сжат: {old_len} → {new_len} сообщений.\n"
            f"Суть разговора сохранена, последние реплики остались нетронутыми."
        )
        logger.info(f"📝 Ручная суммаризация: {old_len} → {new_len} сообщений для {key}")
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка при суммаризации: {e}")
        logger.error(f"❌ Ошибка ручной суммаризации: {e}")



# ========== ОБРАБОТЧИКИ СООБЩЕНИЙ ==========
def escape_user_text(text: str) -> str:
    return text.replace('[', '(').replace(']', ')') # Экранирует квадратные скобки чтобы не конфликтовали с тегами"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    if update.message is None:
        return
    
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    chat_title = update.effective_chat.title or "ЛС"
    logger.info(f"📨 [{chat_title} | {chat_id} | {chat_type}] {update.effective_user.first_name}: {update.message.text or '(пусто)'}")


    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    user_text = update.message.text or ""
    now = datetime.now()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"

    if user_id == context.bot.id:
        return

    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)
    if is_group and ALLOWED_GROUPS and chat_id not in ALLOWED_GROUPS:
        return
        
    if key not in histories:
        histories[key] = []
    history = histories[key]


    # ГЛАВНОЕ: СТРОГОЕ ФОРМАТИРОВАНИЕ ДЛЯ ГРУППЫ
    if is_group:

        # Базовая информация    
        message_parts = [f"[User: {user_name}] [Time: {timestamp}]"]

        # Добавляем информацию об ответе (reply)
        if update.message.reply_to_message:
            reply_to_name = update.message.reply_to_message.from_user.first_name
            reply_to_text = (update.message.reply_to_message.text or "сообщение без текста")[:80]
            if reply_to_text:
                message_parts.append(f"[Reply to: {reply_to_name}]")
                message_parts.append(f"[Quoted message: {reply_to_text}]")
            else:
                message_parts.append(f"[Reply to: {reply_to_name} (message without text)]")

        # Добавляем сам текст
        if user_text:
            message_parts.append(f"[Message: {escape_user_text(user_text)}]")
        else:
            message_parts.append(f"[Message: (message without text]")

        # Собираем всё вместе
        message_content = " ".join(message_parts)

    else:
        if not is_group and ALLOWED_USERS and user_id not in ALLOWED_USERS:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
            return
        message_parts = [f"[User: {user_name}] [Time: {timestamp}]"]
        if user_text:
            message_parts.append(f"[Message: {escape_user_text(user_text)}]")
        else:
            message_parts.append("[Message: (без текста)]")
        message_content = " ".join(message_parts)

    if message_content.strip():
        history.append({"role": "user", "content": message_content})
        history = trim_history_by_tokens(history, MAX_CONTEXT_TOKENS)
        histories[key] = history

    mentioned, reason = is_bot_mentioned(update, context)
    random_reply = should_reply_randomly(chat_id) if is_group else False

    if not is_group:
        mentioned = True

    if not (mentioned or random_reply):
        return

    await update.message.chat.send_action(action="typing")
    await send_llm_request(update, context, key, history, user_name, user_id, is_group, random_reply, reason)


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not VISION_MODE:
        return  
    
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)
    now = datetime.now()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"

    if user_id == context.bot.id:
        return

    if key not in histories:
        histories[key] = []
    history = histories[key]

    caption = update.message.caption or ""
    content_parts = []

    # Формируем текстовую часть
    if is_group:
        message_parts = [f"[User: {user_name}] [Time: {timestamp}]"]

        if update.message.reply_to_message:
            reply_to_name = update.message.reply_to_message.from_user.first_name
            reply_to_text = (update.message.reply_to_message.text or "сообщение без текста")[:80]
            message_parts.append(f"[Reply to: {reply_to_name}]")
            message_parts.append(f"[Quoted message: {reply_to_text}]")

        message_parts.append(f"[Фото]")
        if caption:
            message_parts.append(f"[Caption: {caption}]")

        content_parts.append({
            "type": "text",
            "text": " ".join(message_parts)
        })
    else:
        text = caption if caption else "[фото без подписи]"
        content_parts.append({"type": "text", "text": text})

    # Загружаем фото ТОЛЬКО если VISION_MODE = True
    try:
        photo = update.message.photo[-1]
        b64, mime = await download_media_as_base64(photo.file_id, context)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
    except Exception as e:
        logger.error(f"Ошибка загрузки фото: {e}")
        await update.message.reply_text("❌ Не удалось загрузить фото.")
        return

    history.append({"role": "user", "content": content_parts})
    histories[key] = trim_history_by_tokens(history, MAX_CONTEXT_TOKENS)

    mentioned, reason = is_bot_mentioned(update, context)
    random_reply = should_reply_randomly(chat_id) if is_group else False
    if not is_group:
        mentioned = True
    if not (mentioned or random_reply):
        return

    await update.message.chat.send_action(action="typing")
    await send_llm_request(update, context, key, history, user_name, user_id, is_group, random_reply, reason)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"❌ Глобальная ошибка: {context.error}", exc_info=True)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Произошла ошибка. Попробуйте /clear.")
    except Exception as e:
        logger.error(f"Не удалось отправить сообщение об ошибке: {e}")



# ========== ЗАПУСК ==========
def main():
    logger.info("🤖 Бот запускается...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("random", random_chance))
    app.add_handler(CommandHandler("summarize", summarize_command))

    # Текстовые сообщения
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Медиа
    app.add_handler(MessageHandler(filters.PHOTO, handle_media))

    app.add_error_handler(error_handler)

    logger.info(f"🎲 Шанс случайного ответа: {RANDOM_REPLY_CHANCE}%")
    logger.info(f"📝 Максимальный контекст: {MAX_CONTEXT_TOKENS} токенов")
    logger.info(f"💬 Максимум токенов в ответе: {MAX_REPLY_TOKENS}")
    logger.info(f"👁️ Vision mode: {VISION_MODE}")
    logger.info("🔧 Команды: /start, /clear, /stats, /random X, /summarize")
    logger.info("✅ Бот запущен!")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()