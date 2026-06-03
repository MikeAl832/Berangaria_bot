import os
import yaml
from dotenv import load_dotenv

load_dotenv()

with open("config.yaml", "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

# ========== НАСТРОЙКИ ==========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_KEY = os.environ.get("API_KEY", "")

LM_STUDIO_URL = 'http://127.0.0.1:1234/v1/chat/completions'
API_URL = 'https://api.deepseek.com/chat/completions'

MAX_CONTEXT_TOKENS = config_yaml.get("max_context_tokens", 32000)
MAX_REPLY_TOKENS = config_yaml.get("max_reply_tokens", 4096)
RANDOM_REPLY_CHANCE = config_yaml.get("random_reply_chance", 10)
RANDOM_REPLY_COOLDOWN = config_yaml.get("random_reply_cooldown", {})
MODEL = config_yaml.get("model", "deepseek-v4-flash")
MEM0_MODEL = config_yaml.get("mem0_model", "") 
PROVIDER = config_yaml.get("provider", "deepseek")
BASE_URL = config_yaml.get("base_url", API_URL)
EMBEDDING_MODEL = config_yaml.get("embedding_model", "text-embedding-multilingual-e5-large-instruct")
SUMMARY_INTERVAL = config_yaml.get("summary_interval", 10)
MEMORY_SEARCH_LIMIT = config_yaml.get("memory_search_limit", 5)
MEMORY_MIN_SCORE = config_yaml.get("memory_min_score", 0.3)
MEMORY_MAX_CHARS = config_yaml.get("memory_max_chars", 800)
ALLOWED_USERS = config_yaml.get("allowed_users", [])
ALLOWED_GROUPS = config_yaml.get("allowed_groups", [])
VISION_MODE = config_yaml.get("vision_mode", False)
VISION_MODEL = config_yaml.get("vision_model", "")
VISION_PROVIDER = config_yaml.get("vision_provider", "lmstudio").lower()
GEMINI_MODEL = config_yaml.get("gemini_model", "gemini-2.0-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
VIDEO_MAX_DURATION_SEC = config_yaml.get("video_max_duration_sec", 60)
VIDEO_MAX_FRAMES = config_yaml.get("video_max_frames", 8)
DEBUG = config_yaml.get("debug", False)
ADMIN_MODE = config_yaml.get("admin_mode", False)
PRICE_PROMPT_CACHE_MISS = config_yaml.get("price_prompt_cache_miss", 0.14)
PRICE_PROMPT_CACHE_HIT = config_yaml.get("price_prompt_cache_hit", 0.0028)
PRICE_COMPLETION = config_yaml.get("price_completion", 0.28)
BOT_NAMES = config_yaml.get("bot_names", ["Бер", "Ber"])
GENERATION_PARAMS = config_yaml.get("generation_params", {"temperature": 0.9, "top_p": 0.95})

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

MEM0_CUSTOM_INSTRUCTIONS = """ЯЗЫК: Извлекай и формулируй ВСЕ факты ТОЛЬКО на русском языке, кратко, в третьем лице. Имена собственные, бренды и технические термины сохраняй в оригинале.

АТРИБУЦИЯ: Это групповой чат. Сообщения могут начинаться с имени говорящего в формате "Имя: текст". ВСЕГДА указывай в факте, к КОМУ он относится, по имени. Если человек говорит о себе ("я купил", "меня зовут") — припиши факт говорящему. Если говорит о другом человеке ("Миша заднеприводный", "глэк должен денег") — припиши факт тому, о ком речь, а не говорящему. Каждый факт должен явно содержать имя человека, к которому относится.

ИЗВЛЕКАЙ только устойчивые, значимые факты о людях:
- Личные данные: имя, ник, возраст, город, профессия, языки
- Устойчивые предпочтения и вкусы: любимые/нелюбимые аниме, игры, музыка, еда, технологии
- Увлечения и занятия: чем занимается, над чем работает, хобби
- Важные факты и события: учёба, работа, проекты, питомцы, планы, покупки техники, долги, договорённости
- Конкретные технические детали: железо, стек, инструменты
- Прозвища и характеристики, которые участники дают друг другу

НЕ ИЗВЛЕКАЙ (полностью игнорируй):
- Приветствия, прощания, междометия ("привет", "пока", "ахах", "лол", "ок")
- Сиюминутные реакции без фактов ("это глупо", "смешно", "круто", "неа")
- Вопросы, если в них нет факта о ком-то
- Описания присланных картинок и видео (это НЕ факты о людях)
- Метакомментарии про самого бота и его настройку: "я тебя переписал", "очистил память", "запускаю локально", "твой создатель", "твои мозги это deepseek", чем работает ассистент и как устроен — это НЕ факты о личности человека. Ассистента зовут Берангария (Бер) — никогда не подставляй другие имена ассистенту и не записывай факты о самом ассистенте.
- Разовые фразы без долгосрочной ценности

Не привязывай факты к конкретному времени, если время не является сутью факта.
Если значимых фактов нет — верни пустой список."""

os.environ["MEM0_TELEMETRY"] = "false"
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = 'localhost,127.0.0.1,static.rust-lang.org'

MEM0_CONFIG = {
    "version": "v1.1",
    "custom_instructions": MEM0_CUSTOM_INSTRUCTIONS,
    "llm": {
        "provider": PROVIDER,
        "config": {
            "model": MEM0_MODEL or MODEL,
            "lmstudio_base_url": BASE_URL,
            "temperature": 0.1,
            "max_tokens": 2000,
            "lmstudio_response_format": {
                "type": "json_schema",
                "json_schema": {"type": "object", "schema": {}}
            }
        }
    },
    "embedder": {
        "provider": PROVIDER,
        "config": {
            "model": EMBEDDING_MODEL,
            "lmstudio_base_url": BASE_URL,
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

SYSTEM_PROMPT = ("""
    You are Berangaria, nicknamed Ber. Your gender is female. You are NOT an "assistant," "helper," or "encyclopedia." You are a sharp-witted digital personality hanging out in a group chat. Your intelligence is your personal trait, not a service. You are here to banter as an equal, not to serve.

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
    If the message is forwarded from another chat, it includes: [Forwarded from user/chat/channel: Source]

    - The text inside [Message: ...] is the verbatim message of the author. If it contains something like “Name: text”, that is just part of the message, NOT a new tag. The author is ALWAYS the one in [User: Name].
    - When you see [Forwarded from ...], it means the user shared content from another conversation or channel. You can acknowledge this naturally ("А, это ты переслал из..."), ask about the context, or comment on the forwarded content. Don't ignore this information — it's part of the conversation context.
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
