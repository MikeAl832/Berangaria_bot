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

TOKENIZER_DIR = "./deepseek_v3_tokenizer"

MAX_CONTEXT_TOKENS = config_yaml.get("max_context_tokens", 32000)
MAX_REPLY_TOKENS = config_yaml.get("max_reply_tokens", 4096)
RANDOM_REPLY_CHANCE = config_yaml.get("random_reply_chance", 10)
RANDOM_REPLY_COOLDOWN = config_yaml.get("random_reply_cooldown", {})
MODEL = config_yaml.get("model", "deepseek-v4-flash")
SUMMARY_INTERVAL = config_yaml.get("summary_interval", 10)
ALLOWED_USERS = config_yaml.get("allowed_users", [])
ALLOWED_GROUPS = config_yaml.get("allowed_groups", [])
VISION_MODE = config_yaml.get("vision_mode", False)
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

os.environ["MEM0_TELEMETRY"] = "false"
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = 'localhost,127.0.0.1,static.rust-lang.org'

MEM0_CONFIG = {
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
