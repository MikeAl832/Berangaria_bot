"""Загрузка настроек: config.yaml как источник несекретных дефолтов, env — поверх.

Секреты приходят только из окружения (.env или docker compose env_file);
всё остальное задаётся в config.yaml в корне репозитория.
"""
import os

import yaml
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from berangaria.core.paths import PROJECT_ROOT, project_path
from berangaria.prompts import MEM0_CUSTOM_INSTRUCTIONS

load_dotenv(PROJECT_ROOT / ".env")

CONFIG_FILE = project_path(os.environ.get("BOT_CONFIG_FILE") or "config.yaml")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    loaded_yaml = yaml.safe_load(f) or {}

if not isinstance(loaded_yaml, dict):
    raise ValueError("config.yaml должен содержать YAML mapping/object")

config_yaml: dict[str, object] = loaded_yaml


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_setting(env_name: str, yaml_key: str, default: int) -> int:
    return _as_int(os.environ.get(env_name, config_yaml.get(yaml_key, default)), default)


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_setting(env_name: str, yaml_key: str, default: float) -> float:
    return _as_float(os.environ.get(env_name, config_yaml.get(yaml_key, default)), default)


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _bool_setting(env_name: str, yaml_key: str, default: bool) -> bool:
    return _as_bool(os.environ.get(env_name, config_yaml.get(yaml_key, default)), default)


def _str_setting(env_name: str, yaml_key: str, default: str) -> str:
    value = os.environ.get(env_name, config_yaml.get(yaml_key, default))
    if isinstance(value, str) and value:
        return value
    return default

# ========================================
# 🔑 API КЛЮЧИ
# ========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_API_BASE_URL = os.environ.get("TELEGRAM_BOT_API_BASE_URL", "").rstrip("/")
TELEGRAM_BOT_API_BASE_FILE_URL = os.environ.get(
    "TELEGRAM_BOT_API_BASE_FILE_URL",
    f"{TELEGRAM_BOT_API_BASE_URL}/file" if TELEGRAM_BOT_API_BASE_URL else "",
).rstrip("/")
TELEGRAM_BOT_API_LOCAL_MODE = _as_bool(
    os.environ.get("TELEGRAM_BOT_API_LOCAL_MODE"),
    bool(TELEGRAM_BOT_API_BASE_URL),
)
DEEPSEEK_API_KEY = os.environ.get("API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Валидация обязательных API ключей
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env файле!")
if not DEEPSEEK_API_KEY:
    raise ValueError("API_KEY (DeepSeek) не установлен в .env файле!")

# ========================================
# 🤖 ОСНОВНАЯ МОДЕЛЬ (DeepSeek)
# ========================================
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL = config_yaml.get("model", "deepseek-v4-flash")
MAX_CONTEXT_TOKENS = config_yaml.get("max_context_tokens", 32000)
MAX_REPLY_TOKENS = config_yaml.get("max_reply_tokens", 4096)
GENERATION_PARAMS = config_yaml.get("generation_params", {"temperature": 0.9, "top_p": 0.95})
# Пониженная температура для фактических ответов (после web_search/read_url) — меньше галлюцинаций
FACTUAL_TEMPERATURE = config_yaml.get("factual_temperature", 0.3)
STREAMING_ENABLED = _bool_setting("BOT_STREAMING_ENABLED", "streaming_enabled", True)
STREAM_UPDATE_INTERVAL_SECONDS = max(
    0.25,
    min(_float_setting("BOT_STREAM_UPDATE_INTERVAL", "stream_update_interval_seconds", 0.8), 5.0),
)
STREAM_PREVIEW_MIN_CHARS = max(
    1,
    min(_int_setting("BOT_STREAM_PREVIEW_MIN_CHARS", "stream_preview_min_chars", 12), 200),
)

# ========================================
# 👁️ VISION (Gemini)
# ========================================
VISION_MODE = config_yaml.get("vision_mode", False)
GEMINI_MODEL = config_yaml.get("gemini_model", "gemini-3.5-flash-lite")
VIDEO_MAX_DURATION_SEC = config_yaml.get("video_max_duration_sec", 300)
AUDIO_MAX_DURATION_SEC = config_yaml.get("audio_max_duration_sec", 300)
# Облачный Telegram Bot API отдаёт через getFile только до 20 МБ. В local mode
# сервер снимает этот предел; оставляем настраиваемый предохранитель для диска/RAM.
_video_file_limit_default = 2 * 1024 * 1024 * 1024 if TELEGRAM_BOT_API_LOCAL_MODE else 20 * 1024 * 1024
VIDEO_MAX_FILE_SIZE_BYTES = max(
    1,
    _int_setting("BOT_VIDEO_MAX_FILE_SIZE_BYTES", "video_max_file_size_bytes", _video_file_limit_default),
)
GEMINI_UPLOAD_MAX_WAIT_SEC = config_yaml.get("gemini_upload_max_wait_sec", 180)
GEMINI_UPLOAD_BACKOFF_INITIAL = config_yaml.get("gemini_upload_backoff_initial", 0.5)
GEMINI_UPLOAD_BACKOFF_MAX = config_yaml.get("gemini_upload_backoff_max", 5.0)

# ========================================
# 🧠 ПАМЯТЬ (Mem0 + Embeddings)
# ========================================
MEM0_LLM_MODEL = config_yaml.get("mem0_llm_model", "deepseek-v4-flash")
EMBEDDING_MODEL = config_yaml.get("embedding_model", "gemini-embedding-2")
EMBEDDING_DIMS = config_yaml.get("embedding_dims", 768)
MEMORY_SEARCH_LIMIT = config_yaml.get("memory_search_limit", 5)
MEMORY_MIN_SCORE = config_yaml.get("memory_min_score", 0.3)
MEMORY_MAX_CHARS = config_yaml.get("memory_max_chars", 800)
MEMORY_FLUSH_INTERVAL_SECONDS = config_yaml.get("memory_flush_interval_seconds", 300)
MEMORY_QUERY_MIN_CHARS = config_yaml.get("memory_query_min_chars", 12)
MEMORY_QUERY_RECENT_MESSAGES = config_yaml.get("memory_query_recent_messages", 3)
MEMORY_QUEUE_BATCH_SIZE = max(
    1,
    min(_as_int(config_yaml.get("memory_queue_batch_size", 20), 20), 100),
)
# Через сколько секунд источник в статусе 'waiting' считается следом оборванного
# хода и хоронится. Должно быть заведомо больше debounce + полного retry-бюджета
# LLM-хода, иначе живой ход потеряет свою память.
MEMORY_WAITING_MAX_AGE_SECONDS = max(
    60,
    _as_int(config_yaml.get("memory_waiting_max_age_seconds", 1800), 1800),
)
# Через сколько секунд удалять завершённые строки очереди источников. `dead`
# не удаляется никогда — это форензик-след.
MEMORY_SOURCE_RETENTION_SECONDS = max(
    3600,
    _as_int(config_yaml.get("memory_source_retention_seconds", 30 * 24 * 3600), 30 * 24 * 3600),
)

# Потолок склейки debounce-буфера. Без него непрерывный поток сообщений
# собирается в одну запись истории неограниченного размера.
MAX_BUFFERED_MESSAGES = max(
    2, _as_int(config_yaml.get("max_buffered_messages", 30), 30)
)
MAX_BUFFERED_CHARS = max(
    1000, _as_int(config_yaml.get("max_buffered_chars", 20000), 20000)
)

# ========================================
# 🗄️ QDRANT (общий для mem0 и стикеров)
# ========================================
# В докере бот ходит в сервис "qdrant", с хоста — в "localhost".
# Переопределяется переменной окружения QDRANT_HOST (напр. при запуске скрипта с хоста).
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# ========================================
# 🎨 СТИКЕРЫ (векторный поиск)
# ========================================
STICKER_ENABLED = config_yaml.get("sticker_enabled", True)
STICKER_COLLECTION = config_yaml.get("sticker_collection", "stickers")
STICKER_DIMS = config_yaml.get("sticker_dims", 768)
STICKER_MIN_SCORE = config_yaml.get("sticker_min_score", 0.35)
STICKER_TOP_K = config_yaml.get("sticker_top_k", 5)
STICKER_AUTO_SYNC = config_yaml.get("sticker_auto_sync", True)
STICKER_SYNC_FILE = project_path(config_yaml.get("sticker_sync_file", "data/stickers_clean.jsonl"))
STICKER_SYNC_MAX_PER_START = config_yaml.get("sticker_sync_max_per_start", 0)
STICKER_INDEX_VERSION = config_yaml.get("sticker_index_version", 1)
# Сколько раз за один ход можно вызывать find_stickers (дальше — отказ, бери из уже найденных)
STICKER_FIND_MAX_PER_TURN = _as_int(config_yaml.get("sticker_find_max_per_turn", 3), 3)
STICKER_FIND_MAX_PER_TURN = max(1, min(STICKER_FIND_MAX_PER_TURN, 10))

# ========================================
# ⚙️ ПОВЕДЕНИЕ БОТА
# ========================================
BOT_NAMES = config_yaml.get("bot_names", ["Бер", "Ber"])
RANDOM_REPLY_CHANCE = config_yaml.get("random_reply_chance", 10)
SUMMARY_INTERVAL = config_yaml.get("summary_interval", 10)
MESSAGE_DEBOUNCE_SECONDS = config_yaml.get("message_debounce_seconds", 4.0)
RANDOM_REPLY_COOLDOWN = config_yaml.get("random_reply_cooldown", 30)
ADMIN_MODE = config_yaml.get("admin_mode", False)

# Часовой пояс бота (метки [Time:], CURRENT TIME, автосуммаризация)
_tz_name = config_yaml.get("timezone", "Europe/Moscow")
if not isinstance(_tz_name, str) or not _tz_name.strip():
    _tz_name = "Europe/Moscow"
TIMEZONE_NAME = _tz_name.strip()
try:
    BOT_TZ = ZoneInfo(TIMEZONE_NAME)
except Exception as e:
    raise ValueError(f"Некорректный timezone в config.yaml: {TIMEZONE_NAME!r} ({e})") from e

# Часы локального времени, когда гонять автосуммаризацию (напр. [5, 14] = 05:00 и 14:00 МСК)
_raw_summary_hours = config_yaml.get("summary_hours", [5, 14])
if isinstance(_raw_summary_hours, (int, float)):
    _raw_summary_hours = [int(_raw_summary_hours)]
if not isinstance(_raw_summary_hours, (list, tuple)) or not _raw_summary_hours:
    _raw_summary_hours = [5, 14]
SUMMARY_HOURS: list[int] = sorted({
    h for h in (_as_int(x, -1) for x in _raw_summary_hours) if 0 <= h <= 23
}) or [5, 14]
DEBUG = _bool_setting("BOT_DEBUG", "debug", False)
VERBOSE = _bool_setting("BOT_VERBOSE", "verbose", False)  # Суперподробные логи (включает DEBUG)
FULL_DEBUG_LOGS = DEBUG or _bool_setting("BOT_FULL_DEBUG_LOGS", "full_debug_logs", False)
LOG_FILE = project_path(_str_setting("BOT_LOG_FILE", "log_file", "bot.log"))
LOG_MAX_BYTES = _int_setting("BOT_LOG_MAX_BYTES", "log_max_bytes", 10 * 1024 * 1024)
LOG_BACKUP_COUNT = _int_setting("BOT_LOG_BACKUP_COUNT", "log_backup_count", 5)

# ========================================
# 📊 ТЕХНИЧЕСКИЕ КОНСТАНТЫ
# ========================================
MAX_API_RETRIES = 5  # Максимальное количество попыток обращения к API
MAX_TOOL_ROUNDS = 8  # Отдельный предел последовательных LLM tool-call раундов
MAX_MEDIA_ITEMS_IN_CONTEXT = 10  # Максимум медиа-элементов в одном сообщении для экономии токенов

# ========================================
# 🔐 ДОСТУП
# ========================================
ALLOWED_USERS = config_yaml.get("allowed_users", [])
ALLOWED_GROUPS = config_yaml.get("allowed_groups", [])

# Чат для алертов о критических ошибках (null = выключено)
_admin_alert_chat_id = config_yaml.get("admin_alert_chat_id", None)
if _admin_alert_chat_id is None:
    ADMIN_ALERT_CHAT_ID = None
elif isinstance(_admin_alert_chat_id, bool):
    raise ValueError("admin_alert_chat_id должен быть Telegram chat id или null")
else:
    try:
        ADMIN_ALERT_CHAT_ID = int(_admin_alert_chat_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("admin_alert_chat_id должен быть одним Telegram chat id или null") from exc

# ========================================
# 💰 ЦЕНЫ DeepSeek (за 1M токенов)
# ========================================
PRICE_PROMPT_CACHE_MISS = config_yaml.get("price_prompt_cache_miss", 0.14)
PRICE_PROMPT_CACHE_HIT = config_yaml.get("price_prompt_cache_hit", 0.0028)
PRICE_COMPLETION = config_yaml.get("price_completion", 0.28)

# ========================================
# 🧠 MEM0 КОНФИГУРАЦИЯ
# ========================================

# Отключаем телеметрию и прокси
os.environ["MEM0_TELEMETRY"] = "false"
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = 'localhost,127.0.0.1'

MEM0_CONFIG = {
    "version": "v1.1",
    "custom_instructions": MEM0_CUSTOM_INSTRUCTIONS,
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": MEM0_LLM_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "temperature": 0,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": EMBEDDING_MODEL,
            "api_key": GEMINI_API_KEY
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection_name": "mem0",
            "embedding_model_dims": EMBEDDING_DIMS
        }
    }
}
