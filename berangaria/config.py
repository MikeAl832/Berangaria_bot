"""Settings loading: config.yaml holds the non-secret defaults, env overrides them.

Secrets come only from the environment (.env or docker compose env_file);
everything else is set in config.yaml at the repository root.
"""
import copy
import os
from zoneinfo import ZoneInfo

import yaml
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


def _required_str(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} должен быть непустой строкой в config.yaml")
    return value.strip()


def _required_price(mapping: dict[str, object], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"{key} не задан в config.yaml")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} должен быть неотрицательным числом в config.yaml")
    number = float(value)
    if number < 0:
        raise ValueError(f"{key} должен быть неотрицательным числом в config.yaml")
    return number

# ========================================
# 🔑 API КЛЮЧИ
# ========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DEEPSEEK_API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY", "")
    or os.environ.get("API_KEY", "")
).strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ========================================
# 🤖 ОСНОВНАЯ МОДЕЛЬ (DeepSeek Direct или OpenRouter)
# ========================================
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_URL = f"{DEEPSEEK_API_BASE}/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

CHAT_PROVIDER = _str_setting(
    "BOT_CHAT_PROVIDER",
    "chat_provider",
    "deepseek",
).strip().lower()
if CHAT_PROVIDER not in {"deepseek", "openrouter"}:
    raise ValueError(
        f"chat_provider должен быть 'deepseek' или 'openrouter' (получено {CHAT_PROVIDER!r})"
    )

_default_chat_url = (
    OPENROUTER_API_URL if CHAT_PROVIDER == "openrouter" else DEEPSEEK_API_URL
)
CHAT_API_URL = _str_setting(
    "CHAT_API_URL",
    "chat_api_url",
    _default_chat_url,
)
MODEL = _required_str(config_yaml, "model")
OPENROUTER_HTTP_REFERER = "https://github.com/MikeAl832/Berangaria_bot"
OPENROUTER_APP_TITLE = "Berangaria"
OPENROUTER_API_KEY = (
    os.environ.get("OPENROUTER_API_KEY", "")
    or os.environ.get("CHAT_API_KEY", "")
).strip()

if CHAT_PROVIDER == "openrouter":
    CHAT_API_KEY = OPENROUTER_API_KEY
else:
    CHAT_API_KEY = (
        os.environ.get("CHAT_API_KEY", "").strip()
        or DEEPSEEK_API_KEY
    )

# Валидация обязательных API ключей
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env файле!")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY (или API_KEY) не установлен в .env файле!")
if not CHAT_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY не установлен в .env файле (требуется для chat_provider=openrouter)!"
    )


MAX_CONTEXT_TOKENS = config_yaml.get("max_context_tokens", 32000)
MAX_REPLY_TOKENS = config_yaml.get("max_reply_tokens", 4096)
GENERATION_PARAMS = config_yaml.get(
    "generation_params",
    {"temperature": 1.0, "thinking": {"type": "disabled"}},
)
SUMMARY_GENERATION_PARAMS = config_yaml.get(
    "summary_generation_params",
    {"temperature": 0.3, "thinking": {"type": "enabled"}},
)
_raw_openrouter_provider = config_yaml.get("openrouter_provider")
OPENROUTER_PROVIDER: dict[str, object] | None = (
    copy.deepcopy(_raw_openrouter_provider)
    if isinstance(_raw_openrouter_provider, dict)
    else None
)
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
# 👁️ VISION (Native Chat Vision + Gemini)
# ========================================
VISION_MODE = config_yaml.get("vision_mode", False)
IMAGE_VISION_PROVIDER = _str_setting(
    "BOT_IMAGE_VISION_PROVIDER",
    "image_vision_provider",
    "chat",
).strip().lower()
if IMAGE_VISION_PROVIDER not in {"chat", "gemini"}:
    raise ValueError(
        "image_vision_provider должен быть 'chat' или 'gemini' "
        f"(получено {IMAGE_VISION_PROVIDER!r})"
    )
GEMINI_MODEL = config_yaml.get("gemini_model", "gemini-3.5-flash-lite")
VIDEO_MAX_DURATION_SEC = config_yaml.get("video_max_duration_sec", 300)
AUDIO_MAX_DURATION_SEC = config_yaml.get("audio_max_duration_sec", 300)
_TELEGRAM_GETFILE_LIMIT_BYTES = 20 * 1024 * 1024
VIDEO_MAX_FILE_SIZE_BYTES = max(
    1,
    min(
        _int_setting(
            "BOT_VIDEO_MAX_FILE_SIZE_BYTES",
            "video_max_file_size_bytes",
            _TELEGRAM_GETFILE_LIMIT_BYTES,
        ),
        _TELEGRAM_GETFILE_LIMIT_BYTES,
    ),
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
# 🗄️ QDRANT (серверный или локальный на диске)
# ========================================
# В Docker Compose передаётся QDRANT_HOST=qdrant; локально по умолчанию 127.0.0.1.
# Если задан QDRANT_PATH (напр. при локальном запуске без Docker), используется
# встроенный дисковый Qdrant без отдельного контейнера.
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_raw_qdrant_path = str(
    os.environ.get("QDRANT_PATH") or config_yaml.get("qdrant_path") or ""
).strip()
QDRANT_PATH = str(project_path(_raw_qdrant_path)) if _raw_qdrant_path else ""

# ========================================
# 🎨 СТИКЕРЫ (векторный поиск)
# ========================================
STICKER_ENABLED = config_yaml.get("sticker_enabled", True)
STICKER_COLLECTION = config_yaml.get("sticker_collection", "stickers")
STICKER_DIMS = config_yaml.get("sticker_dims", 768)
STICKER_MIN_SCORE = config_yaml.get("sticker_min_score", 0.25)
STICKER_TOP_K = config_yaml.get("sticker_top_k", 5)
STICKER_AUTO_SYNC = config_yaml.get("sticker_auto_sync", True)
STICKER_SYNC_FILE = project_path(
    config_yaml.get("sticker_sync_file", "data/stickers_clean.json")
)
STICKER_SYNC_MAX_PER_START = config_yaml.get("sticker_sync_max_per_start", 0)
# v3: JSON catalogue + richer sticker_text (situation/use_cases/secondary…).
STICKER_INDEX_VERSION = _as_int(config_yaml.get("sticker_index_version", 3), 3)
# One-shot send_sticker(query) attempts per turn (search+send). Default 2 = one
# refined retry if the first query misses; success ends the turn immediately.
_sticker_send_cap = config_yaml.get(
    "sticker_send_max_per_turn",
    config_yaml.get("sticker_find_max_per_turn", 2),
)
STICKER_SEND_MAX_PER_TURN = max(1, min(_as_int(_sticker_send_cap, 2), 5))

# ========================================
# 🔍 SEARCH
# ========================================
SEARCH_PROVIDER = _str_setting(
    "BOT_SEARCH_PROVIDER",
    "search_provider",
    "deepseek",
).strip().lower()
if SEARCH_PROVIDER not in {"deepseek", "ddgs"}:
    raise ValueError(
        f"search_provider должен быть 'deepseek' или 'ddgs' (получено {SEARCH_PROVIDER!r})"
    )
# How many times web_search may run in one turn. The prompt asks for far more
# fact-checking, and without a ceiling a single runaway turn eats the
# process-global rate limit (10/min), i.e. breaks search for every other chat.
WEB_SEARCH_MAX_PER_TURN = max(1, min(_as_int(config_yaml.get("web_search_max_per_turn", 2), 2), 10))
# Page bodies are much larger than search snippets and are echoed into every
# following provider round, so cap both page reads and the combined web budget.
READ_URL_MAX_PER_TURN = max(1, min(_as_int(config_yaml.get("read_url_max_per_turn", 4), 4), 20))
WEB_TOOL_MAX_PER_TURN = max(1, min(_as_int(config_yaml.get("web_tool_max_per_turn", 6), 6), 30))

# ========================================
# 💬 MULTI-MESSAGE REPLIES (send_messages)
# ========================================
# Caps for the terminal send_messages tool: natural short bursts, not spam.
MULTI_MESSAGE_MAX = max(2, min(_as_int(config_yaml.get("multi_message_max", 5), 5), 5))
MULTI_MESSAGE_MAX_CHARS = max(
    40, min(_as_int(config_yaml.get("multi_message_max_chars", 280), 280), 1000)
)
MULTI_MESSAGE_MAX_TOTAL_CHARS = max(
    80,
    min(_as_int(config_yaml.get("multi_message_max_total_chars", 600), 600), 3000),
)
MULTI_MESSAGE_DELAY_MIN = max(
    0.1, min(_as_float(config_yaml.get("multi_message_delay_min", 0.4), 0.4), 5.0)
)
MULTI_MESSAGE_DELAY_MAX = max(
    MULTI_MESSAGE_DELAY_MIN,
    min(_as_float(config_yaml.get("multi_message_delay_max", 2.0), 2.0), 10.0),
)
MULTI_MESSAGE_DELAY_TOTAL_CAP = max(
    MULTI_MESSAGE_DELAY_MAX,
    min(_as_float(config_yaml.get("multi_message_delay_total_cap", 5.0), 5.0), 30.0),
)
# Approximate typing speed used only to scale inter-bubble pause by length.
MULTI_MESSAGE_CHARS_PER_SEC = max(
    8.0, min(_as_float(config_yaml.get("multi_message_chars_per_sec", 28.0), 28.0), 80.0)
)

# ========================================
# ⚙️ ПОВЕДЕНИЕ БОТА
# ========================================
BOT_NAMES = config_yaml.get("bot_names", ["Бер", "Ber"])
RANDOM_REPLY_CHANCE = config_yaml.get("random_reply_chance", 10)
SUMMARY_INTERVAL = config_yaml.get("summary_interval", 10)
SUMMARY_MAX_CHARS = max(500, _as_int(config_yaml.get("summary_max_chars", 4000), 4000))
# Scheduled job only: skip a chat unless at least this many messages sit
# beyond the keep window. /summarize and the 85% token path ignore this.
SUMMARY_MIN_EXTRA = max(
    2,
    _as_int(config_yaml.get("summary_min_extra", 10), 10),
)
# Scheduled job only: postpone once if the chat spoke this recently.
SUMMARY_QUIET_SECONDS = max(
    0.0,
    min(
        _as_float(config_yaml.get("summary_quiet_seconds", 600.0), 600.0),
        3600.0,
    ),
)
MESSAGE_DEBOUNCE_SECONDS = config_yaml.get("message_debounce_seconds", 4.0)
RANDOM_REPLY_COOLDOWN = config_yaml.get("random_reply_cooldown", 30)
RANDOM_REPLY_RECENT_WINDOW_SECONDS = max(
    1.0,
    min(
        _as_float(config_yaml.get("random_reply_recent_window_seconds", 120), 120),
        3600.0,
    ),
)
RANDOM_REPLY_IDLE_TARGET_SECONDS = max(
    RANDOM_REPLY_RECENT_WINDOW_SECONDS,
    min(
        _as_float(config_yaml.get("random_reply_idle_target_seconds", 600), 600),
        86400.0,
    ),
)
RANDOM_REPLY_PRESENCE_SECONDS = max(
    0.0,
    min(
        _as_float(config_yaml.get("random_reply_presence_seconds", 600), 600),
        86400.0,
    ),
)
RANDOM_REPLY_PRESENCE_MULTIPLIER = max(
    1.0,
    min(
        _as_float(config_yaml.get("random_reply_presence_multiplier", 2.0), 2.0),
        5.0,
    ),
)
ADMIN_MODE = config_yaml.get("admin_mode", False)

# Часовой пояс бота (метки [Time:], отдельная строка даты в payload, автосуммаризация)
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
LOG_MESSAGE_PREVIEW_CHARS = max(
    40,
    min(
        _int_setting(
            "BOT_LOG_MESSAGE_PREVIEW_CHARS",
            "log_message_preview_chars",
            400,
        ),
        2000,
    ),
)

# ========================================
# 📊 ТЕХНИЧЕСКИЕ КОНСТАНТЫ
# ========================================
MAX_API_RETRIES = 5  # Максимальное количество попыток обращения к API
# Ceiling on consecutive LLM tool-call rounds. A turn that verifies a fact AND
# looks for a sticker uses up to eight rounds on its own (search + refined retry
# + read_url + send_sticker + a reaction), and on overflow the code aborts the
# turn with an error posted to the chat. The headroom is what keeps ordinary
# behaviour from hitting that wall.
MAX_TOOL_ROUNDS = 12
MAX_MEDIA_ITEMS_IN_CONTEXT = 10  # Максимум медиа-элементов в одном сообщении для экономии токенов

# ========================================
# 🔐 ДОСТУП
# ========================================
ALLOWED_USERS = config_yaml.get("allowed_users", [])
ALLOWED_GROUPS = config_yaml.get("allowed_groups", [])
# The first allowlisted user is the authenticated owner. This keeps the ID in
# one source of truth instead of duplicating it in a second setting.
OWNER_USER_ID = (
    _as_int(ALLOWED_USERS[0], 0)
    if isinstance(ALLOWED_USERS, (list, tuple)) and ALLOWED_USERS
    else 0
) or None

TELEGRAM_MEDIA_TIMEOUT_SECONDS = max(
    5.0,
    _float_setting("TELEGRAM_MEDIA_TIMEOUT_SECONDS", "telegram_media_timeout_seconds", 120.0),
)
TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES = min(
    _TELEGRAM_GETFILE_LIMIT_BYTES,
    max(
        1,
        _int_setting(
            "TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES",
            "telegram_bot_api_download_max_bytes",
            _TELEGRAM_GETFILE_LIMIT_BYTES,
        ),
    ),
)


# Явный чат для критических алертов; null означает личный чат OWNER_USER_ID.
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
# 💰 ЦЕНЫ основной чат-модели (за 1M токенов)
# ========================================
PRICE_PROMPT_CACHE_MISS = _required_price(config_yaml, "price_prompt_cache_miss")
PRICE_PROMPT_CACHE_HIT = _required_price(config_yaml, "price_prompt_cache_hit")
PRICE_PROMPT_CACHE_WRITE = _required_price(config_yaml, "price_prompt_cache_write")
PRICE_COMPLETION = _required_price(config_yaml, "price_completion")


def chat_api_headers(*, session_id: str | None = None) -> dict[str, str]:
    """Build HTTP headers for the active chat provider."""
    headers = {
        "Authorization": f"Bearer {CHAT_API_KEY}",
        "Content-Type": "application/json",
    }
    if CHAT_PROVIDER == "openrouter":
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        headers["X-Title"] = OPENROUTER_APP_TITLE
        if session_id:
            headers["x-session-id"] = session_id
    return headers


def apply_chat_gateway(payload: dict, *, session_id: str | None) -> dict:
    """Attach optional OpenRouter routing fields when chat_provider=='openrouter'."""
    if CHAT_PROVIDER == "openrouter":
        if session_id:
            payload["session_id"] = session_id
        if isinstance(OPENROUTER_PROVIDER, dict) and OPENROUTER_PROVIDER:
            payload["provider"] = copy.deepcopy(OPENROUTER_PROVIDER)
    return payload

# ========================================
# 🧠 MEM0 КОНФИГУРАЦИЯ
# ========================================

# Отключаем телеметрию и прокси
os.environ["MEM0_TELEMETRY"] = "false"
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = 'localhost,127.0.0.1'

_qdrant_vector_config = (
    {
        "path": os.path.join(QDRANT_PATH, "mem0"),
        "host": None,
        "port": None,
        "on_disk": True,
        "collection_name": "mem0",
        "embedding_model_dims": EMBEDDING_DIMS,
    }
    if QDRANT_PATH
    else {
        "host": QDRANT_HOST,
        "port": QDRANT_PORT,
        "collection_name": "mem0",
        "embedding_model_dims": EMBEDDING_DIMS,
    }
)

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
        "config": _qdrant_vector_config,
    }
}
