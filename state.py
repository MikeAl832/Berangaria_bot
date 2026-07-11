# Глобальные состояния бота
import asyncio
import time
import json
import sqlite3
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Sequence, TypeAlias

logger = logging.getLogger(__name__)

HistoryEntry: TypeAlias = Dict[str, Any]
HistoryList: TypeAlias = List[HistoryEntry]

# История сообщений по ключу (private_X или group_Y)
histories: Dict[str, HistoryList] = {}

# Тайм-ауты для рандомных ответов
random_reply_cooldown: Dict[int, float] = {}

DEFAULT_RANDOM_REPLY_CHANCE = 10


class RuntimeSettingKey(str, Enum):
    RANDOM_REPLY_CHANCE = "random_reply_chance"


@dataclass(frozen=True)
class RuntimeSettings:
    random_reply_chance: int


# Изменяемый шанс случайного ответа (для команды /random)
random_reply_chance: int = DEFAULT_RANDOM_REPLY_CHANCE  # загружается из config/БД при старте

# Последнее известное количество токенов для чата (из API)
chat_tokens: Dict[str, int] = {}

# Счётчик количества вызовов API для каждого ключа
api_call_count: Dict[str, int] = {}

# Буфер сообщений для склеивания (debounce)
# Формат: { "chat_id_user_id": { "messages": [...], "task": asyncio.Task } }
message_buffer: Dict[str, Dict[str, Any]] = {}

# Буфер реплик юзеров, ожидающих сохранения в долговременную память (Mem0).
# Накапливается до лимита по символам, затем флашится батчем. Формат: { key: ["Имя: текст", ...] }
pending_memory: Dict[str, List[str]] = {}
memory_flush_tasks: set[asyncio.Task[None]] = set()

# Кэш описаний медиа: file_unique_id -> описание от vision-модели.
# Позволяет не анализировать повторно одно и то же изображение/стикер заново.
MEDIA_CACHE_MAX = 500
media_description_cache: "OrderedDict[str, str]" = OrderedDict()


def get_cached_media_description(file_unique_id: str) -> str | None:
    """Возвращает закэшированное описание медиа или None."""
    if not file_unique_id:
        return None
    desc = media_description_cache.get(file_unique_id)
    if desc is not None:
        media_description_cache.move_to_end(file_unique_id)  # LRU: освежаем
    return desc


def cache_media_description(file_unique_id: str, description: str) -> None:
    """Кэширует описание медиа с ограничением размера (LRU)."""
    if not file_unique_id or not description:
        return
    media_description_cache[file_unique_id] = description
    media_description_cache.move_to_end(file_unique_id)
    while len(media_description_cache) > MEDIA_CACHE_MAX:
        media_description_cache.popitem(last=False)

# Последнее время активности чата (для очистки старых данных)
last_activity: Dict[str, float] = {}

# Блокировки для предотвращения race conditions
_history_locks: Dict[str, asyncio.Lock] = {}
_turn_locks: Dict[str, asyncio.Lock] = {}
_buffer_lock: asyncio.Lock = asyncio.Lock()

def get_history_key(chat_id: int, is_private: bool, user_id: int = None) -> str:
    if is_private:
        return f"private_{user_id}"
    return f"group_{chat_id}"

def get_history_lock(key: str) -> asyncio.Lock:
    """Получает или создаёт блокировку для конкретного ключа истории."""
    if key not in _history_locks:
        _history_locks[key] = asyncio.Lock()
    return _history_locks[key]


def get_turn_lock(key: str) -> asyncio.Lock:
    """Сериализует полный ход одного чата: user -> LLM/tools -> assistant."""
    if key not in _turn_locks:
        _turn_locks[key] = asyncio.Lock()
    return _turn_locks[key]

def touch_activity(key: str):
    """Обновляет время последней активности для чата."""
    last_activity[key] = time.time()

def cleanup_old_chats(max_age_hours: int = 72) -> int:
    """
    Удаляет данные чатов, неактивных более max_age_hours часов.
    Предотвращает утечку памяти для редко используемых чатов.
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    keys_to_remove = []

    for key, last_time in list(last_activity.items()):
        turn_lock = _turn_locks.get(key)
        if current_time - last_time > max_age_seconds and not (turn_lock and turn_lock.locked()):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        histories.pop(key, None)
        chat_tokens.pop(key, None)
        api_call_count.pop(key, None)
        last_activity.pop(key, None)
        pending_memory.pop(key, None)  # Очищаем буфер памяти
        _history_locks.pop(key, None)  # Очищаем блокировки
        _turn_locks.pop(key, None)
        delete_history(key)            # Удаляем и из БД

    return len(keys_to_remove)


# ========================================
# 💾 ПЕРСИСТЕНТНОСТЬ (SQLite)
# ========================================
# Хранит историю диалогов и runtime-настройки на диске, чтобы рестарт/деплой
# не стирал контекст и изменения команд управления.
DB_PATH = os.environ.get("BOT_DB_PATH", "bot_state.db")


def _db_execute(
    query: str,
    params: Sequence[Any] = (),
    fetch: bool = False,
) -> List[tuple[Any, ...]] | None:
    """Выполняет запрос с гарантированным закрытием соединения. Возвращает строки при fetch=True."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(query, params)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        conn.close()


def init_db() -> None:
    """Создаёт таблицы истории и runtime-настроек, если их ещё нет."""
    try:
        _db_execute(
            "CREATE TABLE IF NOT EXISTS histories ("
            "key TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        _db_execute(
            "CREATE TABLE IF NOT EXISTS runtime_settings ("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL, "
            "updated_at REAL NOT NULL)"
        )
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать БД {DB_PATH}: {e}")


def validate_random_reply_chance(value: Any) -> int:
    """Возвращает валидный шанс случайного ответа или бросает ValueError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("random_reply_chance должен быть целым числом")
    if not 0 <= value <= 100:
        raise ValueError("random_reply_chance должен быть от 0 до 100")
    return value


def _load_runtime_setting(key: RuntimeSettingKey) -> Any | None:
    rows = _db_execute(
        "SELECT value FROM runtime_settings WHERE key=?",
        (key.value,),
        fetch=True,
    )
    if not rows:
        return None
    return json.loads(rows[0][0])


def _save_runtime_setting(key: RuntimeSettingKey, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False)
    _db_execute(
        "INSERT INTO runtime_settings(key, value, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key.value, data, time.time()),
    )


def load_runtime_settings(default_random_reply_chance: int) -> RuntimeSettings:
    """
    Загружает изменяемые настройки из БД.

    Если настройка ещё не сохранялась, берётся значение из config.yaml.
    """
    global random_reply_chance

    try:
        default_chance = validate_random_reply_chance(default_random_reply_chance)
    except ValueError as e:
        logger.warning(
            f"⚠️ Некорректный random_reply_chance в config.yaml: {e}; "
            f"использую {DEFAULT_RANDOM_REPLY_CHANCE}%"
        )
        default_chance = DEFAULT_RANDOM_REPLY_CHANCE

    chance = default_chance
    try:
        stored = _load_runtime_setting(RuntimeSettingKey.RANDOM_REPLY_CHANCE)
        if stored is not None:
            chance = validate_random_reply_chance(stored)
    except Exception as e:
        logger.warning(
            f"⚠️ Не удалось загрузить сохранённый шанс случайного ответа: {e}; "
            "использую config.yaml"
        )

    random_reply_chance = chance
    return RuntimeSettings(random_reply_chance=chance)


def set_random_reply_chance(value: int) -> int:
    """Валидирует, применяет и сохраняет шанс случайного ответа."""
    global random_reply_chance

    chance = validate_random_reply_chance(value)
    _save_runtime_setting(RuntimeSettingKey.RANDOM_REPLY_CHANCE, chance)
    random_reply_chance = chance
    return chance


def load_all_histories() -> int:
    """Загружает все истории из БД в память при старте. Возвращает количество чатов."""
    try:
        rows = _db_execute("SELECT key, data FROM histories", fetch=True)
    except Exception as e:
        logger.error(f"❌ Не удалось загрузить истории из БД: {e}")
        return 0

    loaded = 0
    now = time.time()
    for key, data in (rows or []):
        try:
            histories[key] = json.loads(data)
            last_activity[key] = now  # чтобы свежезагруженные не вычистились сразу
            loaded += 1
        except Exception as e:
            logger.warning(f"⚠️ Пропущена битая запись истории '{key}': {e}")
    return loaded


def save_history(key: str) -> None:
    """Сохраняет (upsert) историю одного чата в БД."""
    try:
        data = json.dumps(histories.get(key, []), ensure_ascii=False)
        _db_execute(
            "INSERT INTO histories(key, data) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET data=excluded.data",
            (key, data),
        )
    except Exception as e:
        logger.warning(f"⚠️ Не удалось сохранить историю '{key}': {e}")


def delete_history(key: str) -> None:
    """Удаляет историю чата из БД."""
    try:
        _db_execute("DELETE FROM histories WHERE key=?", (key,))
    except Exception as e:
        logger.warning(f"⚠️ Не удалось удалить историю '{key}' из БД: {e}")
