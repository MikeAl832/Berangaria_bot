# Глобальные состояния бота
import asyncio
import time
import json
import sqlite3
import logging
import os
from collections import OrderedDict
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# История сообщений по ключу (private_X или group_Y)
histories: Dict[str, List[Dict[str, Any]]] = {}

# Тайм-ауты для рандомных ответов
random_reply_cooldown: Dict[int, float] = {}

# Изменяемый шанс случайного ответа (для команды /random)
random_reply_chance: int = 10  # default, загружается из config при старте

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
        if current_time - last_time > max_age_seconds:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        histories.pop(key, None)
        chat_tokens.pop(key, None)
        api_call_count.pop(key, None)
        last_activity.pop(key, None)
        pending_memory.pop(key, None)  # Очищаем буфер памяти
        _history_locks.pop(key, None)  # Очищаем блокировки
        delete_history(key)            # Удаляем и из БД

    return len(keys_to_remove)


# ========================================
# 💾 ПЕРСИСТЕНТНОСТЬ ИСТОРИИ (SQLite)
# ========================================
# Хранит историю диалогов на диске, чтобы рестарт/деплой не стирал контекст.
DB_PATH = os.environ.get("BOT_DB_PATH", "bot_state.db")


def _db_execute(query: str, params: tuple = (), fetch: bool = False):
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
    """Создаёт таблицу истории, если её ещё нет."""
    try:
        _db_execute(
            "CREATE TABLE IF NOT EXISTS histories ("
            "key TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать БД {DB_PATH}: {e}")


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
