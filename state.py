# Глобальные состояния бота
import asyncio
import time
import json
import sqlite3
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
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
MEMORY_SOURCE_MAX_ATTEMPTS = 5
_MEMORY_SCOPE_RE = re.compile(r"^(?:private|group)_-?\d+$")


def is_valid_memory_scope(scope: str) -> bool:
    """Проверяет канонический ключ области долговременной памяти."""
    return bool(_MEMORY_SCOPE_RE.fullmatch(scope or ""))


class RuntimeSettingKey(str, Enum):
    RANDOM_REPLY_CHANCE = "random_reply_chance"


@dataclass(frozen=True)
class RuntimeSettings:
    random_reply_chance: int


@dataclass(frozen=True)
class MemorySourceRecord:
    id: int
    scope: str
    author_id: str
    author_name: str
    message_id: int
    created_at: float
    text: str
    status: str
    attempts: int
    last_error: str | None


@dataclass(frozen=True)
class MemoryFactRecord:
    id: int
    scope: str
    subject_id: str
    fact_key: str
    fact: str
    source_id: int
    source_quote: str
    source_message_id: int
    source_created_at: float
    mem0_id: str


@dataclass(frozen=True)
class MemoryFactWrite:
    scope: str
    subject_id: str
    fact_key: str
    fact: str
    source_id: int
    source_quote: str
    source_message_id: int
    source_created_at: float
    mem0_id: str


# Изменяемый шанс случайного ответа (для команды /random)
random_reply_chance: int = DEFAULT_RANDOM_REPLY_CHANCE  # загружается из config/БД при старте

# Последнее известное количество токенов для чата (из API)
chat_tokens: Dict[str, int] = {}

# Счётчик количества вызовов API для каждого ключа
api_call_count: Dict[str, int] = {}

# Буфер сообщений для склеивания (debounce)
# Формат: { "chat_id_user_id": { "messages": [...], "task": asyncio.Task } }
message_buffer: Dict[str, Dict[str, Any]] = {}

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
        _db_execute(
            "CREATE TABLE IF NOT EXISTS memory_sources ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "scope TEXT NOT NULL, "
            "author_id TEXT NOT NULL, "
            "author_name TEXT NOT NULL, "
            "message_id INTEGER NOT NULL, "
            "created_at REAL NOT NULL, "
            "text TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'pending', "
            "attempts INTEGER NOT NULL DEFAULT 0, "
            "last_error TEXT, "
            "queued_at REAL NOT NULL, "
            "updated_at REAL NOT NULL)"
        )
        _db_execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_sources_status "
            "ON memory_sources(status, id)"
        )
        _db_execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_sources_scope_message "
            "ON memory_sources(scope, message_id)"
        )
        _db_execute(
            "CREATE TABLE IF NOT EXISTS memory_facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "scope TEXT NOT NULL, "
            "subject_id TEXT NOT NULL, "
            "fact_key TEXT NOT NULL, "
            "fact TEXT NOT NULL, "
            "source_id INTEGER NOT NULL, "
            "source_quote TEXT NOT NULL, "
            "source_message_id INTEGER NOT NULL, "
            "source_created_at REAL NOT NULL, "
            "mem0_id TEXT NOT NULL, "
            "created_at REAL NOT NULL, "
            "updated_at REAL NOT NULL, "
            "UNIQUE(scope, subject_id, fact_key))"
        )
        _db_execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_facts_scope "
            "ON memory_facts(scope, subject_id)"
        )
        _db_execute(
            "UPDATE memory_sources SET status='pending', updated_at=? "
            "WHERE status='processing'",
            (time.time(),),
        )
        _db_execute(
            "UPDATE memory_sources SET status='abandoned', text='', "
            "last_error='turn interrupted before delivery', updated_at=? "
            "WHERE status='waiting'",
            (time.time(),),
        )
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать БД {DB_PATH}: {e}")


def insert_memory_source(
    *,
    scope: str,
    author_id: str,
    author_name: str,
    message_id: int,
    created_at: float,
    text: str,
    status: str = "pending",
) -> int:
    """Долговечно ставит одно сообщение в очередь проверки памяти."""
    if status not in {"pending", "waiting"}:
        raise ValueError("некорректный статус нового источника памяти")
    now = time.time()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO memory_sources "
            "(scope, author_id, author_name, message_id, created_at, text, "
            "status, attempts, queued_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (scope, author_id, author_name, message_id, created_at, text, status, now, now),
        )
        if cur.rowcount == 0:
            row = conn.execute(
                "SELECT id FROM memory_sources WHERE scope=? AND message_id=?",
                (scope, message_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("не удалось получить существующий источник памяти")
            conn.commit()
            return int(row[0])
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def release_memory_sources(source_ids: list[int]) -> int:
    """Делает источники доступными worker после завершения Telegram-хода."""
    ids = [int(source_id) for source_id in source_ids if source_id is not None]
    if not ids:
        return 0
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.executemany(
            "UPDATE memory_sources SET status='pending', updated_at=? "
            "WHERE id=? AND status='waiting'",
            [(time.time(), source_id) for source_id in ids],
        )
        conn.commit()
        return max(cur.rowcount, 0)
    finally:
        conn.close()


def _memory_source_from_row(row: tuple[Any, ...]) -> MemorySourceRecord:
    return MemorySourceRecord(
        id=int(row[0]),
        scope=str(row[1]),
        author_id=str(row[2]),
        author_name=str(row[3]),
        message_id=int(row[4]),
        created_at=float(row[5]),
        text=str(row[6]),
        status=str(row[7]),
        attempts=int(row[8]),
        last_error=row[9],
    )


def list_memory_sources(status: str | None = None) -> list[MemorySourceRecord]:
    """Возвращает сообщения очереди в исходном порядке."""
    query = (
        "SELECT id, scope, author_id, author_name, message_id, created_at, text, "
        "status, attempts, last_error FROM memory_sources"
    )
    params: tuple[Any, ...] = ()
    if status is not None:
        query += " WHERE status=?"
        params = (status,)
    query += " ORDER BY id"
    rows = _db_execute(query, params, fetch=True) or []
    return [_memory_source_from_row(row) for row in rows]


def claim_memory_sources(limit: int) -> list[MemorySourceRecord]:
    """Атомарно забирает FIFO-порцию pending-сообщений на обработку."""
    bounded_limit = max(1, min(int(limit), 100))
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("BEGIN IMMEDIATE")
        candidate_rows = conn.execute(
            "SELECT id, scope, author_id, author_name, message_id, created_at, text, "
            "status, attempts, last_error FROM memory_sources "
            "WHERE status IN ('pending', 'waiting') ORDER BY id LIMIT ?",
            (bounded_limit,),
        ).fetchall()
        rows = []
        for row in candidate_rows:
            if row[7] == "waiting":
                break
            rows.append(row)
        now = time.time()
        for row in rows:
            conn.execute(
                "UPDATE memory_sources SET status='processing', attempts=attempts+1, "
                "updated_at=? WHERE id=? AND status='pending'",
                (now, row[0]),
            )
        conn.commit()
        return [
            replace(
                _memory_source_from_row(row),
                status="processing",
                attempts=int(row[8]) + 1,
            )
            for row in rows
        ]
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def complete_memory_source(source_id: int) -> None:
    """Помечает сообщение обработанным и удаляет полный сырой текст."""
    _db_execute(
        "UPDATE memory_sources SET status='completed', text='', last_error=NULL, updated_at=? "
        "WHERE id=?",
        (time.time(), source_id),
    )


def fail_memory_source(source_id: int, error: str) -> bool:
    """Возвращает сообщение в очередь или переводит в dead-letter.

    Возвращает True, если достигнут терминальный лимит попыток.
    """
    rows = _db_execute(
        "SELECT attempts FROM memory_sources WHERE id=?", (source_id,), fetch=True
    ) or []
    attempts = int(rows[0][0]) if rows else MEMORY_SOURCE_MAX_ATTEMPTS
    terminal = attempts >= MEMORY_SOURCE_MAX_ATTEMPTS
    status = "dead" if terminal else "pending"
    params = (status, str(error)[:500], time.time(), source_id)
    if terminal:
        _db_execute(
            "UPDATE memory_sources SET status=?, last_error=?, updated_at=?, text='' "
            "WHERE id=?",
            params,
        )
    else:
        _db_execute(
            "UPDATE memory_sources SET status=?, last_error=?, updated_at=? WHERE id=?",
            params,
        )
    return terminal


def _memory_fact_from_row(row: tuple[Any, ...]) -> MemoryFactRecord:
    return MemoryFactRecord(
        id=int(row[0]),
        scope=str(row[1]),
        subject_id=str(row[2]),
        fact_key=str(row[3]),
        fact=str(row[4]),
        source_id=int(row[5]),
        source_quote=str(row[6]),
        source_message_id=int(row[7]),
        source_created_at=float(row[8]),
        mem0_id=str(row[9]),
    )


def get_memory_fact(scope: str, subject_id: str, fact_key: str) -> MemoryFactRecord | None:
    rows = _db_execute(
        "SELECT id, scope, subject_id, fact_key, fact, source_id, source_quote, "
        "source_message_id, source_created_at, mem0_id FROM memory_facts "
        "WHERE scope=? AND subject_id=? AND fact_key=?",
        (scope, subject_id, fact_key),
        fetch=True,
    ) or []
    return _memory_fact_from_row(rows[0]) if rows else None


def upsert_memory_fact(
    *,
    scope: str,
    subject_id: str,
    fact_key: str,
    fact: str,
    source_id: int,
    source_quote: str,
    source_message_id: int,
    source_created_at: float,
    mem0_id: str,
) -> MemoryFactRecord | None:
    """Сохраняет provenance одобренного факта и возвращает прежнюю запись."""
    previous = get_memory_fact(scope, subject_id, fact_key)
    commit_memory_facts(
        [
            MemoryFactWrite(
                scope=scope,
                subject_id=subject_id,
                fact_key=fact_key,
                fact=fact,
                source_id=source_id,
                source_quote=source_quote,
                source_message_id=source_message_id,
                source_created_at=source_created_at,
                mem0_id=mem0_id,
            )
        ]
    )
    return previous


def commit_memory_facts(
    writes: list[MemoryFactWrite], *, complete_source_id: int | None = None
) -> None:
    """Атомарно публикует все одобренные факты одного сообщения-источника."""
    if not writes:
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = time.time()
        for write in writes:
            conn.execute(
                "INSERT INTO memory_facts "
                "(scope, subject_id, fact_key, fact, source_id, source_quote, "
                "source_message_id, source_created_at, mem0_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(scope, subject_id, fact_key) DO UPDATE SET "
                "fact=excluded.fact, source_id=excluded.source_id, "
                "source_quote=excluded.source_quote, "
                "source_message_id=excluded.source_message_id, "
                "source_created_at=excluded.source_created_at, "
                "mem0_id=excluded.mem0_id, updated_at=excluded.updated_at",
                (
                    write.scope,
                    write.subject_id,
                    write.fact_key,
                    write.fact,
                    write.source_id,
                    write.source_quote,
                    write.source_message_id,
                    write.source_created_at,
                    write.mem0_id,
                    now,
                    now,
                ),
            )
        if complete_source_id is not None:
            conn.execute(
                "UPDATE memory_sources SET status='completed', text='', "
                "last_error=NULL, updated_at=? WHERE id=?",
                (now, complete_source_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_memory_facts(scope: str | None = None) -> list[MemoryFactRecord]:
    query = (
        "SELECT id, scope, subject_id, fact_key, fact, source_id, source_quote, "
        "source_message_id, source_created_at, mem0_id FROM memory_facts"
    )
    params: tuple[Any, ...] = ()
    if scope is not None:
        query += " WHERE scope=?"
        params = (scope,)
    query += " ORDER BY id"
    rows = _db_execute(query, params, fetch=True) or []
    return [_memory_fact_from_row(row) for row in rows]


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
