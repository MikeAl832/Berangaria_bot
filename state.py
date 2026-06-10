# Глобальные состояния бота
import asyncio
import time
from typing import Dict, Any, List

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
        _history_locks.pop(key, None)  # Очищаем блокировки
    
    return len(keys_to_remove)
