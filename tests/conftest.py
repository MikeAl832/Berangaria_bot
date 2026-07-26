import os
import sys
from pathlib import Path

import pytest

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Подставляем фиктивные ключи до импорта config (он падает без них)
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("API_KEY", "test-deepseek-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")


# Глобалы уровня модуля в core/state.py. Тест, оставивший здесь мусор, меняет
# поведение следующих: история «просачивается» в чужой чат, а закешированный
# asyncio.Lock оказывается привязан к уже закрытому event loop.
_STATE_DICTS = (
    "histories",
    "chat_tokens",
    "api_call_count",
    "message_buffer",
    "last_activity",
    "media_description_cache",
    "random_reply_cooldown",
    "_history_locks",
    "_turn_locks",
)


@pytest.fixture(autouse=True)
def isolate_module_state():
    """Возвращает разделяемое состояние в исходный вид после каждого теста.

    Ключевой момент — `_turn_locks` и `_history_locks`: asyncio.Lock в 3.11
    привязывается к event loop при первой конкуренции, поэтому переживший тест
    лок ломает следующий `asyncio.run` с тем же ключом чата.
    """
    from berangaria.core import state

    yield

    for name in _STATE_DICTS:
        container = getattr(state, name, None)
        if container is not None:
            container.clear()

    memory_pipeline = sys.modules.get("berangaria.memory.pipeline")
    if memory_pipeline is not None:
        task = getattr(memory_pipeline, "_worker_task", None)
        if task is not None and not task.done():
            task.cancel()
        memory_pipeline._worker_task = None


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Отдельная SQLite-база на тест, инициализированная схемой."""
    from berangaria.core import state

    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    return state.DB_PATH
