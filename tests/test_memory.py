import asyncio
import logging

import llm_client
import memory_store
import state
from llm_client import _add_memory_chunks, _chunk_lines_by_chars
from handlers import _build_memory_text


def test_chunks_respect_budget():
    lines = [f"Аня: {'a' * 50}" for _ in range(10)]
    chunks = _chunk_lines_by_chars(lines, 200)
    for c in chunks:
        # либо влезли в бюджет, либо это одиночная строка (не разбиваемая дальше)
        assert sum(len(item) for item in c) <= 200 or len(c) == 1


def test_oversized_single_line_is_split():
    big = ["x" * 500]
    chunks = _chunk_lines_by_chars(big, 200)
    assert len(chunks) == 3
    assert all(len(c[0]) <= 200 for c in chunks)


def test_empty_input():
    assert _chunk_lines_by_chars([], 200) == []


def test_all_lines_preserved():
    lines = [f"line{i}" for i in range(7)]
    chunks = _chunk_lines_by_chars(lines, 20)
    flat = [item for c in chunks for item in c]
    assert flat == lines  # ничего не потеряли и не переставили


def test_memory_text_includes_bounded_media_description():
    text = _build_memory_text("", [("image", "очень важное описание " * 100)])

    assert text.startswith("Изображение:")
    assert len(text) <= 600


def test_memory_text_keeps_user_text_with_media():
    text = _build_memory_text("смотри", [("video", "кот прыгает по столу")])

    assert "смотри" in text
    assert "Видео:" in text


def test_failed_memory_flush_requeues_batch(monkeypatch):
    class FailingMemory:
        def add(self, *args, **kwargs):
            raise RuntimeError("temporary outage")

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_store, "memory", FailingMemory())
    monkeypatch.setattr(llm_client.asyncio, "sleep", no_sleep)
    state.pending_memory.clear()

    asyncio.run(_add_memory_chunks("private_42", ["важная реплика"]))

    assert state.pending_memory["private_42"] == ["важная реплика"]


def test_memory_is_buffered_while_mem0_is_unavailable(monkeypatch):
    monkeypatch.setattr(memory_store, "memory", None)
    state.pending_memory.clear()

    llm_client.record_user_memory(
        "private_42",
        "сегодня купил новую видеокарту и хочу запомнить модель",
        "Миша",
        False,
    )

    assert state.pending_memory["private_42"] == [
        "сегодня купил новую видеокарту и хочу запомнить модель"
    ]


def test_message_shorter_than_mem0_minimum_is_not_buffered(monkeypatch):
    monkeypatch.setattr(memory_store, "memory", None)
    state.pending_memory.clear()

    llm_client.record_user_memory(
        "private_42",
        "x" * 17,
        "Миша",
        False,
    )

    assert "private_42" not in state.pending_memory


def test_repeated_low_signal_fillers_are_not_buffered(monkeypatch):
    monkeypatch.setattr(memory_store, "memory", None)
    state.pending_memory.clear()

    llm_client.record_user_memory(
        "private_42",
        "ну типа как бы ну типа",
        "Миша",
        False,
    )

    assert "private_42" not in state.pending_memory


def test_discarded_mem0_fact_is_deleted_and_logged(monkeypatch, caplog):
    class Memory:
        def __init__(self):
            self.deleted = []

        def add(self, *args, **kwargs):
            return {
                "results": [
                    {
                        "id": "fact-1",
                        "memory": "Миша обычно отвечает: ну типа как бы",
                        "event": "ADD",
                    },
                    {
                        "id": "fact-2",
                        "memory": "Миша написал общее подтверждение",
                        "event": "ADD",
                    },
                ]
            }

        def delete(self, memory_id):
            self.deleted.append(memory_id)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "DISCARD: общая реакция, малоинформативно",
                    }
                }],
                "usage": {
                    "prompt_tokens": 87,
                    "completion_tokens": 12,
                    "total_tokens": 99,
                },
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return Response()

    memory = Memory()
    monkeypatch.setattr(memory_store, "memory", memory)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)
    state.pending_memory.clear()
    state.pending_memory["private_42"] = ["обсуждение достаточно длинное для Mem0"]
    caplog.set_level(logging.INFO, logger="llm_client")

    asyncio.run(llm_client.flush_pending_memory())

    assert memory.deleted == ["fact-1", "fact-2"]
    assert caplog.text.count("🧠 Mem0 модерация: запрос=87, ответ=12, всего=99") == 2
    assert caplog.text.count("❌ Память: факт отклонён (общая реакция, малоинформативно)") == 2


def test_mem0_fact_is_kept_when_moderation_fails(monkeypatch, caplog):
    class Memory:
        def __init__(self):
            self.deleted = []

        def add(self, *args, **kwargs):
            return {
                "results": [{
                    "id": "fact-1",
                    "memory": "Миша использует видеокарту RTX 5070 Ti",
                    "event": "ADD",
                }]
            }

        def delete(self, memory_id):
            self.deleted.append(memory_id)

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise RuntimeError("DeepSeek unavailable")

    memory = Memory()
    monkeypatch.setattr(memory_store, "memory", memory)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", FailingClient)
    state.pending_memory.clear()
    state.pending_memory["private_42"] = ["Моя видеокарта — RTX 5070 Ti"]
    caplog.set_level(logging.WARNING, logger="llm_client")

    asyncio.run(llm_client.flush_pending_memory())

    assert memory.deleted == []
    assert "Mem0 модерация не сработала, факт сохранён" in caplog.text


def test_final_memory_flush_requeues_failed_batch(monkeypatch):
    class FailingMemory:
        def add(self, *args, **kwargs):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(memory_store, "memory", FailingMemory())
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)
    state.pending_memory.clear()
    state.pending_memory["private_42"] = ["важная реплика для повторной попытки"]

    llm_client.flush_pending_memory_blocking()

    assert state.pending_memory["private_42"] == ["важная реплика для повторной попытки"]
