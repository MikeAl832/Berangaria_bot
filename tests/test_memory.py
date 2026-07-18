import asyncio
from types import SimpleNamespace

import handlers
from handlers import _build_memory_text
import state


def test_memory_text_keeps_only_user_text_when_media_is_present():
    text = _build_memory_text(
        "Я использую Fedora",
        [("video", "на видео виден компьютер с Ubuntu")],
    )

    assert text == "Я использую Fedora"


def test_media_only_message_has_no_long_term_memory_source():
    text = _build_memory_text(
        "",
        [("image", "на изображении человек с видеокартой RTX 5090")],
    )

    assert text == ""


def test_forwarded_text_has_no_long_term_memory_source():
    text = _build_memory_text(
        "Я живу в Москве",
        [],
        is_forwarded=True,
    )

    assert text == ""


def test_memory_worker_starts_after_buffered_turn_finishes(monkeypatch):
    events = []
    state.message_buffer.clear()
    monkeypatch.setattr(handlers, "MESSAGE_DEBOUNCE_SECONDS", 0)
    monkeypatch.setattr(handlers, "_check_access_permissions", lambda *args: True)
    monkeypatch.setattr(handlers, "is_bot_mentioned", lambda *args: (False, ""))
    def enqueue(*args, **kwargs):
        events.append("enqueue")
        return 17

    monkeypatch.setattr(handlers, "enqueue_memory_source", enqueue)
    monkeypatch.setattr(
        handlers,
        "release_memory_sources",
        lambda source_ids: events.append(("release", source_ids)),
        raising=False,
    )

    async def finish_turn(*args, **kwargs):
        events.append("turn-finished")
        state.message_buffer["7_42"]["messages"].append(
            {"memory_source_id": 18}
        )

    monkeypatch.setattr(handlers, "process_buffered_messages", finish_turn)
    message = SimpleNamespace(
        message_id=901,
        date=None,
        forward_origin=None,
        reply_to_message=None,
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=7, type="private"),
        effective_user=SimpleNamespace(id=42, first_name="Миша"),
        message=message,
    )
    context = SimpleNamespace()

    async def run():
        await handlers.queue_message(update, context, "Я постоянно использую Fedora")
        task = state.message_buffer["7_42"]["task"]
        await task

    asyncio.run(run())
    state.message_buffer.clear()

    assert events == ["enqueue", "turn-finished", ("release", [17])]
