import asyncio
import time

from berangaria import app as main
from berangaria.chat.summarization import scheduled_summary_status
from berangaria.core import state


def test_scheduled_summary_skips_short_history():
    assert scheduled_summary_status(
        19,
        last_activity=0.0,
        now=1_000.0,
        interval=10,
        min_extra=10,
        quiet_seconds=600,
    ) == "too_short"


def test_scheduled_summary_ready_when_enough_and_quiet():
    assert scheduled_summary_status(
        20,
        last_activity=100.0,
        now=800.0,
        interval=10,
        min_extra=10,
        quiet_seconds=600,
    ) == "ready"


def test_scheduled_summary_postpones_recent_activity():
    assert scheduled_summary_status(
        25,
        last_activity=700.0,
        now=800.0,
        interval=10,
        min_extra=10,
        quiet_seconds=600,
    ) == "recent"


def test_scheduled_summary_missing_activity_is_ready():
    assert scheduled_summary_status(
        20,
        last_activity=None,
        now=800.0,
        interval=10,
        min_extra=10,
        quiet_seconds=600,
    ) == "ready"


def test_scheduled_summary_quiet_zero_never_postpones():
    assert scheduled_summary_status(
        20,
        last_activity=799.0,
        now=800.0,
        interval=10,
        min_extra=10,
        quiet_seconds=0,
    ) == "ready"


def _history(n):
    return [
        {"role": "user", "content": f"[Message: m{i}]", "sid": i + 1, "mid": i + 10}
        for i in range(n)
    ]


def _patch_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "SUMMARY_INTERVAL", 10)
    monkeypatch.setattr(main, "SUMMARY_MIN_EXTRA", 10)
    monkeypatch.setattr(main, "SUMMARY_QUIET_SECONDS", 600)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    state.histories.clear()
    state.last_activity.clear()


def test_scheduled_slot_skips_short_chat(monkeypatch, tmp_path):
    called = []

    async def fake_summarize(history, *, key=None):
        called.append(key)
        return history

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("berangaria.chat.llm_client.summarize_history", fake_summarize)
    _patch_schedule(monkeypatch, tmp_path)
    key = "group_-1"
    state.histories[key] = _history(12)
    state.last_activity[key] = time.time() - 3_600

    count = asyncio.run(main.run_scheduled_summarization_slot(sleep=fake_sleep))

    assert count == 0
    assert called == []
    assert sleeps == []


def test_scheduled_slot_postpones_once_then_skips_if_still_active(
    monkeypatch, tmp_path
):
    called = []

    async def fake_summarize(history, *, key=None):
        called.append(key)
        return history

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("berangaria.chat.llm_client.summarize_history", fake_summarize)
    _patch_schedule(monkeypatch, tmp_path)
    key = "group_-1"
    state.histories[key] = _history(25)
    state.last_activity[key] = time.time()

    count = asyncio.run(main.run_scheduled_summarization_slot(sleep=fake_sleep))

    assert count == 0
    assert called == []
    assert sleeps == [600]


def test_scheduled_slot_summarizes_after_quiet_postpone(monkeypatch, tmp_path):
    called = []

    async def fake_summarize(history, *, key=None):
        called.append(key)
        compressed = [{"role": "user", "content": "[Previous conversation summary: x]"}]
        compressed.extend(history[-10:])
        return compressed

    async def fake_sleep(seconds):
        state.last_activity["group_-1"] = time.time() - 3_600

    monkeypatch.setattr("berangaria.chat.llm_client.summarize_history", fake_summarize)
    _patch_schedule(monkeypatch, tmp_path)
    key = "group_-1"
    state.histories[key] = _history(25)
    state.last_activity[key] = time.time()

    count = asyncio.run(main.run_scheduled_summarization_slot(sleep=fake_sleep))

    assert count == 1
    assert called == [key]
    assert len(state.histories[key]) == 11


def test_scheduled_slot_summarizes_quiet_chat_without_sleep(monkeypatch, tmp_path):
    called = []

    async def fake_summarize(history, *, key=None):
        called.append(key)
        compressed = [{"role": "user", "content": "[Previous conversation summary: x]"}]
        compressed.extend(history[-10:])
        return compressed

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("berangaria.chat.llm_client.summarize_history", fake_summarize)
    _patch_schedule(monkeypatch, tmp_path)
    key = "group_-1"
    state.histories[key] = _history(25)
    state.last_activity[key] = time.time() - 3_600

    count = asyncio.run(main.run_scheduled_summarization_slot(sleep=fake_sleep))

    assert count == 1
    assert called == [key]
    assert sleeps == []
    assert len(state.histories[key]) == 11
