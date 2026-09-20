"""Polling identity helpers for overnight Conflict diagnosis."""

import asyncio
import logging
import time

from berangaria.core import polling_diagnostics as pd


def setup_function(_fn):
    pd.reset_for_tests()


def test_format_context_includes_host_and_pid():
    pd.mark_started(bot_id=8516262902)
    text = pd.format_context()
    assert "bot_id=8516262902" in text
    assert "pid=" in text
    assert "host=" in text
    assert "since_update=never" in text
    assert "since_poll=never" in text
    assert "updates=0" in text
    assert "polls=0" in text


def test_mark_update_received_advances_counter():
    pd.mark_started(bot_id=1)
    pd.mark_update_received()
    pd.mark_update_received()
    snap = pd.snapshot()
    assert snap.updates_seen == 2
    assert snap.last_update_at is not None
    assert "since_update=" in pd.format_context()
    assert "never" not in pd.format_context().split("since_update=")[1].split()[0]


def test_mark_poll_ok_advances_poll_counter(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "POLL_HEARTBEAT_PATH", tmp_path / "poll_heartbeat")
    pd.mark_started(bot_id=1)
    pd.mark_poll_ok(update_count=0)
    pd.mark_poll_ok(update_count=2)
    snap = pd.snapshot()
    assert snap.polls_seen == 2
    assert snap.last_poll_at is not None
    assert (tmp_path / "poll_heartbeat").is_file()


def test_heartbeat_emits_info(monkeypatch, caplog):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    pd.mark_started(bot_id=42)
    pd.mark_poll_ok()
    with caplog.at_level(logging.INFO):
        try:
            asyncio.run(pd.polling_heartbeat_loop(sleep=fake_sleep))
        except asyncio.CancelledError:
            pass

    assert sleeps[0] == pd.HEARTBEAT_INTERVAL_SECONDS
    assert any("Polling heartbeat" in r.message for r in caplog.records)
    assert not any("Polling stalled" in r.message for r in caplog.records)


def test_heartbeat_warns_when_polls_are_stale(monkeypatch, caplog):
    sleeps = []
    clock = [1_000.0]
    monkeypatch.setattr(pd.time, "time", lambda: clock[0])
    exits = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += pd.STALE_POLL_SECONDS + 5
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    pd.mark_started(bot_id=42)
    pd.mark_poll_ok()
    with caplog.at_level(logging.WARNING):
        try:
            asyncio.run(
                pd.polling_heartbeat_loop(sleep=fake_sleep, exit_fn=lambda code: exits.append(code))
            )
        except asyncio.CancelledError:
            pass

    assert any("Polling stalled" in r.message for r in caplog.records)
    assert any("stall dump" in r.message for r in caplog.records)
    assert exits == []  # still under exit budget (+5 over 30m, not 45m)


def test_heartbeat_exits_when_poll_stall_exceeds_budget(monkeypatch, caplog):
    sleeps = []
    clock = [1_000.0]
    monkeypatch.setattr(pd.time, "time", lambda: clock[0])
    exits = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += pd.STALE_POLL_EXIT_SECONDS + 5
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    pd.mark_started(bot_id=42)
    pd.mark_poll_ok()
    with caplog.at_level(logging.CRITICAL):
        try:
            asyncio.run(
                pd.polling_heartbeat_loop(sleep=fake_sleep, exit_fn=lambda code: exits.append(code))
            )
        except asyncio.CancelledError:
            pass

    assert exits == [1]
    assert any("exiting for Docker restart" in r.message for r in caplog.records)


def test_heartbeat_does_not_warn_on_quiet_chat_with_fresh_polls(monkeypatch, caplog):
    """No messages for hours is fine if getUpdates keeps returning."""
    sleeps = []
    clock = [1_000.0]
    monkeypatch.setattr(pd.time, "time", lambda: clock[0])

    async def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += pd.STALE_POLL_SECONDS + 5
        pd.mark_poll_ok()  # long-poll still healthy
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    pd.mark_started(bot_id=42)
    pd.mark_update_received()
    with caplog.at_level(logging.WARNING):
        try:
            asyncio.run(pd.polling_heartbeat_loop(sleep=fake_sleep))
        except asyncio.CancelledError:
            pass

    assert not any("Polling stalled" in r.message for r in caplog.records)


def test_heartbeat_does_not_warn_when_never_polled_yet(monkeypatch, caplog):
    sleeps = []
    clock = [1_000.0]
    monkeypatch.setattr(pd.time, "time", lambda: clock[0])
    exits = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += 60
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    pd.mark_started(bot_id=42)
    with caplog.at_level(logging.WARNING):
        try:
            asyncio.run(
                pd.polling_heartbeat_loop(sleep=fake_sleep, exit_fn=lambda code: exits.append(code))
            )
        except asyncio.CancelledError:
            pass

    assert not any("Polling stalled" in r.message for r in caplog.records)
    assert exits == []


def test_watchdog_does_not_exit_while_heartbeat_is_fresh():
    pd.touch_loop_beat()
    assert not pd.loop_watchdog_should_exit()


def test_watchdog_exits_after_two_missed_heartbeats(monkeypatch):
    pd.touch_loop_beat()
    monkeypatch.setattr(pd, "LOOP_WATCHDOG_SECONDS", 0.05)
    time.sleep(0.08)
    assert pd.loop_watchdog_should_exit()


def test_heartbeat_pets_the_loop_watchdog(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "LOOP_HEARTBEAT_PATH", tmp_path / "loop_heartbeat")
    pd._last_loop_beat = 0.0
    sleeps = []

    async def fake_sleep(_delay):
        sleeps.append(_delay)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    pd.mark_started(bot_id=1)
    pd.mark_poll_ok()
    try:
        asyncio.run(pd.polling_heartbeat_loop(sleep=fake_sleep))
    except asyncio.CancelledError:
        pass

    assert pd._last_loop_beat > 0.0
    assert (tmp_path / "loop_heartbeat").is_file()


def test_install_get_updates_probe_marks_empty_poll():
    class FakeBot:
        async def get_updates(self, *args, **kwargs):
            return ()

    bot = FakeBot()
    pd.install_get_updates_probe(bot)
    asyncio.run(bot.get_updates())
    assert pd.snapshot().polls_seen == 1
    assert pd.snapshot().last_poll_at is not None
