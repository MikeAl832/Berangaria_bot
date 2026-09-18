"""Polling identity helpers for overnight Conflict diagnosis."""

import asyncio
import logging

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
    assert "updates=0" in text


def test_mark_update_received_advances_counter():
    pd.mark_started(bot_id=1)
    pd.mark_update_received()
    pd.mark_update_received()
    snap = pd.snapshot()
    assert snap.updates_seen == 2
    assert snap.last_update_at is not None
    assert "since_update=" in pd.format_context()
    assert "never" not in pd.format_context().split("since_update=")[1].split()[0]


def test_heartbeat_emits_info(monkeypatch, caplog):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError()

    pd.mark_started(bot_id=42)
    with caplog.at_level(logging.INFO):
        try:
            asyncio.run(pd.polling_heartbeat_loop(sleep=fake_sleep))
        except asyncio.CancelledError:
            pass

    assert sleeps[0] == pd.HEARTBEAT_INTERVAL_SECONDS
    assert any("Polling heartbeat" in r.message for r in caplog.records)
