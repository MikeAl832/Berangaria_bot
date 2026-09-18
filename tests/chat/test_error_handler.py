"""Transient Telegram transport errors must not alert the owner."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from telegram.error import Conflict, NetworkError

from berangaria.chat import handlers
from berangaria.core import alerts as alerts_mod
from berangaria.core import polling_diagnostics


class _Bot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)


def test_network_error_is_logged_without_owner_alert(monkeypatch, caplog):
    alerts = []

    async def fake_notify_owner(bot, *, category, message, error=None, **kwargs):
        alerts.append(category)

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)

    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))
    context = SimpleNamespace(
        bot=object(),
        error=NetworkError("httpx.ConnectError: "),
    )

    with caplog.at_level("WARNING"):
        asyncio.run(handlers.error_handler(update, context))

    assert alerts == []
    update.effective_message.reply_text.assert_not_awaited()
    assert "Сетевой сбой" in caplog.text


def test_unexpected_error_still_alerts_owner(monkeypatch):
    alerts = []

    async def fake_notify_owner(bot, *, category, message, error=None, **kwargs):
        alerts.append((category, message))

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)

    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))
    context = SimpleNamespace(bot=object(), error=RuntimeError("boom"))

    asyncio.run(handlers.error_handler(update, context))

    assert alerts == [("Unhandled error", "boom")]
    update.effective_message.reply_text.assert_awaited_once()


def test_raw_httpx_connect_error_is_transient(monkeypatch):
    alerts = []

    async def fake_notify_owner(bot, *, category, message, error=None, **kwargs):
        alerts.append(category)

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)
    context = SimpleNamespace(bot=object(), error=httpx.ConnectError(""))
    update = SimpleNamespace(effective_message=None)

    asyncio.run(handlers.error_handler(update, context))
    assert alerts == []


def _patch_conflict_exit(monkeypatch, exits, *, raise_system_exit=True):
    async def fake_sleep(_delay):
        return None

    def fake_exit(code):
        exits.append(code)
        if raise_system_exit:
            raise SystemExit(code)

    monkeypatch.setattr(handlers.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(handlers.os, "_exit", fake_exit)


def test_conflict_notifies_and_exits(monkeypatch, caplog):
    polling_diagnostics.reset_for_tests()
    polling_diagnostics.mark_started(bot_id=8516262902)
    alerts = []
    exits = []
    messages = []
    details = []

    async def fake_notify_owner(bot, *, category, message, error=None, detail=None, **kwargs):
        alerts.append(category)
        messages.append(message)
        details.append(detail)

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)
    _patch_conflict_exit(monkeypatch, exits)

    context = SimpleNamespace(
        bot=object(),
        error=Conflict("terminated by other getUpdates request"),
    )
    update = None
    with caplog.at_level("CRITICAL"):
        try:
            asyncio.run(handlers.error_handler(update, context))
        except SystemExit as exc:
            assert exc.code == 1

    assert alerts == ["Telegram Conflict"]
    assert exits == [1]
    assert "uptime=" not in messages[0]
    assert "Docker должен поднять polling заново" in messages[0]
    assert "bot_id=8516262902" in details[0]
    assert "pid=" in details[0]
    assert "выходим для рестарта" in caplog.text
    assert "host=" in caplog.text


def test_double_conflict_within_cooldown_sends_one_owner_alert(monkeypatch):
    polling_diagnostics.reset_for_tests()
    polling_diagnostics.mark_started(bot_id=8516262902)
    alerts_mod.reset_alert_throttle()
    recorded = []
    exits = []
    bot = _Bot()
    clock = [1_000.0]

    monkeypatch.setattr(alerts_mod, "ADMIN_ALERT_CHAT_ID", None)
    monkeypatch.setattr(alerts_mod, "OWNER_USER_ID", 42)
    monkeypatch.setattr(alerts_mod.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        alerts_mod.analytics_store,
        "record_alert",
        lambda **kwargs: recorded.append(kwargs) or True,
    )
    _patch_conflict_exit(monkeypatch, exits, raise_system_exit=False)

    context = SimpleNamespace(
        bot=bot,
        error=Conflict("terminated by other getUpdates request"),
    )
    asyncio.run(handlers.error_handler(None, context))
    clock[0] += 4
    asyncio.run(handlers.error_handler(None, context))

    assert exits == [1, 1]
    assert len(bot.messages) == 1
    assert "Telegram Conflict" in bot.messages[0]["text"]
    assert recorded[0]["fingerprint"] == recorded[1]["fingerprint"]
    assert len(recorded) == 2


def test_conflict_exits_even_if_handler_task_is_cancelled(monkeypatch):
    polling_diagnostics.reset_for_tests()
    polling_diagnostics.mark_started(bot_id=1)
    exits = []
    alerts = []
    started = asyncio.Event()

    async def fake_notify_owner(bot, *, category, message, error=None, **kwargs):
        alerts.append(category)
        started.set()
        await asyncio.Event().wait()

    async def fake_sleep(_delay):
        return None

    def fake_exit(code):
        exits.append(code)

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)
    monkeypatch.setattr(handlers.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(handlers.os, "_exit", fake_exit)

    context = SimpleNamespace(
        bot=object(),
        error=Conflict("terminated by other getUpdates request"),
    )

    async def run():
        task = asyncio.create_task(handlers.error_handler(None, context))
        await started.wait()
        task.cancel()
        await task

    asyncio.run(run())
    assert alerts == ["Telegram Conflict"]
    assert exits == [1]
