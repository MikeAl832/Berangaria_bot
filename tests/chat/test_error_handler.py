"""Transient Telegram transport errors must not alert the owner."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from telegram.error import NetworkError

from berangaria.chat import handlers
from berangaria.core import polling_diagnostics


def test_network_error_is_logged_without_owner_alert(monkeypatch, caplog):
    alerts = []

    async def fake_notify_owner(bot, *, category, message, error=None):
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

    async def fake_notify_owner(bot, *, category, message, error=None):
        alerts.append((category, message))

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)

    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))
    context = SimpleNamespace(bot=object(), error=RuntimeError("boom"))

    asyncio.run(handlers.error_handler(update, context))

    assert alerts == [("Unhandled error", "boom")]
    update.effective_message.reply_text.assert_awaited_once()


def test_raw_httpx_connect_error_is_transient(monkeypatch):
    alerts = []

    async def fake_notify_owner(bot, *, category, message, error=None):
        alerts.append(category)

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)
    context = SimpleNamespace(bot=object(), error=httpx.ConnectError(""))
    update = SimpleNamespace(effective_message=None)

    asyncio.run(handlers.error_handler(update, context))
    assert alerts == []


def test_conflict_notifies_and_exits(monkeypatch):
    polling_diagnostics.reset_for_tests()
    polling_diagnostics.mark_started(bot_id=8516262902)
    alerts = []
    exits = []

    messages = []

    async def fake_notify_owner(bot, *, category, message, error=None):
        alerts.append(category)
        messages.append(message)

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(handlers.alerts, "notify_owner", fake_notify_owner)
    monkeypatch.setattr(handlers.asyncio, "sleep", fake_sleep)
    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(handlers.os, "_exit", fake_exit)

    from telegram.error import Conflict

    context = SimpleNamespace(bot=object(), error=Conflict("terminated by other getUpdates request"))
    update = SimpleNamespace(effective_message=None)
    try:
        asyncio.run(handlers.error_handler(update, context))
    except SystemExit as exc:
        assert exc.code == 1

    assert alerts == ["Telegram Conflict"]
    assert exits == [1]
    assert "bot_id=8516262902" in messages[0]
    assert "pid=" in messages[0]
