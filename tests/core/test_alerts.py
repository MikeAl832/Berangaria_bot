import asyncio

from telegram.error import Conflict

from berangaria.core import alerts


class _Bot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)


def test_alert_defaults_to_owner_and_deduplicates(monkeypatch):
    recorded = []
    bot = _Bot()
    clock = [1000.0]
    monkeypatch.setattr(alerts, "ADMIN_ALERT_CHAT_ID", None)
    monkeypatch.setattr(alerts, "OWNER_USER_ID", 42)
    monkeypatch.setattr(alerts.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        alerts.analytics_store,
        "record_alert",
        lambda **kwargs: recorded.append(kwargs) or True,
    )
    alerts.reset_alert_throttle()

    assert asyncio.run(alerts.notify_owner(bot, category="LLM", message="timeout"))
    assert not asyncio.run(alerts.notify_owner(bot, category="LLM", message="timeout"))
    clock[0] += alerts.ALERT_COOLDOWN_SECONDS
    assert asyncio.run(alerts.notify_owner(bot, category="LLM", message="timeout"))

    assert len(recorded) == 3
    assert [item["chat_id"] for item in bot.messages] == [42, 42]
    assert "Повторов подавлено: 1" in bot.messages[-1]["text"]


def test_explicit_alert_chat_overrides_owner(monkeypatch):
    bot = _Bot()
    monkeypatch.setattr(alerts, "ADMIN_ALERT_CHAT_ID", -100500)
    monkeypatch.setattr(alerts, "OWNER_USER_ID", 42)
    monkeypatch.setattr(alerts.analytics_store, "record_alert", lambda **kwargs: True)
    alerts.reset_alert_throttle()

    assert asyncio.run(alerts.notify_owner(bot, category="DB", message="locked"))

    assert bot.messages[0]["chat_id"] == -100500


def test_volatile_detail_does_not_break_conflict_cooldown(monkeypatch):
    recorded = []
    bot = _Bot()
    monkeypatch.setattr(alerts, "ADMIN_ALERT_CHAT_ID", None)
    monkeypatch.setattr(alerts, "OWNER_USER_ID", 42)
    monkeypatch.setattr(
        alerts.analytics_store,
        "record_alert",
        lambda **kwargs: recorded.append(kwargs) or True,
    )
    alerts.reset_alert_throttle()
    conflict = Conflict("terminated by other getUpdates request")

    assert asyncio.run(
        alerts.notify_owner(
            bot,
            category="Telegram Conflict",
            message="Другой getUpdates держит токен бота.",
            error=conflict,
            detail="host=grok-bot-vm pid=1 uptime=40198s since_update=7076s updates=86",
        )
    )
    assert not asyncio.run(
        alerts.notify_owner(
            bot,
            category="Telegram Conflict",
            message="Другой getUpdates держит токен бота.",
            error=conflict,
            detail="host=grok-bot-vm pid=1 uptime=40202s since_update=7081s updates=86",
        )
    )

    assert len(bot.messages) == 1
    assert "uptime=40198s" in bot.messages[0]["text"]
    assert recorded[0]["fingerprint"] == recorded[1]["fingerprint"]
    assert "uptime=40202s" in recorded[1]["message"]


def test_notify_owner_logs_when_analytics_record_fails(monkeypatch, caplog):
    bot = _Bot()
    monkeypatch.setattr(alerts, "ADMIN_ALERT_CHAT_ID", None)
    monkeypatch.setattr(alerts, "OWNER_USER_ID", 42)
    monkeypatch.setattr(alerts.analytics_store, "record_alert", lambda **kwargs: False)
    alerts.reset_alert_throttle()

    with caplog.at_level("ERROR"):
        assert asyncio.run(alerts.notify_owner(bot, category="LLM", message="timeout"))

    assert "analytics_alerts" in caplog.text
    assert len(bot.messages) == 1

