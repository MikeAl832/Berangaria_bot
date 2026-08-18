import asyncio

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
        lambda **kwargs: recorded.append(kwargs),
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

