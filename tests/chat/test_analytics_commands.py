import asyncio
from types import SimpleNamespace

from berangaria.chat import handlers


class _Message:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


def _message_update(*, user_id: int, chat_id: int, chat_type: str):
    return SimpleNamespace(
        message=_Message(),
        effective_user=SimpleNamespace(id=user_id, first_name="User"),
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
    )


def test_top_uses_same_group_allowlist_as_stats(monkeypatch):
    update = _message_update(user_id=999, chat_id=-100, chat_type="supergroup")
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-100])
    monkeypatch.setattr(handlers.analytics_ui, "render_top", lambda chat_id: f"top:{chat_id}")

    asyncio.run(handlers.top(update, SimpleNamespace()))

    assert update.message.replies == [("top:-100", {})]


def test_dashboard_is_owner_only_and_private(monkeypatch):
    monkeypatch.setattr(handlers, "OWNER_USER_ID", 42)
    monkeypatch.setattr(
        handlers.analytics_ui,
        "render_dashboard",
        lambda *args: ("dashboard", "keyboard"),
    )

    stranger = _message_update(user_id=7, chat_id=7, chat_type="private")
    asyncio.run(handlers.dashboard(stranger, SimpleNamespace()))
    assert stranger.message.replies == []

    group_owner = _message_update(user_id=42, chat_id=-100, chat_type="supergroup")
    asyncio.run(handlers.dashboard(group_owner, SimpleNamespace()))
    assert "только в личном чате" in group_owner.message.replies[0][0]

    owner = _message_update(user_id=42, chat_id=42, chat_type="private")
    asyncio.run(handlers.dashboard(owner, SimpleNamespace()))
    assert owner.message.replies == [("dashboard", {"reply_markup": "keyboard"})]


class _Callback:
    def __init__(self, user_id: int):
        self.from_user = SimpleNamespace(id=user_id)
        self.message = SimpleNamespace(chat=SimpleNamespace(type="private"))
        self.data = "dashboard:leaders:30d"
        self.answers = []
        self.edits = []

    async def answer(self, text=None, **kwargs):
        self.answers.append((text, kwargs))

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)


def test_dashboard_callback_rechecks_owner(monkeypatch):
    monkeypatch.setattr(handlers, "OWNER_USER_ID", 42)
    monkeypatch.setattr(
        handlers.analytics_ui,
        "render_dashboard",
        lambda page, period: (f"{page}:{period}", "keyboard"),
    )

    foreign_query = _Callback(7)
    asyncio.run(
        handlers.dashboard_callback(
            SimpleNamespace(callback_query=foreign_query), SimpleNamespace()
        )
    )
    assert foreign_query.edits == []
    assert foreign_query.answers == [("Недоступно", {"show_alert": True})]

    owner_query = _Callback(42)
    asyncio.run(
        handlers.dashboard_callback(
            SimpleNamespace(callback_query=owner_query), SimpleNamespace()
        )
    )
    assert owner_query.edits == [
        {"text": "leaders:30d", "reply_markup": "keyboard"}
    ]

