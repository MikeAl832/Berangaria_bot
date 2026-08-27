import asyncio
from types import SimpleNamespace

from berangaria.chat import handlers
from berangaria.core import state


class _ForbiddenMediaMessage:
    caption = ""

    def __init__(self):
        self.replies = []

    @property
    def photo(self):
        raise AssertionError("photo must not be accessed before authorization")

    @property
    def sticker(self):
        raise AssertionError("sticker must not be accessed before authorization")

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class _User:
    id = 2


class _Chat:
    id = 100
    type = "private"


class _Update:
    def __init__(self):
        self.message = _ForbiddenMediaMessage()
        self.effective_user = _User()
        self.effective_chat = _Chat()


class _Bot:
    id = 999


class _Context:
    bot = _Bot()


def test_log_message_preview_uses_configured_limit(monkeypatch):
    monkeypatch.setattr(handlers, "LOG_MESSAGE_PREVIEW_CHARS", 400)
    text = "я" * 500

    preview = handlers._log_message_preview(text)

    assert len(preview) == 400
    assert preview.endswith("...")
    assert handlers._log_message_preview("коротко") == "коротко"


def test_photo_access_checked_before_media_processing(monkeypatch):
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    update = _Update()

    asyncio.run(handlers.handle_media(update, _Context()))

    assert update.message.replies == ["Не разговариваю с незнакомцами."]


def test_sticker_access_checked_before_media_processing(monkeypatch):
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    update = _Update()

    asyncio.run(handlers.handle_sticker(update, _Context()))

    assert update.message.replies == ["Не разговариваю с незнакомцами."]


def test_owner_bypasses_user_and_group_allowlists(monkeypatch):
    monkeypatch.setattr(handlers, "OWNER_USER_ID", 2)
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-999])

    assert handlers._check_access_permissions(100, 2, False)
    assert handlers._check_access_permissions(-100, 2, True)
    assert not handlers._check_access_permissions(100, 3, False)
    assert not handlers._check_access_permissions(-100, 3, True)


def test_group_media_invalidates_older_ambient_candidate(monkeypatch):
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-100])
    stale_token = state.record_group_activity(-100)
    update = SimpleNamespace(
        message=SimpleNamespace(),
        effective_chat=SimpleNamespace(id=-100, type="supergroup"),
        effective_user=SimpleNamespace(id=2),
    )
    context = SimpleNamespace(bot=SimpleNamespace(id=999))

    handlers._record_incoming_media_activity(update, context)

    assert not state.is_latest_group_activity(-100, stale_token)
