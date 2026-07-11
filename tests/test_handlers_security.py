import asyncio

import handlers


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
