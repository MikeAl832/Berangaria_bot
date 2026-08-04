"""Album gather + single multi-image vision call for Telegram media groups."""

import asyncio

from berangaria.chat import handlers
from berangaria.core import state
from berangaria.media import vision


class _PhotoSize:
    def __init__(self, file_id, file_unique_id):
        self.file_id = file_id
        self.file_unique_id = file_unique_id


class _Message:
    def __init__(self, *, media_group_id, photos, caption="", message_id=1):
        self.media_group_id = media_group_id
        self.photo = photos
        self.caption = caption
        self.message_id = message_id
        self.date = None
        self.reply_to_message = None
        self.forward_origin = None
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class _User:
    id = 1
    first_name = "Tester"


class _Chat:
    id = 42
    type = "private"


class _Update:
    def __init__(self, message):
        self.message = message
        self.effective_user = _User()
        self.effective_chat = _Chat()


class _Bot:
    id = 999


class _Context:
    bot = _Bot()


def _clear_album_state():
    handlers._album_buffer.clear()
    state.media_description_cache.clear()


def test_album_photos_one_gemini_call(monkeypatch):
    """Several media_group photos → one describe_images, one queue_message."""
    _clear_album_state()
    monkeypatch.setattr(handlers, "VISION_MODE", True)
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    monkeypatch.setattr(handlers, "ALBUM_GATHER_SECONDS", 0.05)

    describe_calls = []
    queued = []

    async def fake_download(file_id, context, return_bytes=False):
        return (f"bytes-{file_id}".encode(), "image/jpeg")

    async def fake_describe(images, caption=""):
        describe_calls.append((list(images), caption))
        return "общее описание альбома"

    async def fake_queue(update, context, text, media_description=None, media_kind=None):
        queued.append({
            "text": text,
            "media_description": media_description,
            "media_kind": media_kind,
            "message_id": update.message.message_id,
        })

    monkeypatch.setattr(handlers, "download_media_as_base64", fake_download)
    monkeypatch.setattr(handlers, "describe_images", fake_describe)
    monkeypatch.setattr(handlers, "queue_message", fake_queue)

    photos_a = [_PhotoSize("f1", "u1")]
    photos_b = [_PhotoSize("f2", "u2")]
    photos_c = [_PhotoSize("f3", "u3")]

    u1 = _Update(_Message(media_group_id="mg99", photos=photos_a, caption="", message_id=10))
    u2 = _Update(_Message(media_group_id="mg99", photos=photos_b, caption="кот и пёс", message_id=11))
    u3 = _Update(_Message(media_group_id="mg99", photos=photos_c, caption="", message_id=12))
    ctx = _Context()

    async def run():
        await handlers.handle_media(u1, ctx)
        await handlers.handle_media(u2, ctx)
        await handlers.handle_media(u3, ctx)
        # Wait for album gather timer + processing.
        await asyncio.sleep(0.2)

    asyncio.run(run())

    assert len(describe_calls) == 1
    images, caption = describe_calls[0]
    assert len(images) == 3
    assert caption == "кот и пёс"
    assert len(queued) == 1
    assert queued[0]["media_description"] == "общее описание альбома"
    assert queued[0]["text"] == "кот и пёс"
    assert queued[0]["media_kind"] == "image"
    # Caption-bearing update is preferred for queueing.
    assert queued[0]["message_id"] == 11

    # Cache key covers the whole album.
    cache_key = handlers._album_cache_key(["u1", "u2", "u3"])
    assert state.get_cached_media_description(cache_key) == "общее описание альбома"


def test_single_photo_still_uses_describe_image_bytes(monkeypatch):
    _clear_album_state()
    monkeypatch.setattr(handlers, "VISION_MODE", True)
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])

    describe_images_calls = []
    single_calls = []
    queued = []

    async def fake_download(file_id, context, return_bytes=False):
        return (b"solo", "image/jpeg")

    async def fake_single(image_bytes, mime, caption=""):
        single_calls.append((image_bytes, mime, caption))
        return "одно фото"

    async def fake_multi(images, caption=""):
        describe_images_calls.append(images)
        return "не должно"

    async def fake_queue(update, context, text, media_description=None, media_kind=None):
        queued.append(media_description)

    monkeypatch.setattr(handlers, "download_media_as_base64", fake_download)
    monkeypatch.setattr(handlers, "describe_image_bytes", fake_single)
    monkeypatch.setattr(handlers, "describe_images", fake_multi)
    monkeypatch.setattr(handlers, "queue_message", fake_queue)

    msg = _Message(media_group_id=None, photos=[_PhotoSize("fx", "ux")], caption="hi")
    asyncio.run(handlers.handle_media(_Update(msg), _Context()))

    assert len(single_calls) == 1
    assert describe_images_calls == []
    assert queued == ["одно фото"]


def test_album_cache_hit_skips_vision(monkeypatch):
    _clear_album_state()
    monkeypatch.setattr(handlers, "VISION_MODE", True)
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    monkeypatch.setattr(handlers, "ALBUM_GATHER_SECONDS", 0.05)

    cache_key = handlers._album_cache_key(["a", "b"])
    state.cache_media_description(cache_key, "из кэша")

    describe_calls = []
    downloads = []
    queued = []

    async def fake_download(*args, **kwargs):
        downloads.append(1)
        return (b"x", "image/jpeg")

    async def fake_describe(*args, **kwargs):
        describe_calls.append(1)
        return "свежее"

    async def fake_queue(update, context, text, media_description=None, media_kind=None):
        queued.append(media_description)

    monkeypatch.setattr(handlers, "download_media_as_base64", fake_download)
    monkeypatch.setattr(handlers, "describe_images", fake_describe)
    monkeypatch.setattr(handlers, "queue_message", fake_queue)

    u1 = _Update(_Message(media_group_id="mg1", photos=[_PhotoSize("f1", "a")]))
    u2 = _Update(_Message(media_group_id="mg1", photos=[_PhotoSize("f2", "b")]))

    async def run():
        await handlers.handle_media(u1, _Context())
        await handlers.handle_media(u2, _Context())
        await asyncio.sleep(0.2)

    asyncio.run(run())

    assert describe_calls == []
    assert downloads == []
    assert queued == ["из кэша"]


def test_album_policy_block_passed_to_queue(monkeypatch):
    _clear_album_state()
    monkeypatch.setattr(handlers, "VISION_MODE", True)
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    monkeypatch.setattr(handlers, "ALBUM_GATHER_SECONDS", 0.05)

    queued = []

    async def fake_download(file_id, context, return_bytes=False):
        return (b"x", "image/jpeg")

    async def fake_describe(images, caption=""):
        return vision.POLICY_BLOCKED_IMAGE

    async def fake_queue(update, context, text, media_description=None, media_kind=None):
        queued.append(media_description)

    monkeypatch.setattr(handlers, "download_media_as_base64", fake_download)
    monkeypatch.setattr(handlers, "describe_images", fake_describe)
    monkeypatch.setattr(handlers, "queue_message", fake_queue)

    u1 = _Update(_Message(media_group_id="mg2", photos=[_PhotoSize("f1", "p1")]))
    u2 = _Update(_Message(media_group_id="mg2", photos=[_PhotoSize("f2", "p2")]))

    async def run():
        await handlers.handle_media(u1, _Context())
        await handlers.handle_media(u2, _Context())
        await asyncio.sleep(0.2)

    asyncio.run(run())

    assert queued == [vision.POLICY_BLOCKED_IMAGE]
    # Policy placeholder is cacheable so we do not re-hit Gemini for the same album.
    assert state.get_cached_media_description(handlers._album_cache_key(["p1", "p2"]))
