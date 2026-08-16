"""Bridge vision helpers must call keyword-only media APIs correctly."""

import asyncio
from types import SimpleNamespace

from berangaria.user_bridge import media


def _client(payload: bytes = b"fake-bytes"):
    class _Client:
        async def download_media(self, msg, file=None):
            return payload

    return _Client()


def test_bridge_video_uses_keyword_only_describe_video(monkeypatch):
    monkeypatch.setattr(media, "VISION_MODE", True)

    called = {}

    async def fake_describe_video(*, video_path, mime="video/mp4", caption="", duration=0.0):
        called["kwargs"] = {
            "video_path": video_path,
            "mime": mime,
            "caption": caption,
            "duration": duration,
        }
        return "a video of a cat"

    monkeypatch.setattr(media, "describe_video", fake_describe_video)

    msg = SimpleNamespace(
        photo=None,
        video=SimpleNamespace(duration=3),
        video_note=None,
        gif=None,
        animation=None,
        document=None,
        voice=None,
        audio=None,
    )

    desc, kind = asyncio.run(
        media.describe_bridge_media(_client(), msg, caption="look")
    )

    assert kind == "video"
    assert desc == "a video of a cat"
    assert called["kwargs"]["caption"] == "look"
    assert called["kwargs"]["duration"] == 3.0
    assert called["kwargs"]["video_path"]


def test_bridge_audio_uses_keyword_only_transcribe_audio(monkeypatch):
    monkeypatch.setattr(media, "VISION_MODE", True)

    called = {}

    async def fake_transcribe_audio(*, audio_path, mime="audio/ogg", caption=""):
        called["kwargs"] = {
            "audio_path": audio_path,
            "mime": mime,
            "caption": caption,
        }
        return "hello from a bot"

    monkeypatch.setattr(media, "transcribe_audio", fake_transcribe_audio)

    msg = SimpleNamespace(
        photo=None,
        video=None,
        video_note=None,
        gif=None,
        animation=None,
        document=None,
        voice=SimpleNamespace(duration=2),
        audio=None,
    )

    desc, kind = asyncio.run(
        media.describe_bridge_media(_client(), msg, caption="voice")
    )

    assert kind == "audio"
    assert desc == "hello from a bot"
    assert called["kwargs"]["caption"] == "voice"
    assert called["kwargs"]["audio_path"]
    assert "duration" not in called["kwargs"]
