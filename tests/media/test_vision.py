"""Unit tests for Gemini vision helpers: multi-image and safety blocks."""

import asyncio
import logging

from berangaria.media import vision


def test_gemini_extract_text_prompt_block():
    text, blocked = vision._gemini_extract_text({
        "promptFeedback": {"blockReason": "SAFETY"},
        "candidates": [],
    })
    assert text == ""
    assert blocked is True


def test_gemini_extract_text_finish_reason_safety():
    text, blocked = vision._gemini_extract_text({
        "candidates": [{
            "finishReason": "SAFETY",
            "content": {"parts": []},
        }],
    })
    assert text == ""
    assert blocked is True


def test_gemini_extract_text_ok():
    text, blocked = vision._gemini_extract_text({
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"text": "Кот на столе"}]},
        }],
    })
    assert text == "Кот на столе"
    assert blocked is False


def test_gemini_extract_text_empty_candidates_not_blocked():
    text, blocked = vision._gemini_extract_text({"candidates": []})
    assert text == ""
    assert blocked is False


def test_image_prompt_album_mentions_count():
    single = vision._image_prompt(1)
    album = vision._image_prompt(3)
    assert "альбом" not in single.lower()
    assert "3" in album
    assert "альбом" in album.lower()


def test_audio_prompt_hints_configured_spoken_names(monkeypatch):
    monkeypatch.setattr(vision, "BOT_NAMES", ["Бер", "Ber"])

    prompt = vision._audio_prompt()

    assert "«Бер»" in prompt
    assert "«Ber»" in prompt
    assert "только если оно действительно произнесено" in prompt


def test_gemini_audio_400_retries_then_returns_empty_and_removes_temp_file(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(vision, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(vision, "GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(vision, "GEMINI_GENERATE_MAX_ATTEMPTS", 3)
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"not-real-audio")
    posts = {"n": 0}

    class _Resp:
        status_code = 400
        text = "invalid audio payload"

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            posts["n"] += 1
            return _Resp()

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(vision.httpx, "AsyncClient", lambda **kwargs: _Client())
    monkeypatch.setattr(vision.asyncio, "sleep", _no_sleep)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(vision.transcribe_audio(
            audio_path=str(audio_path),
            mime="audio/ogg",
        ))

    assert result == ""
    assert posts["n"] == 3
    assert not audio_path.exists()
    assert "Gemini audio API 400" in caplog.text


def test_gemini_generate_content_retries_400_then_succeeds(monkeypatch):
    posts = {"n": 0}

    class _Resp:
        def __init__(self, status_code, text="ok", payload=None):
            self.status_code = status_code
            self.text = text
            self._payload = payload or {}

        def json(self):
            return self._payload

    class _Client:
        async def post(self, url, json=None, headers=None):
            posts["n"] += 1
            if posts["n"] < 3:
                return _Resp(400, "invalid argument")
            return _Resp(200, "ok", {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(vision.asyncio, "sleep", _no_sleep)
    response = asyncio.run(
        vision._gemini_generate_content(
            _Client(),
            "https://example.test",
            {},
            {},
            label="video",
        )
    )
    assert posts["n"] == 3
    assert response.status_code == 200


def test_describe_images_single_call_with_multiple_parts(monkeypatch):
    """N images must produce one generateContent request with N inline_data parts."""
    monkeypatch.setattr(vision, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(vision, "GEMINI_MODEL", "gemini-test")

    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "два кадра: кот и собака"}]},
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    monkeypatch.setattr(vision.httpx, "AsyncClient", lambda **kwargs: _Client())

    images = [
        (b"img-one", "image/jpeg"),
        (b"img-two", "image/png"),
    ]
    result = asyncio.run(vision.describe_images(images, caption="смотри"))

    assert result == "два кадра: кот и собака"
    parts = captured["json"]["contents"][0]["parts"]
    # 1 text + 2 images
    assert len(parts) == 3
    assert "text" in parts[0]
    assert "альбом" in parts[0]["text"].lower()
    assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert parts[2]["inline_data"]["mime_type"] == "image/png"
    assert captured["json"]["generationConfig"]["maxOutputTokens"] == 4096


def test_describe_images_returns_policy_placeholder_when_blocked(monkeypatch):
    monkeypatch.setattr(vision, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(vision, "GEMINI_MODEL", "gemini-test")

    class _Resp:
        status_code = 200

        def json(self):
            return {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "candidates": []}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            return _Resp()

    monkeypatch.setattr(vision.httpx, "AsyncClient", lambda **kwargs: _Client())

    result = asyncio.run(vision.describe_image_bytes(b"x", "image/jpeg"))
    assert result == vision.POLICY_BLOCKED_IMAGE
    assert "NSFW" in result or "чувствительн" in result
