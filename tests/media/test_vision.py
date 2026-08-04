"""Unit tests for Gemini vision helpers: multi-image and safety blocks."""

import asyncio

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
