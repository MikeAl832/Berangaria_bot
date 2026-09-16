"""Empty Gemini video descriptions must become a visible failure placeholder."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from berangaria.chat import media_handlers
from berangaria.media.vision import VISION_FAILED_VIDEO


def _runtime(**overrides):
    queued = []

    async def queue_message(update, context, text="", media_description=None, media_kind=None):
        queued.append(
            {
                "text": text,
                "media_description": media_description,
                "media_kind": media_kind,
            }
        )

    base = dict(
        vision_mode=True,
        album_gather_seconds=0.1,
        max_media_items_in_context=10,
        video_max_duration_sec=300,
        video_max_file_size_bytes=100_000_000,
        audio_max_duration_sec=300,
        vision_failed_image="(image fail)",
        vision_failed_video=VISION_FAILED_VIDEO,
        check_access_permissions=lambda chat_id, user_id, is_group: True,
        queue_message=queue_message,
        download_media_as_base64=AsyncMock(),
        download_video_to_file=AsyncMock(return_value=("/tmp/v.mp4", "video/mp4", 1000)),
        download_audio_to_file=AsyncMock(),
        get_video_duration=lambda obj: 3.0,
        describe_image_bytes=AsyncMock(return_value=""),
        describe_images=AsyncMock(return_value=""),
        describe_video=AsyncMock(return_value=""),
        transcribe_audio=AsyncMock(return_value=""),
    )
    base.update(overrides)
    return media_handlers.MediaRuntime(**base), queued


def test_empty_video_description_becomes_failure_placeholder(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")

    runtime, queued = _runtime(
        download_video_to_file=AsyncMock(
            return_value=(str(video_path), "video/mp4", 1000)
        ),
        describe_video=AsyncMock(return_value=""),
    )
    monkeypatch.setattr(media_handlers.state, "get_cached_media_description", lambda _k: None)
    monkeypatch.setattr(media_handlers.state, "cache_media_description", lambda *a, **k: None)

    update = SimpleNamespace(
        message=SimpleNamespace(
            video=SimpleNamespace(file_id="f1", file_unique_id="u1", file_size=1000),
            video_note=None,
            animation=None,
            caption="",
            reply_text=AsyncMock(),
        ),
        effective_user=SimpleNamespace(id=42),
        effective_chat=SimpleNamespace(id=-100, type="supergroup"),
    )
    context = SimpleNamespace(bot=SimpleNamespace(id=1))

    asyncio.run(media_handlers.handle_video(update, context, runtime))

    assert len(queued) == 1
    assert queued[0]["media_kind"] == "video"
    assert queued[0]["media_description"] == VISION_FAILED_VIDEO
    assert "попроси скинуть" in queued[0]["media_description"]


def test_vision_prompt_mentions_technical_failure_resend():
    from berangaria.prompts import VISION_PROMPT_SUFFIX

    assert "technical-failure" in VISION_PROMPT_SUFFIX
    assert "send the same file again" in VISION_PROMPT_SUFFIX
