"""Best-effort media → vision description for bridge messages."""

from __future__ import annotations

import logging
from typing import Any, Optional

from berangaria.config import (
    AUDIO_MAX_DURATION_SEC,
    VIDEO_MAX_DURATION_SEC,
    VIDEO_MAX_FILE_SIZE_BYTES,
    VISION_MODE,
)
from berangaria.media.vision import (
    VISION_FAILED_AUDIO,
    VISION_FAILED_IMAGE,
    VISION_FAILED_VIDEO,
    describe_image_bytes,
    describe_video,
    transcribe_audio,
)

logger = logging.getLogger(__name__)


def _guess_image_mime(file_name: str | None) -> str:
    name = (file_name or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    if name.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


async def describe_bridge_media(
    client: Any,
    event_message: Any,
    *,
    caption: str = "",
) -> tuple[Optional[str], Optional[str]]:
    """Download media via Telethon and run the same vision helpers as Bot API path.

    Returns ``(description, media_kind)``. On any failure returns a failed-placeholder
    or ``(None, None)`` when there is no media / vision is off.
    """
    if not VISION_MODE:
        return None, None

    msg = event_message
    has_photo = bool(getattr(msg, "photo", None))
    has_video = bool(getattr(msg, "video", None) or getattr(msg, "video_note", None))
    has_animation = bool(getattr(msg, "gif", None) or getattr(msg, "animation", None))
    # Telethon uses DocumentAttribute* for some kinds; also check document mime.
    document = getattr(msg, "document", None)
    voice = getattr(msg, "voice", None) or getattr(msg, "audio", None)

    kind: Optional[str] = None
    if has_photo:
        kind = "image"
    elif has_video or has_animation:
        kind = "video"
    elif voice:
        kind = "audio"
    elif document is not None:
        mime = (getattr(document, "mime_type", None) or "").lower()
        if mime.startswith("image/"):
            kind = "image"
        elif mime.startswith("video/"):
            kind = "video"
        elif mime.startswith("audio/"):
            kind = "audio"
        else:
            return None, None
    else:
        return None, None

    try:
        raw = await client.download_media(msg, file=bytes)
    except Exception as exc:
        logger.warning("user_bridge: media download failed: %s", exc)
        if kind == "video":
            return VISION_FAILED_VIDEO, kind
        if kind == "audio":
            return VISION_FAILED_AUDIO, kind
        return VISION_FAILED_IMAGE, kind

    if not raw:
        return None, None

    if len(raw) > VIDEO_MAX_FILE_SIZE_BYTES and kind in {"video", "audio"}:
        logger.warning(
            "user_bridge: media too large (%s bytes > %s)",
            len(raw),
            VIDEO_MAX_FILE_SIZE_BYTES,
        )
        return (
            VISION_FAILED_VIDEO if kind == "video" else VISION_FAILED_AUDIO,
            kind,
        )

    try:
        if kind == "image":
            file_name = None
            if document is not None:
                file_name = getattr(document, "file_name", None)
            mime = _guess_image_mime(file_name)
            desc = await describe_image_bytes(raw, mime, caption=caption or "")
            return (desc or VISION_FAILED_IMAGE), "image"

        if kind == "video":
            # describe_video expects a path; write a short-lived temp file.
            import os
            import tempfile

            duration = 0.0
            video_obj = getattr(msg, "video", None) or getattr(msg, "video_note", None)
            if video_obj is not None:
                duration = float(getattr(video_obj, "duration", 0) or 0)
            if duration and duration > VIDEO_MAX_DURATION_SEC:
                return VISION_FAILED_VIDEO, "video"

            suffix = ".mp4"
            fd, path = tempfile.mkstemp(prefix="bridge_vid_", suffix=suffix)
            os.close(fd)
            try:
                with open(path, "wb") as fh:
                    fh.write(raw)
                desc = await describe_video(path, duration=duration, caption=caption or "")
                return (desc or VISION_FAILED_VIDEO), "video"
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        if kind == "audio":
            import os
            import tempfile

            duration = float(getattr(voice, "duration", 0) or 0) if voice else 0.0
            if duration and duration > AUDIO_MAX_DURATION_SEC:
                return VISION_FAILED_AUDIO, "audio"

            suffix = ".ogg"
            fd, path = tempfile.mkstemp(prefix="bridge_aud_", suffix=suffix)
            os.close(fd)
            try:
                with open(path, "wb") as fh:
                    fh.write(raw)
                desc = await transcribe_audio(path, duration=duration)
                return (desc or VISION_FAILED_AUDIO), "audio"
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    except Exception as exc:
        logger.warning("user_bridge: vision failed: %s", exc)
        if kind == "video":
            return VISION_FAILED_VIDEO, kind
        if kind == "audio":
            return VISION_FAILED_AUDIO, kind
        return VISION_FAILED_IMAGE, kind

    return None, None
