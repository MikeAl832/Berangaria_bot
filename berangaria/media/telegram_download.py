"""Telegram cloud Bot API media download helpers (up to 20 MiB)."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from berangaria import config

logger = logging.getLogger(__name__)


def media_suffix(header: bytes) -> str:
    """Determine format from content: file IDs carry no filename or MIME type."""
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if header.startswith(b"RIFF"):
        return {b"WEBP": ".webp", b"WAVE": ".wav", b"AVI ": ".avi"}.get(
            header[8:12], ""
        )
    if header.startswith(b"OggS"):
        return ".ogg"
    if header.startswith(b"fLaC"):
        return ".flac"
    if header.startswith(b"ID3") or (
        len(header) >= 2 and header[0] == 255 and header[1] & 0xE0 == 0xE0
    ):
        return ".aac" if len(header) >= 2 and header[1] & 0xF6 == 0xF0 else ".mp3"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return ".webm" if b"webm" in header else ".mkv"
    if header[4:8] == b"ftyp":
        if header[8:12] == b"qt  ":
            return ".mov"
        return ".m4a" if header[8:12] in {b"M4A ", b"M4B "} else ".mp4"
    return ""


async def download_telegram_file(
    context, file_id: str, *, file=None, file_size: int | None = None
):
    """Download media via Telegram's cloud Bot API getFile (max 20 MiB)."""
    max_bytes = min(
        config.TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES,
        config.VIDEO_MAX_FILE_SIZE_BYTES,
    )
    if file_size is not None and (file_size < 0 or file_size > max_bytes):
        raise ValueError("Telegram media exceeds configured size limit")

    bot = getattr(context, "bot", None)
    if bot is None:
        raise RuntimeError("Telegram bot instance is required for media downloads")

    started = time.monotonic()
    try:
        async with asyncio.timeout(config.TELEGRAM_MEDIA_TIMEOUT_SECONDS):
            telegram_file = await bot.get_file(file_id)
            reported = (
                file_size
                if file_size is not None
                else getattr(telegram_file, "file_size", None)
            )
            if reported is not None and reported > max_bytes:
                raise ValueError("Telegram media exceeds configured size limit")

            if file is not None:
                await telegram_file.download_to_drive(custom_path=file)
                received = Path(file).stat().st_size
                result = file
            else:
                result = bytes(await telegram_file.download_as_bytearray())
                received = len(result)

            if not received:
                raise RuntimeError("Telegram returned empty media")
            if received > max_bytes:
                raise ValueError("Telegram media exceeds configured size limit")

            logger.debug(
                "Telegram Bot API download complete: bytes=%s, seconds=%.1f",
                received,
                time.monotonic() - started,
            )
            return result
    except TimeoutError:
        raise TimeoutError("Telegram Bot API download timed out") from None
