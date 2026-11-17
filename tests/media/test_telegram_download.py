"""Unit tests for pure Bot API media downloads and format sniffing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from berangaria.media import telegram_download as media


class _FakeTelegramFile:
    def __init__(self, payload: bytes, *, file_size: int | None = None, delay: float = 0.0):
        self._payload = payload
        self.file_size = len(payload) if file_size is None else file_size
        self._delay = delay

    async def download_as_bytearray(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return bytearray(self._payload)

    async def download_to_drive(self, *, custom_path):
        if self._delay:
            await asyncio.sleep(self._delay)
        Path(custom_path).write_bytes(self._payload)


def _ctx(tg_file: _FakeTelegramFile):
    async def get_file(_file_id: str):
        return tg_file

    return SimpleNamespace(bot=SimpleNamespace(get_file=get_file))


def test_media_suffix_detects_common_headers():
    assert media.media_suffix(b"\xff\xd8\xff\xe0rest") == ".jpg"
    assert media.media_suffix(b"\x89PNG\r\n\x1a\nrest") == ".png"
    assert media.media_suffix(b"GIF89arest") == ".gif"
    assert media.media_suffix(b"RIFF\x00\x00\x00\x00WEBPrest") == ".webp"
    assert media.media_suffix(b"OggSrest") == ".ogg"
    assert media.media_suffix(b"\x00\x00\x00\x18ftypmp42") == ".mp4"
    assert media.media_suffix(b"\x1aE\xdf\xa3...webm...") == ".webm"
    assert media.media_suffix(b"unknown") == ""


def test_download_bytes_via_bot_api():
    payload = b"\xff\xd8\xffhello"
    result = asyncio.run(
        media.download_telegram_file(_ctx(_FakeTelegramFile(payload)), "fid-1")
    )
    assert result == payload


def test_download_to_file_via_bot_api(tmp_path):
    payload = b"\x00\x00\x00\x18ftypmp42video"
    dest = tmp_path / "video.bin"
    result = asyncio.run(
        media.download_telegram_file(
            _ctx(_FakeTelegramFile(payload)), "fid-2", file=str(dest)
        )
    )
    assert result == str(dest)
    assert dest.read_bytes() == payload


def test_download_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(media.config, "VIDEO_MAX_FILE_SIZE_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds configured size limit"):
        asyncio.run(
            media.download_telegram_file(
                _ctx(_FakeTelegramFile(b"01234567890")), "fid-big", file_size=11
            )
        )


def test_download_rejects_empty_payload():
    with pytest.raises(RuntimeError, match="empty media"):
        asyncio.run(
            media.download_telegram_file(_ctx(_FakeTelegramFile(b"")), "fid-empty")
        )


def test_download_honors_timeout(monkeypatch):
    monkeypatch.setattr(media.config, "TELEGRAM_MEDIA_TIMEOUT_SECONDS", 0.02)
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(
            media.download_telegram_file(
                _ctx(_FakeTelegramFile(b"abc", delay=0.2)), "fid-slow"
            )
        )
