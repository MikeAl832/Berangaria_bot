import asyncio
from types import SimpleNamespace

import httpx
from berangaria.core import utils


class _FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200):
        self._payload = payload or {}
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        self.calls.append((url, params))
        if url.endswith("/getFile"):
            return _FakeResponse(
                payload={"ok": True, "result": {"file_path": "photos/file_1.jpg"}},
            )
        if "/file/bot" in url:
            return _FakeResponse(content=b"fake-image-bytes")
        raise AssertionError(f"unexpected url {url}")


def test_download_media_prefers_cloud(monkeypatch):
    monkeypatch.setattr(utils, "TELEGRAM_TOKEN", "123:ABC")
    monkeypatch.setattr(utils.httpx, "AsyncClient", _FakeAsyncClient)

    # Local bot should not be consulted on the happy path.
    class BoomBot:
        async def get_file(self, *_args, **_kwargs):
            raise AssertionError("local get_file should not run")

    context = SimpleNamespace(bot=BoomBot())
    raw, mime = asyncio.run(
        utils.download_media_as_base64("file-id", context, return_bytes=True)
    )
    assert raw == b"fake-image-bytes"
    assert mime == "image/jpeg"


def test_download_media_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(utils, "TELEGRAM_TOKEN", "123:ABC")

    class FailingClient(_FakeAsyncClient):
        async def get(self, url, params=None):
            raise httpx.ConnectError("cloud down")

    monkeypatch.setattr(utils.httpx, "AsyncClient", FailingClient)

    class LocalFile:
        file_path = "stickers/sticker.webp"

        async def download_as_bytearray(self, buf):
            buf.extend(b"local-bytes")

    class LocalBot:
        async def get_file(self, file_id):
            assert file_id == "file-id"
            return LocalFile()

    context = SimpleNamespace(bot=LocalBot())
    raw, mime = asyncio.run(
        utils.download_media_as_base64("file-id", context, return_bytes=True)
    )
    assert raw == b"local-bytes"
    assert mime == "image/webp"
