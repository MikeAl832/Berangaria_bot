import asyncio
import base64
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telethon.tl import types

from berangaria.media import telegram_download as media


def file_id(kind=4, *, minor=35, reference=b'reference', source=b''):
    # TDLib persistent remote file ID wire fixture; no production credentials.
    ref = bytes([len(reference)]) + reference
    ref += bytes(-len(ref) % 4)
    raw = struct.pack('<ii', kind | (1 << 25), 2) + ref
    raw += struct.pack('<qq', 123456, -987654) + source + bytes([minor, 4])
    encoded = bytearray()
    i = 0
    while i < len(raw):
        if raw[i]:
            encoded.append(raw[i])
            i += 1
        else:
            end = i + 1
            while end < len(raw) and raw[end] == 0 and end - i < 255:
                end += 1
            encoded.extend((0, end - i))
            i = end
    return base64.urlsafe_b64encode(encoded).decode().rstrip('=')


@pytest.mark.parametrize('kind', [3, 4, 5, 8, 9, 10, 13, 17])
def test_document_file_id_keeps_reference(kind):
    location, dc = media.decode_file_id(file_id(kind))
    assert isinstance(location, types.InputDocumentFileLocation)
    assert (location.id, location.access_hash, location.file_reference, dc) == (
        123456, -987654, b'reference', 2
    )
    assert location.thumb_size == ''


@pytest.mark.parametrize('minor', [30, 32, 35])
def test_photo_file_id_keeps_requested_size(minor):
    source = struct.pack('<iii', 1, 2, ord('y'))
    if minor < 32:
        source = struct.pack('<q', 777) + source + struct.pack('<i', 88)
    location, dc = media.decode_file_id(file_id(2, minor=minor, source=source))
    assert isinstance(location, types.InputPhotoFileLocation)
    assert location.thumb_size == 'y'
    assert location.file_reference == b'reference'
    assert dc == 2


def test_new_group_photo_location():
    source = struct.pack('<iqq', 3, -1000000000042, 456)
    location, _ = media.decode_file_id(file_id(1, source=source))
    assert isinstance(location, types.InputPeerPhotoFileLocation)
    assert location.peer.channel_id == 42
    assert location.peer.access_hash == 456
    assert location.big


@pytest.mark.parametrize('value', ['', 'not a file ID', 'AA', 'AAAA', file_id(6), file_id(2)])
def test_rejects_invalid_or_unsupported_file_id(value):
    with pytest.raises(ValueError, match='Unsupported or invalid'):
        media.decode_file_id(value)


def test_long_file_reference():
    location, _ = media.decode_file_id(file_id(reference=bytes(range(200))))
    assert location.file_reference == bytes(range(200))


def test_download_routes_to_correct_dc_and_enforces_limit(monkeypatch):
    monkeypatch.setattr(media.config, 'VIDEO_MAX_FILE_SIZE_BYTES', 4)

    async def run():
        downloader = media.TelegramMediaDownloader()
        client = SimpleNamespace(download_file=AsyncMock(return_value=b'data'))
        monkeypatch.setattr(downloader, '_connect', AsyncMock(return_value=client))
        assert await downloader.download(file_id()) == b'data'
        kwargs = client.download_file.call_args.kwargs
        assert kwargs['dc_id'] == 2
        with pytest.raises(ValueError, match='size limit'):
            await kwargs['progress_callback'](5, 0)
    asyncio.run(run())


def test_timeout_releases_download_slot(monkeypatch):
    monkeypatch.setattr(media.config, 'TELEGRAM_MEDIA_TIMEOUT_SECONDS', 0.01)

    async def run():
        downloader = media.TelegramMediaDownloader()
        async def stall():
            await asyncio.Event().wait()
        monkeypatch.setattr(downloader, '_connect', stall)
        with pytest.raises(TimeoutError):
            await downloader.download(file_id())
        assert not downloader._lock.locked()
    asyncio.run(run())


def test_shutdown_closes_once():
    async def run():
        downloader = SimpleNamespace(close=AsyncMock())
        app = SimpleNamespace(bot_data={'telegram_media_downloader': downloader})
        await media.stop_media_downloader(app)
        await media.stop_media_downloader(app)
        downloader.close.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize('authorized', [True, False])
@pytest.mark.parametrize('session_name', ['media.session', 'media'])
def test_bot_session_authenticates_without_interactive_login(monkeypatch, tmp_path, authorized, session_name):
    from unittest.mock import Mock
    monkeypatch.setattr(media.config, 'TELEGRAM_MEDIA_SESSION_PATH', str(tmp_path / session_name))
    monkeypatch.setattr(media.config, 'TELEGRAM_API_ID', 123)
    monkeypatch.setattr(media.config, 'TELEGRAM_API_HASH', 'test-hash')
    monkeypatch.setattr(media.config, 'TELEGRAM_TOKEN', '42:test-token')
    monkeypatch.setattr(media.config, 'TELEGRAM_MEDIA_PORT', 5222)
    clients = []
    def factory(session, *args, **kwargs):
        assert session.port == 5222
        assert kwargs['receive_updates'] is False
        assert kwargs['auto_reconnect'] is False
        client = SimpleNamespace(
            session=session, connect=AsyncMock(), disconnect=AsyncMock(),
            is_connected=Mock(return_value=True),
            is_user_authorized=AsyncMock(return_value=authorized),
            get_me=AsyncMock(return_value=SimpleNamespace(id=42, bot=True)),
            sign_in=AsyncMock(),
        )
        clients.append(client)
        return client
    monkeypatch.setattr(media, 'TelegramClient', factory)
    async def run():
        downloader = media.TelegramMediaDownloader()
        client = await downloader._connect()
        assert await downloader._connect() is client
        assert len(clients) == 1
        if authorized:
            client.sign_in.assert_not_awaited()
        else:
            client.sign_in.assert_awaited_once_with(bot_token='42:test-token')
        assert (tmp_path / 'media.session').stat().st_mode & 0o777 == 0o600
        await downloader.close()
        client.disconnect.assert_awaited_once()
        assert downloader._client is None
    asyncio.run(run())


@pytest.mark.parametrize('failure', [RuntimeError('auth failed'), asyncio.CancelledError()])
def test_failed_start_closes_session(monkeypatch, tmp_path, failure):
    from unittest.mock import Mock
    monkeypatch.setattr(media.config, 'TELEGRAM_MEDIA_SESSION_PATH', str(tmp_path / 'media.session'))
    monkeypatch.setattr(media.config, 'TELEGRAM_API_ID', 123)
    monkeypatch.setattr(media.config, 'TELEGRAM_API_HASH', 'test-hash')
    client = SimpleNamespace(connect=AsyncMock(side_effect=failure), disconnect=AsyncMock(),
                             session=SimpleNamespace(close=Mock()))
    def factory(session, *args, **kwargs):
        session.close()
        return client
    monkeypatch.setattr(media, 'TelegramClient', factory)
    async def run():
        downloader = media.TelegramMediaDownloader()
        with pytest.raises(type(failure)):
            await downloader._connect()
        assert downloader._client is None
        client.disconnect.assert_awaited_once()
        client.session.close.assert_called_once()
    asyncio.run(run())
