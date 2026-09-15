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
        assert kwargs['part_size_kb'] == 512
        with pytest.raises(ValueError, match='size limit'):
            await kwargs['progress_callback'](5, 0)
    asyncio.run(run())


@pytest.mark.parametrize('error', [BufferError, struct.error])
def test_truncated_binary_reader_errors_are_normalized(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error('truncated input')
    monkeypatch.setattr(media.BinaryReader, 'read_int', fail)
    with pytest.raises(ValueError, match='Unsupported or invalid'):
        media.decode_file_id(file_id())


def test_stalled_download_exits_before_total_deadline(monkeypatch):
    monkeypatch.setattr(media, '_DOWNLOAD_IDLE_TIMEOUT', 0.01)

    async def run():
        async def stall(*args, **kwargs):
            await asyncio.Event().wait()
        downloader = media.TelegramMediaDownloader()
        client = SimpleNamespace(download_file=stall)
        monkeypatch.setattr(downloader, '_connect', AsyncMock(return_value=client))
        with pytest.raises(TimeoutError, match='no progress'):
            await asyncio.wait_for(downloader.download(file_id()), 1)
        assert not downloader._lock.locked()
    asyncio.run(run())


def test_progress_extends_idle_deadline(monkeypatch):
    monkeypatch.setattr(media, '_DOWNLOAD_IDLE_TIMEOUT', 0.04)

    async def run():
        async def download(*args, progress_callback, **kwargs):
            for count in range(1, 6):
                await asyncio.sleep(0.02)
                await progress_callback(count, 5)
            return b'data'
        downloader = media.TelegramMediaDownloader()
        client = SimpleNamespace(download_file=download)
        monkeypatch.setattr(downloader, '_connect', AsyncMock(return_value=client))
        assert await downloader.download(file_id()) == b'data'
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


class RangeClient:
    def __init__(self, data, chunk_bytes, *, failure=None):
        self.data = data
        self.chunk_bytes = chunk_bytes
        self.failure = failure
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.ready = asyncio.Event()

    def iter_download(self, location, *, offset, limit, request_size, chunk_size, file_size, dc_id):
        assert request_size == chunk_size == self.chunk_bytes
        assert file_size == len(self.data)
        assert offset % self.chunk_bytes == 0
        assert dc_id == 2
        self.calls.append((offset, limit))
        client = self

        class Iterator:
            def __init__(self):
                self.position = offset
                self.left = limit

            async def __aenter__(self):
                client.active += 1
                client.max_active = max(client.max_active, client.active)
                if client.active == 4:
                    client.ready.set()
                return self

            async def __aexit__(self, *args):
                client.active -= 1

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.left:
                    raise StopAsyncIteration
                await client.ready.wait()
                if client.failure and offset == 0:
                    if client.failure == 'network':
                        raise ConnectionError('download failed')
                    if client.failure == 'empty':
                        raise StopAsyncIteration
                    if client.failure == 'short':
                        return b'x'
                if client.failure:
                    await asyncio.Event().wait()
                # Later ranges arrive first, so accidental append/shared seek
                # implementations cannot pass the byte-for-byte comparison.
                await asyncio.sleep(0.001 if offset else 0.005)
                chunk = client.data[self.position:self.position + chunk_size]
                self.position += len(chunk)
                self.left -= 1
                return chunk
        return Iterator()


@pytest.mark.parametrize('extra', [0, 37])
def test_parallel_download_reassembles_out_of_order_ranges(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(media, '_DOWNLOAD_CHUNK_BYTES', 4096)
    monkeypatch.setattr(media, '_PARALLEL_DOWNLOAD_MIN_BYTES', 1)
    data = (bytes(range(251)) * 200)[:4096 * 9 + extra]

    async def run():
        client = RangeClient(data, 4096)
        downloader = media.TelegramMediaDownloader()
        monkeypatch.setattr(downloader, '_connect', AsyncMock(return_value=client))
        path = tmp_path / 'video.mp4'
        result = await asyncio.wait_for(downloader.download(
            file_id(), file=path, file_size=len(data)
        ), 1)
        assert result == path
        assert path.read_bytes() == data
        assert client.max_active == 4
        assert client.active == 0
        assert len(client.calls) == 4
        assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['network', 'empty', 'short'])
def test_parallel_failure_cancels_and_joins_other_ranges(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(media, '_DOWNLOAD_CHUNK_BYTES', 4096)
    data = bytes(4096 * 9 + 37)

    async def run():
        client = RangeClient(data, 4096, failure=failure)
        with pytest.raises((ConnectionError, ValueError)):
            await asyncio.wait_for(media._download_parallel(
                client, object(), file=tmp_path / 'partial', file_size=len(data),
                dc_id=2, progress_callback=AsyncMock(),
            ), 1)
        assert client.active == 0
        assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(run())


def test_parallel_timeout_removes_partial_file_and_joins_workers(monkeypatch, tmp_path):
    from berangaria.core import utils
    monkeypatch.setattr(media, '_DOWNLOAD_CHUNK_BYTES', 4096)
    monkeypatch.setattr(media, '_PARALLEL_DOWNLOAD_MIN_BYTES', 1)
    monkeypatch.setattr(media, '_DOWNLOAD_IDLE_TIMEOUT', 0.01)
    monkeypatch.setattr(utils.tempfile, 'tempdir', str(tmp_path))
    data = bytes(4096 * 9 + 37)

    async def run():
        client = RangeClient(data, 4096, failure='stall')
        downloader = media.TelegramMediaDownloader()
        monkeypatch.setattr(downloader, '_connect', AsyncMock(return_value=client))
        context = SimpleNamespace(application=SimpleNamespace(
            bot_data={'telegram_media_downloader': downloader}
        ))
        with pytest.raises(TimeoutError, match='no progress'):
            await utils.download_video_to_file(file_id(), context, file_size=len(data))
        assert client.active == 0
        assert not list(tmp_path.iterdir())
        assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(run())


@pytest.mark.parametrize('wait_seconds,failures', [(0, 1), (0, 4), (11, 1)])
def test_parallel_flood_wait_resumes_or_stops_with_bounded_retries(monkeypatch, tmp_path, wait_seconds, failures):
    monkeypatch.setattr(media, '_DOWNLOAD_CHUNK_BYTES', 4096)
    data = (bytes(range(251)) * 200)[:4096 * 9 + 37]

    class FloodClient(RangeClient):
        remaining_failures = failures

        def iter_download(self, *args, **kwargs):
            inner = super().iter_download(*args, **kwargs)
            client = self

            class Iterator:
                async def __aenter__(self):
                    await inner.__aenter__()
                    return self

                async def __aexit__(self, *args):
                    await inner.__aexit__(*args)

                async def __anext__(self):
                    if inner.position == 4096 and client.remaining_failures:
                        client.remaining_failures -= 1
                        raise media.FloodWaitError(request=None, capture=wait_seconds)
                    return await anext(inner)
            return Iterator()

    async def run():
        client = FloodClient(data, 4096)
        path = tmp_path / 'video'
        progress = AsyncMock()
        download = media._download_parallel(
            client, object(), file=path, file_size=len(data), dc_id=2,
            progress_callback=progress,
        )
        if wait_seconds > 10 or failures > 3:
            with pytest.raises(media.FloodWaitError):
                await asyncio.wait_for(download, 5)
        else:
            await asyncio.wait_for(download, 5)
            assert path.read_bytes() == data
            assert (4096, 1) in client.calls
            amounts = [call.args[0] for call in progress.await_args_list]
            assert amounts == sorted(set(amounts))
            assert amounts[-1] == len(data)
        assert client.remaining_failures == 0
        assert client.active == 0
        assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(run())
