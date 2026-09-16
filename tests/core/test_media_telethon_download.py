"""Small files use Bot API getFile; larger files and getFile-too-big use MTProto."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest

from berangaria.core import utils
from berangaria.media import telegram_download as media


class Downloader:
    def __init__(self, data, error=None):
        self.data = data
        self.error = error
        self.paths = []
        self.file_sizes = []

    async def download(self, file_id, *, file=None, file_size=None):
        assert file_id == 'file-id'
        self.file_sizes.append(file_size)
        if file:
            self.paths.append(file)
            Path(file).write_bytes(self.data)
        if self.error:
            raise self.error
        return self.data if file is None else file


def context_for(downloader, bot=None):
    return SimpleNamespace(
        bot=bot,
        application=SimpleNamespace(bot_data={'telegram_media_downloader': downloader}),
    )


class FakeTelegramFile:
    def __init__(self, data, file_size=None):
        self.data = data
        self.file_size = len(data) if file_size is None else file_size
        self.file_path = 'https://api.telegram.org/file/botTOKEN/doc'

    async def download_as_bytearray(self, **kwargs):
        return bytearray(self.data)

    async def download_to_drive(self, custom_path=None, **kwargs):
        Path(custom_path).write_bytes(self.data)
        return Path(custom_path)


class FakeBot:
    def __init__(self, data=b'', error=None, file_size=None):
        self.data = data
        self.error = error
        self.file_size = file_size
        self.calls = []

    async def get_file(self, file_id):
        self.calls.append(file_id)
        if self.error:
            raise self.error
        return FakeTelegramFile(self.data, self.file_size)


@pytest.mark.parametrize('raw,mime', [
    (b'\xff\xd8\xffphoto', 'image/jpeg'),
    (b'RIFF1234WEBPsticker', 'image/webp'),
    (b'\x89PNG\r\n\x1a\nphoto', 'image/png'),
])
def test_photo_download_without_bot_uses_mtproto(raw, mime):
    assert asyncio.run(utils.download_media_as_base64(
        'file-id', context_for(Downloader(raw)), return_bytes=True
    )) == (raw, mime)


def test_photo_base64():
    assert asyncio.run(utils.download_media_as_base64(
        'file-id', context_for(Downloader(b'GIF89a'))
    )) == ('R0lGODlh', 'image/gif')


@pytest.mark.parametrize('error', [RuntimeError('offline'), asyncio.CancelledError()])
def test_download_failure_never_falls_back_to_http(error):
    with pytest.raises(type(error)):
        asyncio.run(utils.download_media_as_base64('file-id', context_for(Downloader(b'', error))))


@pytest.mark.parametrize('helper,raw,mime,suffix', [
    (utils.download_video_to_file, b'\x1aE\xdf\xa3webm', 'video/webm', '.webm'),
    (utils.download_video_to_file, b'1234ftypqt  ', 'video/quicktime', '.mov'),
    (utils.download_video_to_file, b'1234ftypisom', 'video/mp4', '.mp4'),
    (utils.download_audio_to_file, b'OggSvoice', 'audio/ogg', '.ogg'),
    (utils.download_audio_to_file, b'ID3audio', 'audio/mp3', '.mp3'),
    (utils.download_audio_to_file, b'1234ftypM4A ', 'audio/mp4', '.m4a'),
])
def test_media_streams_to_disk_and_preserves_format(helper, raw, mime, suffix):
    downloader = Downloader(raw)
    result = asyncio.run(helper('file-id', context_for(downloader), file_size=len(raw)))
    path = Path(result[0])
    try:
        assert downloader.paths
        assert downloader.file_sizes == [len(raw)]
        assert path.read_bytes() == raw
        assert path.suffix == suffix
        assert result[1] == mime
        assert not Path(downloader.paths[0]).exists()
    finally:
        path.unlink()


@pytest.mark.parametrize('error', [TimeoutError(), asyncio.CancelledError(), RuntimeError('offline')])
def test_partial_file_is_removed_on_failure(error):
    downloader = Downloader(b'partial', error)
    with pytest.raises(type(error)):
        asyncio.run(utils.download_video_to_file('file-id', context_for(downloader)))
    assert downloader.paths
    assert all(not Path(path).exists() for path in downloader.paths)


def test_empty_download_is_failure_and_removes_file():
    downloader = Downloader(b'')
    with pytest.raises(RuntimeError, match='empty media'):
        asyncio.run(utils.download_audio_to_file('file-id', context_for(downloader)))
    assert all(not Path(path).exists() for path in downloader.paths)


def test_small_video_streams_via_bot_api():
    downloader = Downloader(b'should-not-run')
    bot = FakeBot(b'1234ftypisom')
    result = asyncio.run(utils.download_video_to_file(
        'file-id', context_for(downloader, bot), file_size=12
    ))
    path = Path(result[0])
    try:
        assert bot.calls == ['file-id']
        assert downloader.paths == []
        assert path.read_bytes() == b'1234ftypisom'
        assert result[1] == 'video/mp4'
    finally:
        path.unlink()


def test_small_file_uses_bot_api_not_mtproto():
    downloader = Downloader(b'should-not-run')
    bot = FakeBot(b'\xff\xd8\xffphoto')
    assert asyncio.run(utils.download_media_as_base64(
        'file-id', context_for(downloader, bot), return_bytes=True, file_size=100
    )) == (b'\xff\xd8\xffphoto', 'image/jpeg')
    assert bot.calls == ['file-id']
    assert downloader.file_sizes == []


def test_unknown_size_uses_bot_api():
    downloader = Downloader(b'should-not-run')
    bot = FakeBot(b'\xff\xd8\xffphoto')
    assert asyncio.run(utils.download_media_as_base64(
        'file-id', context_for(downloader, bot), return_bytes=True
    )) == (b'\xff\xd8\xffphoto', 'image/jpeg')
    assert bot.calls == ['file-id']
    assert downloader.file_sizes == []


def test_large_file_uses_mtproto_not_bot_api(monkeypatch):
    monkeypatch.setattr(media.config, 'TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES', 20)
    downloader = Downloader(b'\x1aE\xdf\xa3webm')
    bot = FakeBot(b'should-not-run')
    result = asyncio.run(utils.download_video_to_file(
        'file-id', context_for(downloader, bot), file_size=21
    ))
    path = Path(result[0])
    try:
        assert bot.calls == []
        assert downloader.file_sizes == [21]
        assert path.read_bytes() == b'\x1aE\xdf\xa3webm'
    finally:
        path.unlink()


def test_getfile_too_big_falls_back_to_mtproto():
    downloader = Downloader(b'\xff\xd8\xffphoto')
    bot = FakeBot(error=RuntimeError('Bad Request: file is too big'))
    assert asyncio.run(utils.download_media_as_base64(
        'file-id', context_for(downloader, bot), return_bytes=True
    )) == (b'\xff\xd8\xffphoto', 'image/jpeg')
    assert bot.calls == ['file-id']
    assert downloader.file_sizes == [None]


def test_bot_api_other_errors_do_not_fall_back_to_mtproto():
    downloader = Downloader(b'should-not-run')
    bot = FakeBot(error=RuntimeError('httpx.ConnectError'))
    with pytest.raises(RuntimeError, match='ConnectError'):
        asyncio.run(utils.download_media_as_base64(
            'file-id', context_for(downloader, bot), return_bytes=True
        ))
    assert bot.calls == ['file-id']
    assert downloader.file_sizes == []


def test_mtproto_failure_does_not_fall_back_to_bot_api():
    downloader = Downloader(b'', error=RuntimeError('offline'))
    bot = FakeBot(b'should-not-run')
    with pytest.raises(RuntimeError, match='offline'):
        asyncio.run(utils.download_video_to_file(
            'file-id', context_for(downloader, bot),
            file_size=media.config.TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES + 1,
        ))
    assert bot.calls == []
    assert downloader.file_sizes == [media.config.TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES + 1]
