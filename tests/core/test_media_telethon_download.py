"""Downloads must use MTProto even when the HTTP Bot API is available."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from berangaria.core import utils


class Downloader:
    def __init__(self, data, error=None):
        self.data = data
        self.error = error
        self.paths = []

    async def download(self, file_id, *, file=None):
        assert file_id == 'file-id'
        if file:
            self.paths.append(file)
            Path(file).write_bytes(self.data)
        if self.error:
            raise self.error
        return self.data if file is None else file


def context_for(downloader):
    return SimpleNamespace(application=SimpleNamespace(
        bot_data={'telegram_media_downloader': downloader}
    ))


@pytest.mark.parametrize('raw,mime', [
    (b'\xff\xd8\xffphoto', 'image/jpeg'),
    (b'RIFF1234WEBPsticker', 'image/webp'),
    (b'\x89PNG\r\n\x1a\nphoto', 'image/png'),
])
def test_photo_download_uses_telethon(raw, mime):
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
    result = asyncio.run(helper('file-id', context_for(downloader)))
    path = Path(result[0])
    try:
        assert downloader.paths
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
