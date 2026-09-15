"""Bot-owned MTProto downloads; HTTP Bot API is only used for updates/replies.

File IDs contain account-specific access hashes and file references, so the
read-only user bridge cannot download the bot's files with its user identity.
Wire layout: tdlib/td, files/FileLocation.hpp and PhotoSizeSource.hpp (v4).
Telethon 1.40's resolve_bot_file_id discards modern file references.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from pathlib import Path

from telethon import TelegramClient
from telethon.extensions import BinaryReader
from telethon.sessions import SQLiteSession
from telethon.tl import types

from berangaria import config


def decode_file_id(file_id: str) -> tuple[object, int]:
    """Decode the document/photo locations emitted by the Bot API.

    Reject unknown layouts rather than guessing a different file or size.
    Never include a file ID (which grants access to media) in an error.
    """
    try:
        packed = base64.b64decode(file_id + '=' * (-len(file_id) % 4),
                                  altchars=b'-_', validate=True)
        data = bytearray()
        cursor = iter(packed)
        for value in cursor:
            if value:
                data.append(value)
            else:
                count = next(cursor)
                if not count:
                    raise ValueError
                data.extend(bytes(count))
        if len(data) < 26 or data[-1] != 4:
            raise ValueError
        minor = data[-2]
        with BinaryReader(bytes(data[:-2])) as reader:
            flags, dc_id = reader.read_int(), reader.read_int()
            if flags & (1 << 24) or not 1 <= dc_id <= 5:
                raise ValueError
            reference = reader.tgread_bytes() if flags & (1 << 25) else b''
            kind = flags & ~(1 << 25)
            media_id, access_hash = reader.read_long(), reader.read_long()
            args = dict(id=media_id, access_hash=access_hash, file_reference=reference)
            if kind in {3, 4, 5, 8, 9, 10, 13, 17}:
                return types.InputDocumentFileLocation(**args, thumb_size=''), dc_id
            if kind not in {0, 1, 2}:
                raise ValueError
            volume_id = reader.read_long() if minor < 32 else None
            source = reader.read_int() if minor >= 22 else 0
            if source == 1:
                photo_type, size = reader.read_int(), reader.read_int()
                if photo_type != kind or not ord('a') <= size <= ord('z'):
                    raise ValueError
                location = types.InputPhotoFileLocation if kind == 2 else types.InputDocumentFileLocation
                return location(**args, thumb_size=chr(size)), dc_id
            if source in {2, 3}:
                chat_id, chat_hash = reader.read_long(), reader.read_long()
                if chat_id > 0:
                    peer = types.InputPeerUser(chat_id, chat_hash)
                elif chat_id <= -1000000000000:
                    peer = types.InputPeerChannel(-chat_id - 1000000000000, chat_hash)
                else:
                    peer = types.InputPeerChat(-chat_id)
                return types.InputPeerPhotoFileLocation(peer, media_id, big=source == 3), dc_id
            if source in {0, 5}:
                if source == 5:
                    volume_id = reader.read_long()
                secret, local_id = reader.read_long(), reader.read_int()
                if volume_id is None:
                    raise ValueError
                return types.InputPhotoLegacyFileLocation(
                    **args, volume_id=volume_id, local_id=local_id, secret=secret
                ), dc_id
            raise ValueError
    except (ValueError, IndexError, StopIteration, BufferError, binascii.Error) as exc:
        raise ValueError('Unsupported or invalid Telegram media file ID') from exc


def media_suffix(header: bytes) -> str:
    """Determine format from content: file IDs carry no filename or MIME type."""
    if header.startswith(b'\xff\xd8\xff'):
        return '.jpg'
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if header.startswith((b'GIF87a', b'GIF89a')):
        return '.gif'
    if header.startswith(b'RIFF'):
        return {b'WEBP': '.webp', b'WAVE': '.wav', b'AVI ': '.avi'}.get(header[8:12], '')
    if header.startswith(b'OggS'):
        return '.ogg'
    if header.startswith(b'fLaC'):
        return '.flac'
    if header.startswith(b'ID3') or (len(header) >= 2 and header[0] == 255 and header[1] & 0xe0 == 0xe0):
        return '.aac' if len(header) >= 2 and header[1] & 0xf6 == 0xf0 else '.mp3'
    if header.startswith(b'\x1aE\xdf\xa3'):
        return '.webm' if b'webm' in header else '.mkv'
    if header[4:8] == b'ftyp':
        if header[8:12] == b'qt  ':
            return '.mov'
        return '.m4a' if header[8:12] in {b'M4A ', b'M4B '} else '.mp4'
    return ''


class TelegramMediaDownloader:
    """One lazy bot session, serialized downloads, bounded network waits."""

    def __init__(self) -> None:
        self._client = None
        self._lock = asyncio.Lock()

    async def _connect(self):
        if self._client is not None and self._client.is_connected():
            return self._client
        await self._disconnect()
        if not (config.TELEGRAM_API_ID and config.TELEGRAM_API_HASH):
            raise RuntimeError('Media downloads require TELEGRAM_API_ID and TELEGRAM_API_HASH')
        session_path = Path(config.TELEGRAM_MEDIA_SESSION_PATH)
        if not str(session_path).endswith('.session'):
            session_path = Path(str(session_path) + '.session')
        session_path.parent.mkdir(parents=True, exist_ok=True)
        # Precreate with private permissions; SQLite opens the same file.
        session_path.touch(mode=0o600, exist_ok=True)
        session_path.chmod(0o600)
        session = SQLiteSession(str(session_path))
        session.set_dc(session.dc_id or 2, session.server_address or '149.154.167.51',
                       config.TELEGRAM_MEDIA_PORT)
        self._client = TelegramClient(
            session, config.TELEGRAM_API_ID, config.TELEGRAM_API_HASH,
            receive_updates=False, auto_reconnect=False, connection_retries=0,
            request_retries=1, flood_sleep_threshold=0, timeout=10,
        )
        try:
            async with asyncio.timeout(30):
                await self._client.connect()
                if not await self._client.is_user_authorized():
                    await self._client.sign_in(bot_token=config.TELEGRAM_TOKEN)
                me = await self._client.get_me()
                if not me.bot or str(me.id) != config.TELEGRAM_TOKEN.split(':', 1)[0]:
                    raise RuntimeError('Telegram media session belongs to a different account')
            return self._client
        except BaseException:
            await self._disconnect()
            raise

    async def download(self, file_id: str, *, file=None):
        location, dc_id = decode_file_id(file_id)

        async def check_size(current, total):
            if current > config.VIDEO_MAX_FILE_SIZE_BYTES:
                raise ValueError('Telegram media exceeds configured size limit')

        async with asyncio.timeout(config.TELEGRAM_MEDIA_TIMEOUT_SECONDS):
            async with self._lock:
                client = await self._connect()
                return await client.download_file(
                    location, file=file, dc_id=dc_id, progress_callback=check_size,
                )

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.disconnect()
            finally:
                client.session.close()

    async def close(self) -> None:
        async with self._lock:
            await self._disconnect()


def get_downloader(context) -> TelegramMediaDownloader:
    data = context.application.bot_data
    if 'telegram_media_downloader' not in data:
        data['telegram_media_downloader'] = TelegramMediaDownloader()
    return data['telegram_media_downloader']


async def stop_media_downloader(application) -> None:
    downloader = application.bot_data.pop('telegram_media_downloader', None)
    if downloader is not None:
        await downloader.close()
