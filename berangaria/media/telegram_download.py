"""Media downloads: HTTP Bot API for small files, bot-owned MTProto above that.

Cloud getFile is capped at 20 MiB. Larger files, or getFile 'file is too big',
use the bot Telethon session. File IDs contain account-specific access hashes,
so the read-only user bridge cannot download the bot's files. Wire layout:
tdlib/td, files/FileLocation.hpp and PhotoSizeSource.hpp (v4). Telethon 1.40's
resolve_bot_file_id discards modern file references.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import struct
import time
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError, FloodPremiumWaitError
from telethon.extensions import BinaryReader
from telethon.sessions import SQLiteSession
from telethon.tl import types

from berangaria import config

logger = logging.getLogger(__name__)
_DOWNLOAD_IDLE_TIMEOUT = 30.0
_DISCONNECT_TIMEOUT = 5.0
_DOWNLOAD_CHUNK_BYTES = 512 * 1024
_PARALLEL_DOWNLOAD_WORKERS = 4
_PARALLEL_DOWNLOAD_MIN_BYTES = 4 * 1024 * 1024


async def _download_parallel(client, location, *, file, file_size, dc_id, progress_callback):
    """Fetch disjoint aligned ranges with bounded memory and verify every byte."""
    chunks = (file_size + _DOWNLOAD_CHUNK_BYTES - 1) // _DOWNLOAD_CHUNK_BYTES
    workers = min(_PARALLEL_DOWNLOAD_WORKERS, chunks)
    received = 0
    flood_until = 0.0
    with open(file, 'wb') as output:
        output.truncate(file_size)

    async def fetch_range(first, last):
        nonlocal received, flood_until
        offset = first * _DOWNLOAD_CHUNK_BYTES
        end = min(last * _DOWNLOAD_CHUNK_BYTES, file_size)
        # Each worker has its own cursor; no shared seek position can corrupt
        # the result when network requests finish out of order.
        with open(file, 'r+b', buffering=0) as output:
            retries = 0
            while offset < end:
                iterator = client.iter_download(
                    location, offset=offset, limit=last - offset // _DOWNLOAD_CHUNK_BYTES,
                    request_size=_DOWNLOAD_CHUNK_BYTES, chunk_size=_DOWNLOAD_CHUNK_BYTES,
                    file_size=file_size, dc_id=dc_id,
                )
                try:
                    async with iterator:
                        while offset < end:
                            # A rate limit applies to all workers. Resume at the
                            # first unwritten chunk after Telegram's requested wait.
                            while flood_until > time.monotonic():
                                await asyncio.sleep(flood_until - time.monotonic())
                            try:
                                chunk = await anext(iterator)
                            except StopAsyncIteration:
                                raise ValueError('Incomplete Telegram media range') from None
                            expected = min(_DOWNLOAD_CHUNK_BYTES, end - offset)
                            if len(chunk) != expected:
                                raise ValueError('Incomplete Telegram media chunk')
                            output.seek(offset)
                            if output.write(chunk) != len(chunk):
                                raise OSError('Incomplete Telegram media write')
                            offset += len(chunk)
                            received += len(chunk)
                            retries = 0
                            await progress_callback(received, file_size)
                except (FloodWaitError, FloodPremiumWaitError) as exc:
                    retries += 1
                    if exc.seconds > 10 or retries > 3:
                        raise
                    flood_until = max(flood_until, time.monotonic() + max(1, exc.seconds))
                    logger.info('Telegram media rate limit: waiting %ss', max(1, exc.seconds))
        if offset != end:
            raise ValueError('Incomplete Telegram media range')

    tasks = [asyncio.create_task(fetch_range(chunks * i // workers, chunks * (i + 1) // workers))
             for i in range(workers)]
    try:
        await asyncio.gather(*tasks)
    finally:
        # A failed/timed-out worker must not leave siblings writing a file
        # after the caller has removed it or started processing another update.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if received != file_size:
        raise ValueError('Incomplete Telegram media download')
    return file


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
    except (ValueError, IndexError, StopIteration, BufferError, binascii.Error, struct.error) as exc:
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



def _force_configured_media_port(client) -> None:
    """Rewrite Telethon DC ports to TELEGRAM_MEDIA_PORT.

    Session home DC is already set to 5222, but downloads from another DC
    export auth and connect via help.GetConfig ports (usually 443). On this
    host 443 is an HTTP middlebox (Pingora), which breaks MTProto with
    IncompleteReadError. Force the configured media port for every DC.
    """
    if getattr(client, '_berangaria_media_port_forced', False):
        return
    original_get_dc = getattr(client, '_get_dc', None)
    if original_get_dc is None:
        return

    async def _get_dc(dc_id, cdn=False):
        dc = await original_get_dc(dc_id, cdn=cdn)
        if getattr(dc, 'port', None) != config.TELEGRAM_MEDIA_PORT:
            logger.info(
                'Telegram media DC%s port %s -> %s',
                dc_id,
                getattr(dc, 'port', None),
                config.TELEGRAM_MEDIA_PORT,
            )
            dc.port = config.TELEGRAM_MEDIA_PORT
        return dc

    client._get_dc = _get_dc  # type: ignore[method-assign]
    client._berangaria_media_port_forced = True


def _task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _is_hung_download_error(exc: BaseException) -> bool:
    """True when Telegram aborted the RPC without cancelling our handler task.

    Telethon's failed reconnect cancels pending GetFile futures. Awaiting that
    cancelled future raises CancelledError in the PTB update fetcher, which
    treats uncaught CancelledError as a dead fetcher and stops processing
    updates. Convert that to TimeoutError so media handlers can fail closed.
    """
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, asyncio.CancelledError) and not _task_is_cancelling()


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
            request_retries=1, flood_sleep_threshold=0, raise_last_call_error=True, timeout=10,
        )
        _force_configured_media_port(self._client)
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

    async def download(self, file_id: str, *, file=None, file_size: int | None = None):
        location, dc_id = decode_file_id(file_id)
        if file_size is not None and (file_size < 0 or file_size > config.VIDEO_MAX_FILE_SIZE_BYTES):
            raise ValueError('Telegram media exceeds configured size limit')
        received = 0
        async with asyncio.timeout(config.TELEGRAM_MEDIA_TIMEOUT_SECONDS):
            async with self._lock:
                client = await self._connect()
                parallel = file is not None and file_size is not None and file_size >= _PARALLEL_DOWNLOAD_MIN_BYTES
                started = time.monotonic()
                logger.info('Telegram media download started: dc=%s, bytes=%s, workers=%s',
                            dc_id, file_size, _PARALLEL_DOWNLOAD_WORKERS if parallel else 1)
                try:
                    async with asyncio.timeout(_DOWNLOAD_IDLE_TIMEOUT) as idle:
                        async def check_size(current, total):
                            nonlocal received
                            if current > config.VIDEO_MAX_FILE_SIZE_BYTES:
                                raise ValueError('Telegram media exceeds configured size limit')
                            if current > received:
                                received = current
                                idle.reschedule(asyncio.get_running_loop().time() + _DOWNLOAD_IDLE_TIMEOUT)

                        # Without file_size, Telethon defaults to 64 KiB requests.
                        # Large videos then require thousands of sequential round
                        # trips. Use Telegram's maximum 512 KiB request size.
                        if parallel:
                            result = await _download_parallel(
                                client, location, file=file, file_size=file_size,
                                dc_id=dc_id, progress_callback=check_size,
                            )
                        else:
                            result = await client.download_file(
                                location, file=file, dc_id=dc_id, part_size_kb=512,
                                file_size=file_size, progress_callback=check_size,
                            )
                    if file_size and received != file_size:
                        raise ValueError('Incomplete Telegram media download')
                    logger.info('Telegram media download complete: bytes=%s, seconds=%.1f',
                                received, time.monotonic() - started)
                    return result
                except BaseException as exc:
                    try:
                        await self._disconnect()
                    except asyncio.CancelledError:
                        if _task_is_cancelling():
                            raise
                        logger.warning('Telegram media session drop cancelled')
                    except Exception:
                        logger.warning('Telegram media session drop failed', exc_info=True)
                    if _is_hung_download_error(exc):
                        raise TimeoutError(
                            f'Telegram media download made no progress for '
                            f'{_DOWNLOAD_IDLE_TIMEOUT:g}s (received {received} bytes)'
                        ) from None
                    raise

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            async with asyncio.timeout(_DISCONNECT_TIMEOUT):
                await client.disconnect()
        except TimeoutError:
            logger.warning('Telegram media client disconnect timed out')
        except asyncio.CancelledError:
            if _task_is_cancelling():
                raise
            logger.warning('Telegram media client disconnect cancelled')
        except Exception:
            logger.warning('Telegram media client disconnect failed', exc_info=True)
        finally:
            try:
                client.session.close()
            except Exception:
                logger.warning('Telegram media session close failed', exc_info=True)

    async def close(self) -> None:
        async with self._lock:
            await self._disconnect()


def get_downloader(context) -> TelegramMediaDownloader:
    data = context.application.bot_data
    if 'telegram_media_downloader' not in data:
        data['telegram_media_downloader'] = TelegramMediaDownloader()
    return data['telegram_media_downloader']


class _BotApiTooBig(Exception):
    """getFile cannot return this file; the caller should use MTProto."""


def uses_bot_api(file_size: int | None) -> bool:
    """True when cloud getFile can accept this file (unknown size counts as small)."""
    limit = config.TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES
    if limit <= 0:
        return False
    return file_size is None or file_size <= limit


def _bot_api_file_too_big(exc: BaseException) -> bool:
    if isinstance(exc, _BotApiTooBig):
        return True
    text = str(exc).lower()
    return 'file is too big' in text or 'file_too_big' in text


async def _download_via_bot_api(bot, file_id: str, *, file=None, file_size: int | None = None):
    started = time.monotonic()
    try:
        async with asyncio.timeout(config.TELEGRAM_MEDIA_TIMEOUT_SECONDS):
            telegram_file = await bot.get_file(file_id)
            reported = file_size if file_size is not None else getattr(telegram_file, 'file_size', None)
            if reported is not None and reported > config.TELEGRAM_BOT_API_DOWNLOAD_MAX_BYTES:
                raise _BotApiTooBig()
            if reported is not None and reported > config.VIDEO_MAX_FILE_SIZE_BYTES:
                raise ValueError('Telegram media exceeds configured size limit')
            logger.info('Telegram Bot API download started: bytes=%s', reported)
            if file is not None:
                await telegram_file.download_to_drive(custom_path=file)
                received = Path(file).stat().st_size
                result = file
            else:
                result = bytes(await telegram_file.download_as_bytearray())
                received = len(result)
            if not received:
                raise RuntimeError('Telegram returned empty media')
            if received > config.VIDEO_MAX_FILE_SIZE_BYTES:
                raise ValueError('Telegram media exceeds configured size limit')
            logger.info(
                'Telegram Bot API download complete: bytes=%s, seconds=%.1f',
                received, time.monotonic() - started,
            )
            return result
    except TimeoutError:
        raise TimeoutError('Telegram Bot API download timed out') from None


async def download_telegram_file(context, file_id: str, *, file=None, file_size: int | None = None):
    """Download via Bot API when the file fits getFile; otherwise MTProto."""
    if file_size is not None and (file_size < 0 or file_size > config.VIDEO_MAX_FILE_SIZE_BYTES):
        raise ValueError('Telegram media exceeds configured size limit')
    bot = getattr(context, 'bot', None)
    if bot is not None and uses_bot_api(file_size):
        try:
            return await _download_via_bot_api(bot, file_id, file=file, file_size=file_size)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError) and _task_is_cancelling():
                raise
            if not _bot_api_file_too_big(exc):
                raise
            logger.info('Telegram Bot API file exceeds getFile limit; using MTProto')
    return await get_downloader(context).download(file_id, file=file, file_size=file_size)


async def stop_media_downloader(application) -> None:
    downloader = application.bot_data.pop('telegram_media_downloader', None)
    if downloader is not None:
        await downloader.close()
