"""Telegram media handlers and photo-album gathering.

The orchestration module supplies :class:`MediaRuntime` explicitly. This keeps
the media pipeline independent from ``chat.handlers`` while preserving its
existing public handler functions and test seams.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from berangaria.core import state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaRuntime:
    """Configuration and call boundaries required by the media pipeline."""

    vision_mode: bool
    album_gather_seconds: float
    max_media_items_in_context: int
    video_max_duration_sec: float
    video_max_file_size_bytes: int
    audio_max_duration_sec: float
    vision_failed_image: str
    check_access_permissions: Callable[[int, int, bool], bool]
    queue_message: Callable[..., Awaitable[None]]
    download_media_as_base64: Callable[..., Awaitable[Any]]
    download_video_to_file: Callable[..., Awaitable[Any]]
    download_audio_to_file: Callable[..., Awaitable[Any]]
    get_video_duration: Callable[[Any], float]
    describe_image_bytes: Callable[..., Awaitable[str]]
    describe_images: Callable[..., Awaitable[str]]
    describe_video: Callable[..., Awaitable[str]]
    transcribe_audio: Callable[..., Awaitable[str]]


# Telegram delivers album photos as separate updates with the same
# media_group_id. The timer is restarted after every new member.
album_buffer: dict[str, dict] = {}
album_lock = asyncio.Lock()


def album_cache_key(file_unique_ids: list[str]) -> str:
    """Return a stable cache key for a multi-image album."""
    return "album:" + ",".join(sorted(file_unique_ids))


async def flush_album_after_delay(album_key: str, runtime: MediaRuntime) -> None:
    """Wait for late album members, then process all photos together."""
    try:
        await asyncio.sleep(runtime.album_gather_seconds)
        async with album_lock:
            data = album_buffer.pop(album_key, None)
        if not data:
            return
        await process_photo_album(data, runtime)
    except asyncio.CancelledError:
        # Timer restarted because another photo joined the album.
        pass
    except Exception as error:
        logger.error("❌ [red]Ошибка обработки альбома:[/] %s", error)


async def process_photo_album(data: dict, runtime: MediaRuntime) -> None:
    """Download an album and send one multi-image vision request."""
    items: list[dict] = data["items"]
    update = data["update"]
    context = data["context"]
    if not items:
        return

    items = items[: runtime.max_media_items_in_context]
    unique_ids = [item["file_unique_id"] for item in items]
    cache_key = album_cache_key(unique_ids)
    captions = [item["caption"] for item in items if item.get("caption")]
    caption = "\n".join(captions)

    # Telegram normally puts the caption on one album member. Prefer that
    # update so reply threading points to the visible caption.
    for item in items:
        if item.get("caption") and item.get("update") is not None:
            update = item["update"]
            break

    cached = state.get_cached_media_description(cache_key)
    if cached is not None:
        logger.info(
            "♻️ [dim]Альбом (%s фото) уже разобран, берём из кэша[/]",
            len(items),
        )
        await runtime.queue_message(
            update,
            context,
            text=caption,
            media_description=cached,
            media_kind="image",
        )
        return

    images: list[tuple[bytes, str]] = []
    for item in items:
        try:
            image_bytes, mime = await runtime.download_media_as_base64(
                item["file_id"], context, return_bytes=True
            )
            images.append((image_bytes, mime))
        except Exception as error:
            logger.error(
                "❌ [red]Не удалось скачать фото альбома %s:[/] %s",
                item.get("file_unique_id"),
                error,
            )

    if not images:
        image_description = runtime.vision_failed_image
    else:
        try:
            image_description = await runtime.describe_images(images, caption=caption)
        except Exception as error:
            logger.error("❌ [red]Ошибка multi-image vision:[/] %s", error)
            image_description = ""
        if not image_description:
            image_description = runtime.vision_failed_image
        else:
            state.cache_media_description(cache_key, image_description)

    logger.info(
        "🖼️ [dim]Альбом: %s/%s фото → 1 Gemini-вызов[/]",
        len(images),
        len(items),
    )
    await runtime.queue_message(
        update,
        context,
        text=caption,
        media_description=image_description,
        media_kind="image",
    )


async def buffer_album_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_group_id: str,
    caption: str,
    runtime: MediaRuntime,
) -> None:
    """Accumulate one Telegram album until its quiet window expires."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    photo = update.message.photo[-1]
    album_key = f"{chat_id}_{user_id}_{media_group_id}"
    item = {
        "file_id": photo.file_id,
        "file_unique_id": photo.file_unique_id,
        "caption": caption or "",
        "update": update,
    }

    async with album_lock:
        if album_key in album_buffer:
            entry = album_buffer[album_key]
            task = entry.get("task")
            if task is not None:
                task.cancel()
            entry["items"].append(item)
            entry["context"] = context
            if caption:
                entry["update"] = update
        else:
            album_buffer[album_key] = {
                "items": [item],
                "update": update,
                "context": context,
            }
        album_buffer[album_key]["task"] = asyncio.create_task(
            flush_album_after_delay(album_key, runtime)
        )


async def handle_media(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    runtime: MediaRuntime,
) -> None:
    """Handle a photo or a member of a photo album."""
    if update.message is None:
        return
    if update.effective_user.id == context.bot.id:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ["group", "supergroup"]
    if not runtime.check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    caption = update.message.caption or ""
    if not runtime.vision_mode:
        if caption:
            await runtime.queue_message(update, context, text=caption)
        return

    media_group_id = getattr(update.message, "media_group_id", None)
    if media_group_id:
        await buffer_album_photo(
            update, context, str(media_group_id), caption, runtime
        )
        return

    image_description = None
    try:
        photo = update.message.photo[-1]
        cached = state.get_cached_media_description(photo.file_unique_id)
        if cached is not None:
            logger.info("♻️ [dim]Фото уже разобрано ранее, берём из кэша[/]")
            image_description = cached
        else:
            image_bytes, mime = await runtime.download_media_as_base64(
                photo.file_id, context, return_bytes=True
            )
            image_description = await runtime.describe_image_bytes(
                image_bytes, mime, caption=caption
            )
            if image_description:
                state.cache_media_description(
                    photo.file_unique_id, image_description
                )
    except Exception as error:
        logger.error("❌ [red]Ошибка обработки фото:[/] %s", error)

    if not image_description:
        image_description = runtime.vision_failed_image

    await runtime.queue_message(
        update,
        context,
        text=caption,
        media_description=image_description,
        media_kind="image",
    )


async def handle_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    runtime: MediaRuntime,
) -> None:
    """Handle Telegram videos, video notes, and animations."""
    if update.message is None:
        return
    if update.effective_user.id == context.bot.id:
        return

    caption = update.message.caption or ""
    if not runtime.vision_mode:
        if caption:
            await runtime.queue_message(update, context, text=caption)
        return

    video_obj = (
        update.message.video
        or update.message.video_note
        or update.message.animation
    )
    if video_obj is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ["group", "supergroup"]
    if not runtime.check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    duration = runtime.get_video_duration(video_obj)
    if duration and duration > runtime.video_max_duration_sec:
        await update.message.reply_text(
            f"Видео длиннее {runtime.video_max_duration_sec} сек — не буду смотреть."
        )
        return
    if (
        video_obj.file_size
        and video_obj.file_size > runtime.video_max_file_size_bytes
    ):
        await update.message.reply_text(
            "Видео больше настроенного лимита "
            f"{runtime.video_max_file_size_bytes // (1024 * 1024)} МБ — "
            "не буду скачивать."
        )
        return

    cached = state.get_cached_media_description(video_obj.file_unique_id)
    if cached is not None:
        logger.info("♻️ [dim]Видео уже разобрано ранее, берём из кэша[/]")
        await runtime.queue_message(
            update,
            context,
            text=caption,
            media_description=cached,
            media_kind="video",
        )
        return

    video_description = None
    video_path = None
    try:
        video_path, mime, _ = await runtime.download_video_to_file(
            video_obj.file_id, context
        )
        if not video_path:
            video_description = "(не удалось скачать видео)"
        else:
            video_description = await runtime.describe_video(
                video_path=video_path,
                mime=mime,
                caption=caption,
                duration=duration,
            )
            video_path = None
    except Exception as error:
        logger.error(
            "❌ [red]Ошибка обработки видео:[/] %s", error, exc_info=True
        )
        video_description = "(не удалось разобрать видео)"
    finally:
        if video_path:
            try:
                import os

                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.debug(
                        "🗑️ Удалён временный файл (fallback): %s", video_path
                    )
            except OSError as error:
                logger.warning(
                    "⚠️ Не удалось удалить временный файл %s: %s",
                    video_path,
                    error,
                )

    if video_description and not video_description.startswith("(не удалось"):
        state.cache_media_description(video_obj.file_unique_id, video_description)

    await runtime.queue_message(
        update,
        context,
        text=caption,
        media_description=video_description,
        media_kind="video",
    )


async def handle_sticker(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    runtime: MediaRuntime,
) -> None:
    """Handle static, animated, and video stickers."""
    if update.message is None:
        return
    if update.effective_user.id == context.bot.id:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ["group", "supergroup"]
    if not runtime.check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    sticker = update.message.sticker
    if sticker is None:
        return
    emoji = sticker.emoji or ""

    if not runtime.vision_mode:
        await runtime.queue_message(update, context, text=(emoji or "(стикер)"))
        return
    if sticker.is_animated:
        description = (
            f"Анимированный стикер с эмодзи {emoji}"
            if emoji
            else "Анимированный стикер"
        )
        await runtime.queue_message(
            update,
            context,
            text="",
            media_description=description,
            media_kind="image",
        )
        return

    cached = state.get_cached_media_description(sticker.file_unique_id)
    if cached is not None:
        logger.info("♻️ [dim]Стикер уже разобран ранее, берём из кэша[/]")
        await runtime.queue_message(
            update,
            context,
            text="",
            media_description=cached,
            media_kind="image",
        )
        return

    sticker_description = None
    sticker_kind = "image"
    hint = (
        f"Это стикер из Telegram с эмодзи {emoji}."
        if emoji
        else "Это стикер из Telegram."
    )
    try:
        if sticker.is_video:
            sticker_kind = "video"
            video_path, mime, duration = await runtime.download_video_to_file(
                sticker.file_id, context
            )
            if not video_path:
                sticker_description = "(не удалось скачать стикер)"
            else:
                sticker_description = await runtime.describe_video(
                    video_path=video_path,
                    mime=mime,
                    caption=hint,
                    duration=duration,
                )
        else:
            image_bytes, mime = await runtime.download_media_as_base64(
                sticker.file_id, context, return_bytes=True
            )
            sticker_description = await runtime.describe_image_bytes(
                image_bytes, mime, caption=hint
            )
    except Exception as error:
        logger.error("❌ [red]Ошибка обработки стикера:[/] %s", error)

    if not sticker_description:
        sticker_description = (
            f"Стикер с эмодзи {emoji}"
            if emoji
            else "(не удалось разобрать стикер)"
        )
    elif not sticker_description.startswith("(не удалось"):
        state.cache_media_description(
            sticker.file_unique_id, sticker_description
        )

    await runtime.queue_message(
        update,
        context,
        text="",
        media_description=sticker_description,
        media_kind=sticker_kind,
    )


async def handle_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    runtime: MediaRuntime,
) -> None:
    """Transcribe Telegram voice messages and audio files."""
    if update.message is None:
        return
    if update.effective_user.id == context.bot.id:
        return
    if not runtime.vision_mode:
        return

    audio_obj = update.message.voice or update.message.audio
    if audio_obj is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ["group", "supergroup"]
    if not runtime.check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    duration = runtime.get_video_duration(audio_obj)
    if duration and duration > runtime.audio_max_duration_sec:
        await update.message.reply_text(
            f"Аудио длиннее {runtime.audio_max_duration_sec} сек — слушать не буду."
        )
        return

    caption = update.message.caption or ""
    transcript = state.get_cached_media_description(audio_obj.file_unique_id)
    if transcript is None:
        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass
        try:
            audio_path, mime = await runtime.download_audio_to_file(
                audio_obj.file_id, context
            )
            if not audio_path:
                transcript = ""
            else:
                transcript = await runtime.transcribe_audio(
                    audio_path=audio_path, mime=mime, caption=caption
                )
        except Exception as error:
            logger.error(
                "❌ [red]Ошибка обработки голосового:[/] %s", error, exc_info=True
            )
            transcript = ""
        if transcript:
            state.cache_media_description(audio_obj.file_unique_id, transcript)

    if not transcript:
        await runtime.queue_message(
            update,
            context,
            text="",
            media_description="(голосовое сообщение, не удалось распознать)",
            media_kind="audio",
        )
        return

    logger.info(
        "🎤 [cyan]Транскрипция:[/] %s%s",
        transcript[:80],
        "..." if len(transcript) > 80 else "",
    )
    await runtime.queue_message(
        update,
        context,
        text=caption,
        media_description=transcript,
        media_kind="audio",
    )
