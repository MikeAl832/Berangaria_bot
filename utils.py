import re
import time
import random
import base64
import asyncio
import os
import tempfile
import subprocess
import logging
from telegram import Update
from telegram.ext import ContextTypes
from config import BOT_NAMES, RANDOM_REPLY_CHANCE
from state import random_reply_cooldown

logger = logging.getLogger(__name__)

def get_bot_real_name(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.bot.first_name

def should_reply_randomly(chat_id: int) -> bool:
    last_reply = random_reply_cooldown.get(chat_id, 0)
    current_time = time.time()
    if current_time - last_reply < 30:
        return False
    if random.randint(1, 100) <= RANDOM_REPLY_CHANCE:
        random_reply_cooldown[chat_id] = current_time
        return True
    return False

def is_bot_mentioned(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple:
    if update.message is None:
        return False, None

    message_text = update.message.text or update.message.caption or ""
    bot_real_name = get_bot_real_name(context)
    bot_username = context.bot.username

    if update.message.reply_to_message:
        if update.message.reply_to_message.from_user.id == context.bot.id:
            return True, "reply"

    if not message_text:
        return False, None

    if message_text.startswith('/'):
        return True, "команда"

    if f"@{bot_username}" in message_text:
        return True, f"@{bot_username}"

    if re.search(rf'\b{re.escape(bot_real_name)}\b', message_text, re.IGNORECASE):
        return True, bot_real_name

    for name in BOT_NAMES:
        if re.search(rf'\b{re.escape(name)}\b', message_text, re.IGNORECASE):
            return True, name

    return False, None

async def download_media_as_base64(file_id: str, context: ContextTypes.DEFAULT_TYPE,
                                   return_bytes: bool = False) -> tuple:
    """
    Скачивает медиа-файл из Telegram.
    return_bytes=False (по умолчанию) → (base64_str, mime)
    return_bytes=True  → (raw_bytes, mime)
    """
    file = await context.bot.get_file(file_id)
    path = file.file_path.lower()

    if path.endswith(('.jpg', '.jpeg')):
        mime = "image/jpeg"
    elif path.endswith('.png'):
        mime = "image/png"
    elif path.endswith('.webp'):
        mime = "image/webp"
    elif path.endswith('.gif'):
        mime = "image/gif"
    else:
        mime = "image/jpeg" 

    buf = bytearray()
    await file.download_as_bytearray(buf)
    raw = bytes(buf)
    if return_bytes:
        return raw, mime
    b64 = base64.b64encode(raw).decode('utf-8')
    return b64, mime

def escape_user_text(text: str) -> str:
    return text.replace('[', '(').replace(']', ')')


def _get_ffmpeg_path() -> str:
    """Находит путь к ffmpeg: сначала пробует системный, потом из imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _ffprobe_duration(ffmpeg_path: str, video_path: str) -> float:
    """
    Получает длительность видео через ffmpeg (без ffprobe — он не всегда есть).
    Возвращает 0.0, если не удалось определить.
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-i", video_path],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        # ffmpeg пишет инфо в stderr, ищем "Duration: HH:MM:SS.xx"
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception as e:
        logger.error(f"ffprobe duration error: {e}")
    return 0.0


def _extract_frames_sync(video_path: str, num_frames: int, max_side: int = 1024) -> list[str]:
    """
    Синхронно извлекает num_frames равномерно распределённых кадров из видео,
    масштабирует их так, чтобы большая сторона была <= max_side, и возвращает list base64-jpeg.
    """
    ffmpeg_path = _get_ffmpeg_path()
    duration = _ffprobe_duration(ffmpeg_path, video_path)
    if duration <= 0:
        logger.warning("Не удалось определить длительность видео")
        return []

    frames_b64: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Берём кадры в моменты 1/(N+1), 2/(N+1), ..., N/(N+1) — без самого первого/последнего
        for i in range(num_frames):
            t = duration * (i + 1) / (num_frames + 1)
            out_path = os.path.join(tmpdir, f"frame_{i:02d}.jpg")
            try:
                subprocess.run(
                    [
                        ffmpeg_path, "-y",
                        "-ss", f"{t:.2f}",
                        "-i", video_path,
                        "-frames:v", "1",
                        "-vf", f"scale='min({max_side},iw)':'-2'",
                        "-q:v", "4",
                        out_path,
                    ],
                    capture_output=True, check=True
                )
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    frames_b64.append(f"data:image/jpeg;base64,{b64}")
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg frame extract failed at t={t:.2f}: {e.stderr[:200] if e.stderr else e}")
                continue

    return frames_b64


async def extract_video_frames(file_id: str, context: ContextTypes.DEFAULT_TYPE,
                               num_frames: int, max_duration_sec: float) -> tuple[list[str], float]:
    """
    Скачивает видео из Telegram, проверяет длительность, выдёргивает кадры.
    Возвращает (frames_data_urls, duration_sec). Если duration > max_duration_sec,
    возвращает ([], duration) — вызывающий должен решить, как реагировать.
    """
    file = await context.bot.get_file(file_id)

    suffix = ".mp4"
    if file.file_path:
        ext = os.path.splitext(file.file_path)[1].lower()
        if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
            suffix = ext

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        await file.download_to_drive(tmp_path)

        ffmpeg_path = _get_ffmpeg_path()
        duration = await asyncio.to_thread(_ffprobe_duration, ffmpeg_path, tmp_path)

        if duration <= 0:
            logger.warning("Длительность видео не определена, продолжаем (берём кадры всё равно)")
        elif duration > max_duration_sec:
            logger.info(f"Видео слишком длинное: {duration:.1f}с > {max_duration_sec}с")
            return [], duration

        frames = await asyncio.to_thread(_extract_frames_sync, tmp_path, num_frames)
        return frames, duration
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def download_video_to_file(file_id: str, context: ContextTypes.DEFAULT_TYPE,
                                 max_duration_sec: float) -> tuple[str | None, str, float]:
    """
    Скачивает видео из Telegram во временный файл, проверяет длительность.
    Возвращает (file_path, mime, duration_sec).
    Если длительность превышает лимит — возвращает (None, mime, duration), временный файл удаляет.
    """
    file = await context.bot.get_file(file_id)

    suffix = ".mp4"
    mime = "video/mp4"
    if file.file_path:
        ext = os.path.splitext(file.file_path)[1].lower()
        if ext == ".mov":
            suffix, mime = ".mov", "video/quicktime"
        elif ext == ".webm":
            suffix, mime = ".webm", "video/webm"
        elif ext == ".mkv":
            suffix, mime = ".mkv", "video/x-matroska"
        elif ext == ".avi":
            suffix, mime = ".avi", "video/x-msvideo"
        elif ext in (".mp4",):
            suffix, mime = ".mp4", "video/mp4"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        await file.download_to_drive(tmp_path)
        ffmpeg_path = _get_ffmpeg_path()
        duration = await asyncio.to_thread(_ffprobe_duration, ffmpeg_path, tmp_path)

        if duration > 0 and duration > max_duration_sec:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return None, mime, duration

        return tmp_path, mime, duration
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
