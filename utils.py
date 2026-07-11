import re
import time
import random
import base64
import os
import tempfile
import logging
from datetime import datetime
from typing import Tuple, Optional
from telegram import Update
from telegram.ext import ContextTypes
from config import BOT_NAMES, RANDOM_REPLY_COOLDOWN, BOT_TZ, SUMMARY_HOURS
from state import random_reply_cooldown
import state

logger = logging.getLogger(__name__)

# URL в тексте (для отсечения «голых» ссылок из ambient/Mem0)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# TikTok: модели не отдаём — только вырезаем, без плейсхолдеров
_TIKTOK_URL_RE = re.compile(
    r"(?i)(?:https?://)?(?:(?:www|vm|vt)\.)?tiktok\.com/[^\s<>\]\)\"']+"
)


def strip_tiktok_urls(text: str | None) -> str:
    """Удаляет TikTok-ссылки из текста. Ничего вместо них не подставляет."""
    if not text:
        return ""
    cleaned = _TIKTOK_URL_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

# Односложные реплики, на которых нет смысла тратить ambient-LLM / Mem0 search
_TRIVIAL_USER_TEXTS = {
    "",
    "(сообщение без текста)",
    "сообщение без текста",
    "без текста",
    "ладно",
    "ок",
    "окей",
    "пон",
    "норм",
    "да",
    "нет",
    "угу",
    "ага",
    "лол",
    "кек",
    "+",
    "-",
    "жми",
    "ждём",
    "ждем",
}


def now_local() -> datetime:
    """Текущее время в часовом поясе бота (по умолчанию Europe/Moscow)."""
    return datetime.now(BOT_TZ)


def next_summary_run(now: datetime | None = None) -> datetime:
    """Ближайший запуск автосуммаризации по SUMMARY_HOURS в timezone бота."""
    from datetime import timedelta

    now = now or now_local()
    if now.tzinfo is None:
        now = now.replace(tzinfo=BOT_TZ)
    candidates = []
    for hour in SUMMARY_HOURS:
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        candidates.append(target)
    return min(candidates)


def is_url_only_text(text: str | None) -> bool:
    """True, если текст — только URL(ы) без осмысленных слов."""
    if not text or not str(text).strip():
        return False
    remainder = _URL_RE.sub(" ", text)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    # остались только пунктуация/мусор — считаем URL-only
    alnum = sum(1 for ch in remainder if ch.isalnum())
    return alnum < 3


def is_low_signal_user_text(text: str | None, *, min_alnum: int = 12) -> bool:
    """
    Пустые/односложные/URL-only реплики: не гоняем ambient LLM и не ищем в Mem0.
    История при этом может писаться — контекст чата сохраняется.
    """
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if normalized in _TRIVIAL_USER_TEXTS:
        return True
    if is_url_only_text(text):
        return True
    alnum_chars = sum(1 for ch in normalized if ch.isalnum())
    return alnum_chars < min_alnum


def get_bot_real_name(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Возвращает имя бота из Telegram."""
    return context.bot.first_name

def should_reply_randomly(chat_id: int) -> bool:
    """Определяет, должен ли бот ответить случайно в группе."""
    last_reply = random_reply_cooldown.get(chat_id, 0)
    current_time = time.time()
    if current_time - last_reply < RANDOM_REPLY_COOLDOWN:
        return False
    if random.randint(1, 100) <= state.random_reply_chance:
        random_reply_cooldown[chat_id] = current_time
        return True
    return False

def is_bot_mentioned(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tuple[bool, Optional[str]]:
    """Проверяет, был ли упомянут бот в сообщении."""
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
                                   return_bytes: bool = False) -> Tuple[bytes | str, str]:
    """
    Скачивает медиа-файл из Telegram (используется для изображений).
    
    Args:
        file_id: ID файла в Telegram
        context: Контекст бота
        return_bytes: Если True, возвращает bytes, иначе base64 строку
    
    Returns:
        Tuple из (данные, mime_type)
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
    """
    Экранирует текст пользователя для безопасной вставки в промпт.
    Заменяет служебные скобки, которые могут быть восприняты как системные теги.
    """
    if not text:
        return ''
    
    # Список служебных тегов, которые могут конфликтовать с форматом промпта
    service_tags = [
        'User', 'Time', 'Message', 'Event', 'Reply to', 'Quoted message',
        'Image description', 'Video description', 'Audio description',
        'Context from memory', 'Forwarded from'
    ]
    
    # Экранируем только потенциально опасные паттерны
    for tag in service_tags:
        # Заменяем [Тег: значение] на (Тег: значение)
        text = re.sub(
            rf'\[{re.escape(tag)}:\s*(.*?)\]',
            r'(\1)',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

    # Нейтрализуем поддельные reply-хэндлы [#5] в тексте пользователя, чтобы их
    # нельзя было выдать за наш служебный тег. Наш настоящий [#N] добавляется отдельно.
    text = re.sub(r'\[#(\d+)\]', r'(#\1)', text)

    return text


def get_video_duration(video_obj) -> float:
    """
    Извлекает длительность видео из Telegram объекта.
    Поддерживает как int/float, так и timedelta (PTB v22.2+).
    """
    duration_raw = getattr(video_obj, "duration", None)
    if duration_raw is None:
        return 0.0
    if hasattr(duration_raw, "total_seconds"):
        return duration_raw.total_seconds()
    try:
        return float(duration_raw)
    except (ValueError, TypeError):
        return 0.0


async def download_video_to_file(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> Tuple[Optional[str], str, float]:
    """
    Скачивает видео из Telegram во временный файл для Gemini.
    
    Args:
        file_id: ID файла в Telegram
        context: Контекст бота
    
    Returns:
        Tuple из (путь_к_файлу, mime_type, длительность_сек).
        file_path может быть None при ошибке.
    
    ВАЖНО: Вызывающий код должен самостоятельно удалить временный файл после использования!
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

        # Длительность проверяется в handle_video через метаданные Telegram
        duration = 0.0

        return tmp_path, mime, duration
    except Exception:
        # Только при ошибке удаляем файл здесь
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


async def download_audio_to_file(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> Tuple[Optional[str], str]:
    """
    Скачивает аудио/голосовое из Telegram во временный файл для Gemini.

    Returns:
        Tuple из (путь_к_файлу, mime_type). file_path может быть None при ошибке.

    ВАЖНО: Вызывающий код должен удалить временный файл после использования!
    """
    file = await context.bot.get_file(file_id)

    # Голосовые Telegram приходят как .oga (OGG/opus)
    suffix, mime = ".oga", "audio/ogg"
    if file.file_path:
        ext = os.path.splitext(file.file_path)[1].lower()
        audio_map = {
            ".oga": ("audio/ogg"), ".ogg": ("audio/ogg"),
            ".mp3": ("audio/mp3"), ".m4a": ("audio/mp4"),
            ".aac": ("audio/aac"), ".wav": ("audio/wav"),
            ".flac": ("audio/flac"), ".opus": ("audio/ogg"),
        }
        if ext in audio_map:
            suffix, mime = ext, audio_map[ext]

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        await file.download_to_drive(tmp_path)
        return tmp_path, mime
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise