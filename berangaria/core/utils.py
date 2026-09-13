import re
import time
import random
import base64
import os
import tempfile
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional
from telegram import Update
from telegram.ext import ContextTypes
from berangaria.config import (
    BOT_NAMES,
    RANDOM_REPLY_COOLDOWN,
    RANDOM_REPLY_IDLE_TARGET_SECONDS,
    RANDOM_REPLY_PRESENCE_MULTIPLIER,
    RANDOM_REPLY_PRESENCE_SECONDS,
    RANDOM_REPLY_RECENT_WINDOW_SECONDS,
    BOT_TZ,
    SUMMARY_HOURS,
)
from berangaria.core.state import random_reply_cooldown
from berangaria.core import state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RandomReplyProbability:
    """Рассчитанный шанс и сигналы, которые на него повлияли."""

    chance: float
    gap_seconds: float | None
    recent_turns: int
    presence_multiplier: float
    presence_age_seconds: float | None

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


def strip_tiktok_urls_preserving_whitespace(text: str | None) -> str:
    """Удаляет TikTok-ссылки, не реконструируя остальной доказательный текст."""
    if not text:
        return ""
    return _TIKTOK_URL_RE.sub("", text).strip()

# Односложные реплики, на которых нет смысла тратить ambient-LLM / Mem0 search
_TRIVIAL_USER_TEXTS = {
    "",
    "(сообщение без текста)",
    "сообщение без текста",
    "без текста",
    "ладно",
    "ок",
    "окей",
    "хорошо",
    "ясно",
    "понятно",
    "пон",
    "понял",
    "поняла",
    "норм",
    "нормально",
    "да",
    "нет",
    "угу",
    "ага",
    "лол",
    "кек",
    "бля",
    "блин",
    "хз",
    "не знаю",
    "без понятия",
    "ну",
    "типа",
    "как бы",
    "короче",
    "в общем",
    "вобщем",
    "мда",
    "мм",
    "ммм",
    "хм",
    "хмм",
    "жесть",
    "капец",
    "пофиг",
    "похуй",
    "согласен",
    "согласна",
    "спасибо",
    "пожалуйста",
    "+",
    "-",
    "жми",
    "ждём",
    "ждем",
}

_TRIVIAL_USER_TOKENS = {
    token
    for phrase in _TRIVIAL_USER_TEXTS
    for token in re.findall(r"\w+", phrase)
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
    normalized = re.sub(r"[^\w\s+-]", " ", (text or "").strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in _TRIVIAL_USER_TEXTS:
        return True
    words = normalized.split()
    if words and all(word in _TRIVIAL_USER_TOKENS for word in words):
        return True
    if is_url_only_text(text):
        return True
    alnum_chars = sum(1 for ch in normalized if ch.isalnum())
    return alnum_chars < min_alnum


def get_bot_real_name(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Возвращает имя бота из Telegram."""
    return context.bot.first_name

def calculate_random_reply_probability(
    history: list[dict],
    *,
    current_created_at: float | None,
    base_chance: int,
    presence_age_seconds: float | None = None,
) -> RandomReplyProbability:
    """Рассчитывает ambient-шанс по паузе и плотности недавнего разговора.

    Формула сохраняет ``base_chance`` как операторскую базу::

        P = base * idle_factor * presence_factor / (1 + 0.5 * recent_turns)
        idle_factor = 0.1 + 2.9 * min(gap / idle_target, 1)

    Быстрый человеческий диалог одновременно получает маленький idle_factor и
    штраф за число недавних реплик. После длинной тишины recent_turns равен нулю,
    а шанс доходит до 3x базы. После успешного ответа на пинг presence_factor
    начинается с настроенного boost и линейно затухает до 1. Для 0 и 100
    сохраняется явный смысл команды ``/random``: всегда выключено либо всегда
    включено (после остальных gates).
    """
    try:
        presence_age = (
            float(presence_age_seconds)
            if presence_age_seconds is not None
            else None
        )
    except (TypeError, ValueError):
        presence_age = None
    if presence_age is not None and (
        not math.isfinite(presence_age) or presence_age < 0
    ):
        presence_age = None

    presence_multiplier = 1.0
    if (
        presence_age is not None
        and RANDOM_REPLY_PRESENCE_SECONDS > 0
        and presence_age < RANDOM_REPLY_PRESENCE_SECONDS
    ):
        remaining = 1.0 - presence_age / RANDOM_REPLY_PRESENCE_SECONDS
        presence_multiplier += (
            RANDOM_REPLY_PRESENCE_MULTIPLIER - 1.0
        ) * remaining

    if base_chance <= 0:
        return RandomReplyProbability(
            0.0,
            None,
            0,
            presence_multiplier,
            presence_age,
        )

    try:
        current = float(current_created_at) if current_created_at is not None else None
    except (TypeError, ValueError):
        current = None
    if current is not None and not math.isfinite(current):
        current = None

    prior_times: list[float] = []
    if current is not None:
        for entry in history:
            if entry.get("role") != "user":
                continue
            try:
                created_at = float(entry.get("created_at"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(created_at) and created_at <= current:
                prior_times.append(created_at)

    gap_seconds = None
    recent_turns = 0
    if prior_times:
        gap_seconds = max(0.0, current - max(prior_times))
        recent_turns = sum(
            0.0 <= current - created_at <= RANDOM_REPLY_RECENT_WINDOW_SECONDS
            for created_at in prior_times
        )

    if base_chance >= 100:
        chance = 100.0
    elif gap_seconds is None:
        # Legacy history may not have timestamps. Unknown activity must not be
        # mistaken for a ten-minute silence, so fall back to the configured base.
        chance = min(100.0, base_chance * presence_multiplier)
    else:
        idle_progress = min(gap_seconds / RANDOM_REPLY_IDLE_TARGET_SECONDS, 1.0)
        idle_factor = 0.1 + 2.9 * idle_progress
        density_factor = 1.0 / (1.0 + 0.5 * recent_turns)
        chance = min(
            100.0,
            base_chance
            * idle_factor
            * density_factor
            * presence_multiplier,
        )

    return RandomReplyProbability(
        chance,
        gap_seconds,
        recent_turns,
        presence_multiplier,
        presence_age,
    )


def should_reply_randomly(
    chat_id: int,
    activity_token: int | None,
    history: list[dict],
    current_created_at: float | None,
) -> bool:
    """Решает до LLM-вызова, заслуживает ли последняя реплика ambient-ответа."""
    if not state.is_latest_group_activity(chat_id, activity_token):
        logger.debug(
            "🤫 Ambient пропущен: после кандидата появилась новая активность (chat=%s)",
            chat_id,
        )
        return False

    last_reply = random_reply_cooldown.get(chat_id, 0)
    current_time = time.monotonic()
    if current_time - last_reply < RANDOM_REPLY_COOLDOWN:
        return False

    probability = calculate_random_reply_probability(
        history,
        current_created_at=current_created_at,
        base_chance=state.random_reply_chance,
        presence_age_seconds=state.get_bot_presence_age(chat_id, now=current_time),
    )
    selected = random.random() * 100 < probability.chance
    logger.debug(
        "🎲 Ambient-фильтр: chance=%.2f%% base=%s%% gap=%s recent=%s "
        "presence=%.2fx selected=%s chat=%s",
        probability.chance,
        state.random_reply_chance,
        "unknown" if probability.gap_seconds is None else f"{probability.gap_seconds:.1f}s",
        probability.recent_turns,
        probability.presence_multiplier,
        selected,
        chat_id,
    )
    if selected:
        random_reply_cooldown[chat_id] = current_time
        return True
    return False

def is_bot_mentioned(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    additional_text: str | None = None,
) -> Tuple[bool, Optional[str]]:
    """Проверяет обращение к боту в Telegram-тексте и дополнительном тексте.

    ``additional_text`` нужен для уже распознанной речи: транскрипт участвует
    только в выборе отвечать или молчать и не становится пользовательским
    текстом либо источником долговременной памяти.
    """
    if update.message is None:
        return False, None

    message_text = update.message.text or update.message.caption or ""
    bot_real_name = get_bot_real_name(context)
    bot_username = context.bot.username

    if update.message.reply_to_message:
        if update.message.reply_to_message.from_user.id == context.bot.id:
            return True, "reply"

    if not message_text and not additional_text:
        return False, None

    if message_text.startswith('/'):
        return True, "команда"

    searchable_texts = [message_text]
    if additional_text:
        searchable_texts.append(additional_text)

    for searchable_text in searchable_texts:
        if bot_username and f"@{bot_username}" in searchable_text:
            return True, f"@{bot_username}"

        if bot_real_name and re.search(
            rf'\b{re.escape(bot_real_name)}\b', searchable_text, re.IGNORECASE
        ):
            return True, bot_real_name

        for name in BOT_NAMES:
            if re.search(
                rf'\b{re.escape(name)}\b', searchable_text, re.IGNORECASE
            ):
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

    Квадратная скобка — единственный разделитель структуры промпта
    (`[Message: ...]`, `[Context from memory: ...]`, `[#N]`), поэтому в тексте
    пользователя не может остаться ни одной. Совпадение по шаблону `[Тег: ...]`
    здесь недостаточно: вызывающий код сам дописывает закрывающую скобку
    (`chat/handlers.py`), так что одиночная `]` внутри текста закрывает наш тег и
    делает следующий за ней блок пользователя неотличимым от служебного.
    """
    if not text:
        return ''

    return text.replace('[', '(').replace(']', ')')


def strip_internal_tags(text: str) -> str:
    """Убирает служебные токены модели из текста, предназначенного пользователю.

    Живёт здесь, а не в llm_client, чтобы streaming мог применять то же правило
    к превью: llm_client импортирует streaming, и обратный импорт дал бы цикл.
    Содержит только вычистку тегов — срез финальной точки и правило молчания
    относятся к готовому ответу и остаются в llm_client._clean_reply.
    """
    if not text:
        return ''

    text = re.sub(r'<\|channel\>.*?<channel\|>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<\|.*?\|>', '', text).strip()

    text = re.sub(
        r'\[Context from memory(?:\s*:[^\]]*)?\]',
        'долгосрочной памяти',
        text,
        flags=re.IGNORECASE,
    )

    # [#N] — внутренние reply-хэндлы для инструментов. Модель иногда всё же
    # цитирует их вопреки системному промпту, поэтому не выпускаем их в Telegram.
    text = re.sub(r'\[#\d+\](?:\s*(?:,|и|или)\s*\[#\d+\])*', '', text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'[,;:]+([.!?])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


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
