"""
Vision-провайдеры: LM Studio (локальная Qwen3-VL) и Google Gemini API.

Унифицированный интерфейс:
    - describe_image_bytes(image_bytes, mime, caption) -> str
    - describe_video_path(video_path, caption) -> str

Выбор провайдера — по config.VISION_PROVIDER.
"""

import re
import os
import asyncio
import base64
import logging
import httpx

from config import (
    LM_STUDIO_URL, VISION_MODEL, VISION_PROVIDER,
    GEMINI_MODEL, GEMINI_API_KEY,
    VIDEO_MAX_FRAMES, VIDEO_MAX_DURATION_SEC,
)

logger = logging.getLogger(__name__)


# ========================== ОБЩИЕ ПРОМПТЫ ==========================

_IMAGE_PROMPT = (
    "Проанализируй изображение в три шага и оформи ответ строго по этим разделам.\n\n"
    "1. ДЕТАЛИ: подробно перечисли всё важное, что видишь — людей и их внешность/одежду/позы, "
    "объекты, текст и логотипы (процитируй дословно), цвета, фон, обстановку, освещение. "
    "Не сокращай, не обобщай.\n\n"
    "2. УЗНАВАНИЕ: попробуй опознать конкретных персонажей, людей, мемы, бренды, локации, "
    "произведения. Если узнал — назови имя и франшизу/источник (например: «Геральт из The Witcher», "
    "«мем Distracted Boyfriend», «логотип NVIDIA»). Если не уверен — пиши «возможно, это …» с "
    "альтернативами. Если ничего не узнаёшь — честно напиши «не узнаю конкретных персонажей/брендов».\n\n"
    "3. ИТОГ: 2-3 предложения для разговорного пересказа — суть и настроение картинки. "
    "Без вступительных фраз вроде 'на картинке'.\n\n"
    "Отвечай на русском языке."
)


def _video_prompt(duration: float, num_frames: int = 0, native_video: bool = False) -> str:
    if native_video:
        intro = "Тебе дано видео"
        if duration > 0:
            intro += f" длительностью около {duration:.0f} сек"
        intro += "."
    else:
        intro = f"Тебе дано {num_frames} кадров из видео"
        if duration > 0:
            intro += f" длительностью около {duration:.0f} сек"
        intro += ", расположенных по порядку от начала к концу."

    return (
        f"{intro}\n\n"
        "Проанализируй видео в три шага и оформи ответ строго по этим разделам.\n\n"
        "1. ДЕТАЛИ: подробно перечисли что видишь — людей и их внешность/одежду, "
        "объекты, текст и логотипы (процитируй дословно), обстановку, ключевые действия и "
        "как сцена меняется со временем.\n\n"
        "2. УЗНАВАНИЕ: попробуй опознать конкретных персонажей, людей, мемы, бренды, локации, "
        "произведения, игры, фильмы. Если узнал — назови имя и франшизу/источник. Если не уверен — "
        "пиши «возможно, это …» с альтернативами. Если ничего не узнаёшь — честно скажи об этом.\n\n"
        "3. ИТОГ: 3-4 предложения для разговорного пересказа — что происходит и какое настроение. "
        "Без вступительных фраз вроде 'на видео' или 'в кадрах'.\n\n"
        "Отвечай на русском языке."
    )


def _strip_thinking(text: str) -> str:
    """Убирает <think>...</think> блоки и стоп-токены."""
    if not text:
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in text and '</think>' not in text:
        return ""
    text = re.sub(r'<\|.*?\|>', '', text)
    return text.strip()


# ========================== LM STUDIO ==========================

async def _lmstudio_describe_image(image_bytes: bytes, mime: str, caption: str = "") -> str:
    if not VISION_MODEL:
        logger.warning("⚠️ VISION_MODEL не задана — пропускаю описание картинки")
        return ""

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime};base64,{b64}"

    user_text = _IMAGE_PROMPT
    if caption:
        user_text += (
            f"\n\nПодпись пользователя к картинке: «{caption}». "
            "Используй её как подсказку — проверь, согласуется ли подпись с тем, что видишь, "
            "и при узнавании персонажей учитывай её контекст."
        )

    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        }],
        "max_tokens": 5000,
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 20,
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(LM_STUDIO_URL, json=payload)
            r.raise_for_status()
            data = r.json()
            description = _strip_thinking(data['choices'][0]['message']['content'] or "")
            usage = data.get('usage') or {}
            logger.info(
                f"🖼️ [LMStudio:{VISION_MODEL}] completion={usage.get('completion_tokens', '?')} | "
                f"{description[:160]}..."
            )
            return description
    except Exception as e:
        logger.error(f"❌ LMStudio image error: {e}")
        return ""


async def _lmstudio_describe_video_frames(frames_data_urls: list[str], caption: str, duration: float) -> str:
    if not VISION_MODEL:
        logger.warning("⚠️ VISION_MODEL не задана — пропускаю описание видео")
        return ""
    if not frames_data_urls:
        return ""

    user_text = _video_prompt(duration, num_frames=len(frames_data_urls), native_video=False)
    if caption:
        user_text += (
            f"\n\nПодпись пользователя к видео: «{caption}». "
            "Используй её как подсказку и при узнавании учитывай контекст."
        )

    content = [{"type": "text", "text": user_text}]
    for url in frames_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4500,
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 20,
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(LM_STUDIO_URL, json=payload)
            r.raise_for_status()
            data = r.json()
            description = _strip_thinking(data['choices'][0]['message']['content'] or "")
            usage = data.get('usage') or {}
            logger.info(
                f"🎬 [LMStudio:{VISION_MODEL}, {len(frames_data_urls)} кадров] "
                f"completion={usage.get('completion_tokens', '?')} | {description[:160]}..."
            )
            return description
    except Exception as e:
        logger.error(f"❌ LMStudio video error: {e}")
        return ""


# ========================== GEMINI ==========================

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Inline-данные у Gemini ограничены ~20МБ на запрос.
GEMINI_INLINE_MAX_BYTES = 18 * 1024 * 1024  # с запасом


def _gemini_extract_text(resp_json: dict) -> str:
    """Достаёт текст из ответа Gemini, обрабатывая prompt feedback / safety blocks."""
    candidates = resp_json.get("candidates") or []
    if not candidates:
        feedback = resp_json.get("promptFeedback") or {}
        block_reason = feedback.get("blockReason")
        if block_reason:
            logger.warning(f"⚠️ Gemini заблокировал запрос: {block_reason}")
            return ""
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    return "".join(text_parts).strip()


async def _gemini_describe_image(image_bytes: bytes, mime: str, caption: str = "") -> str:
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY не задан в .env")
        return ""

    user_text = _IMAGE_PROMPT
    if caption:
        user_text += (
            f"\n\nПодпись пользователя к картинке: «{caption}». "
            "Используй её как подсказку — проверь, согласуется ли подпись с тем, что видишь, "
            "и при узнавании персонажей учитывай её контекст."
        )

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_text},
                {"inline_data": {"mime_type": mime, "data": b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        },
    }

    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"❌ Gemini image API {r.status_code}: {r.text[:300]}")
                return ""
            data = r.json()
            description = _gemini_extract_text(data)
            usage = data.get("usageMetadata") or {}
            logger.info(
                f"🖼️ [Gemini:{GEMINI_MODEL}] tokens="
                f"prompt={usage.get('promptTokenCount', '?')}, "
                f"out={usage.get('candidatesTokenCount', '?')} | {description[:160]}..."
            )
            return description
    except Exception as e:
        logger.error(f"❌ Gemini image error: {e}")
        return ""


async def _gemini_upload_file(file_path: str, mime: str) -> str | None:
    """
    Заливает файл в Gemini Files API через resumable upload.
    Возвращает file_uri вида files/abc123 или None при ошибке.
    """
    if not GEMINI_API_KEY:
        return None

    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)

    # 1) Старт сессии загрузки
    start_url = f"{GEMINI_API_BASE}/files"
    start_headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json",
    }
    start_body = {"file": {"display_name": file_name}}

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            r = await client.post(start_url, headers=start_headers, json=start_body)
            r.raise_for_status()
            upload_url = r.headers.get("X-Goog-Upload-URL") or r.headers.get("x-goog-upload-url")
            if not upload_url:
                logger.error(f"❌ Gemini upload: нет upload URL в ответе ({dict(r.headers)})")
                return None
        except Exception as e:
            logger.error(f"❌ Gemini upload start: {e}")
            return None

        # 2) Загрузка содержимого
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            up_headers = {
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            }
            r2 = await client.post(upload_url, headers=up_headers, content=file_bytes)
            r2.raise_for_status()
            data = r2.json()
            file_info = data.get("file") or {}
            file_uri = file_info.get("uri")
            file_name_id = file_info.get("name")  # files/abc123
            if not file_uri:
                logger.error(f"❌ Gemini upload finalize: нет uri в ответе: {data}")
                return None
            logger.info(f"📤 Gemini Files: загружено {file_name_id} ({file_size} байт)")

            # 3) Дождаться, пока файл станет ACTIVE (для видео идёт препроцессинг)
            poll_url = f"{GEMINI_API_BASE}/{file_name_id}"
            for _ in range(60):  # до 60 секунд
                pr = await client.get(poll_url, headers={"x-goog-api-key": GEMINI_API_KEY})
                if pr.status_code != 200:
                    logger.warning(f"Gemini poll {pr.status_code}: {pr.text[:200]}")
                    await asyncio.sleep(1)
                    continue
                state = pr.json().get("state", "")
                if state == "ACTIVE":
                    return file_uri
                if state == "FAILED":
                    logger.error(f"❌ Gemini обработка файла FAILED: {pr.json()}")
                    return None
                await asyncio.sleep(1)

            logger.error("❌ Gemini upload: файл не стал ACTIVE за 60 сек")
            return None
        except Exception as e:
            logger.error(f"❌ Gemini upload finalize: {e}")
            return None


async def _gemini_delete_file(file_uri: str) -> None:
    """Best-effort удаление файла из Gemini Files API."""
    if not file_uri:
        return
    # file_uri имеет вид https://...../v1beta/files/abc — берём только files/abc
    name_match = re.search(r"(files/[^/?#]+)", file_uri)
    if not name_match:
        return
    name = name_match.group(1)
    url = f"{GEMINI_API_BASE}/{name}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.delete(url, headers={"x-goog-api-key": GEMINI_API_KEY})
    except Exception as e:
        logger.debug(f"Gemini delete file ignored: {e}")


async def _gemini_describe_video(video_path: str, mime: str, caption: str, duration: float) -> str:
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY не задан в .env")
        return ""

    file_size = os.path.getsize(video_path)
    user_text = _video_prompt(duration, native_video=True)
    if caption:
        user_text += (
            f"\n\nПодпись пользователя к видео: «{caption}». "
            "Используй её как подсказку и при узнавании учитывай контекст."
        )

    file_uri: str | None = None

    # Маленькие видео можно слать inline; крупные — через Files API
    if file_size <= GEMINI_INLINE_MAX_BYTES:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode("utf-8")
        parts = [
            {"text": user_text},
            {"inline_data": {"mime_type": mime, "data": b64}},
        ]
    else:
        file_uri = await _gemini_upload_file(video_path, mime)
        if not file_uri:
            return ""
        parts = [
            {"text": user_text},
            {"file_data": {"mime_type": mime, "file_uri": file_uri}},
        ]

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        },
    }

    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"❌ Gemini video API {r.status_code}: {r.text[:300]}")
                return ""
            data = r.json()
            description = _gemini_extract_text(data)
            usage = data.get("usageMetadata") or {}
            logger.info(
                f"🎬 [Gemini:{GEMINI_MODEL}, {file_size} байт] tokens="
                f"prompt={usage.get('promptTokenCount', '?')}, "
                f"out={usage.get('candidatesTokenCount', '?')} | {description[:160]}..."
            )
            return description
    except Exception as e:
        logger.error(f"❌ Gemini video error: {e}")
        return ""
    finally:
        # Подчищаем за собой загруженный файл
        if file_uri:
            asyncio.create_task(_gemini_delete_file(file_uri))


# ========================== ПУБЛИЧНЫЙ API ==========================

async def describe_image_bytes(image_bytes: bytes, mime: str, caption: str = "") -> str:
    """
    Возвращает текстовое описание картинки.
    Маршрутизирует на провайдер из config.VISION_PROVIDER.
    """
    if VISION_PROVIDER == "gemini":
        return await _gemini_describe_image(image_bytes, mime, caption)
    return await _lmstudio_describe_image(image_bytes, mime, caption)


async def describe_video(
    *,
    video_path: str | None = None,
    mime: str = "video/mp4",
    frames_data_urls: list[str] | None = None,
    caption: str = "",
    duration: float = 0.0,
) -> str:
    """
    Возвращает текстовое описание видео.
    - Для gemini: используется video_path (видео идёт целиком).
    - Для lmstudio: используется frames_data_urls (заранее извлечённые кадры).
    """
    if VISION_PROVIDER == "gemini":
        if not video_path:
            logger.error("❌ Gemini video: video_path обязателен")
            return ""
        return await _gemini_describe_video(video_path, mime, caption, duration)

    return await _lmstudio_describe_video_frames(frames_data_urls or [], caption, duration)


def is_native_video_supported() -> bool:
    """True если провайдер умеет принимать видео целиком (без извлечения кадров)."""
    return VISION_PROVIDER == "gemini"
