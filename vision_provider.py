import re
import os
import asyncio
import base64
import logging
import httpx

from config import (
    GEMINI_MODEL, GEMINI_API_KEY, DEBUG,
    GEMINI_UPLOAD_MAX_WAIT_SEC, GEMINI_UPLOAD_BACKOFF_INITIAL, GEMINI_UPLOAD_BACKOFF_MAX
)

logger = logging.getLogger(__name__)


def _log_description(prefix: str, description: str, meta: str = "") -> None:
    """
    Logs vision model results.
    DEBUG=True  → full description text.
    DEBUG=False → metadata only (model, tokens, length).
    """
    head = f"{prefix} {meta}".rstrip()
    # Обрезаем длинные описания для INFO режима (80 символов), в DEBUG будет полное
    short_desc = description[:80] + "..." if len(description) > 80 else description
    logger.info(f"{head} | {len(description)} chars | {short_desc}")
    
    if DEBUG:
        logger.debug(f"Полное описание:\n{description}")


# ========================== PROMPTS ==========================

_IMAGE_PROMPT = (
    "Опиши это изображение так, как ты рассказал бы другу в переписке. "
    "Твой ответ будет передан в текстовый чат, где тебя не видят — только читают твои слова.\n\n"
    
    "Что упомянуть обязательно:\n"
    "• Кто/что в кадре — люди (внешность, одежда, эмоции), объекты, обстановка\n"
    "• Текст и логотипы (процитируй дословно, если видишь надписи)\n"
    "• Узнаваемые персонажи, мемы, бренды — назови конкретно (например: «Геральт из The Witcher», "
    "«мем Distracted Boyfriend», «логотип NVIDIA»). Если не уверен — скажи «похоже на...\". "
    "Если не узнаёшь — честно напиши, что не опознал\n"
    "• Настроение, атмосферу, цветовую гамму\n\n"
    
    "Важно:\n"
    "- Говори естественно, без заголовков и пунктов — как в сообщении мессенджера\n"
    "- Не начинай с «На картинке...» или «Изображение показывает...» — сразу к делу\n"
    "- Если что-то смешное или странное — можешь добавить лёгкий комментарий\n\n"
    
    "Пиши на русском языке."
)


def _video_prompt(duration: float) -> str:
    intro = "Это видео"
    if duration > 0:
        intro += f" длиной {duration:.0f} секунд"
    intro += ". Опиши его так, как рассказал бы другу в переписке — твой ответ уйдёт в текстовый чат."

    return (
        f"{intro}\n\n"
        "Что упомянуть обязательно:\n"
        "• Кто/что в кадре, как выглядят, что делают\n"
        "• Текст и логотипы (процитируй, если видишь надписи)\n"
        "• Узнаваемые персонажи, мемы, бренды, игры, фильмы — назови конкретно. "
        "Если не уверен — «похоже на...\". Если не узнал — так и скажи\n"
        "• Как меняется сцена: начало → развитие → концовка\n"
        "• Настроение, музыка (если слышна), общая атмосфера\n\n"
        
        "Важно:\n"
        "- Рассказывай естественно, без заголовков и разделов — как в сообщении мессенджера\n"
        "- Не начинай с «На видео...» или «В кадрах...» — сразу к делу\n"
        "- Если что-то смешное или странное — можешь прокомментировать\n\n"
        
        "Пиши на русском языке."
    )

# ========================== GEMINI API ==========================

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_INLINE_MAX_BYTES = 18 * 1024 * 1024  # ~18MB inline limit with buffer


def _gemini_extract_text(resp_json: dict) -> str:
    """Extract text from Gemini response, handling prompt feedback / safety blocks."""
    candidates = resp_json.get("candidates") or []
    if not candidates:
        feedback = resp_json.get("promptFeedback") or {}
        block_reason = feedback.get("blockReason")
        if block_reason:
            logger.warning(f"Gemini blocked request: {block_reason}")
            return ""
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    return "".join(text_parts).strip()


async def _gemini_describe_image(image_bytes: bytes, mime: str, caption: str = "") -> str:
    """Describe image using Gemini API."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set in .env")
        return ""

    user_text = _IMAGE_PROMPT
    if caption:
        user_text += (
            f"\n\nПользователь добавил подпись к картинке: «{caption}»\n"
            "Учитывай её при описании — проверь, совпадает ли подпись с тем, что видишь, "
            "и используй как подсказку для узнавания персонажей."
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
                logger.error(f"Gemini image API {r.status_code}: {r.text[:300]}")
                return ""
            data = r.json()
            description = _gemini_extract_text(data)
            usage = data.get("usageMetadata") or {}
            
            # Логирование с превью для INFO, полное для DEBUG
            meta = (
                f"tokens prompt={usage.get('promptTokenCount', '?')}, "
                f"out={usage.get('candidatesTokenCount', '?')}"
            )
            _log_description(f"🖼️ [Gemini:{GEMINI_MODEL}]", description, meta=meta)
            
            return description
    except Exception as e:
        logger.error(f"Gemini image error: {e}")
        return ""


async def _gemini_upload_file(file_path: str, mime: str) -> str | None:
    """
    Заливает файл в Gemini Files API через resumable upload.
    Возвращает file_uri вида files/abc123 или None при ошибке.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set in .env")
        return None
    
    if not os.path.exists(file_path):
        logger.error(f"File not found for upload: {file_path}")
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
                logger.error(f"Gemini upload: нет upload URL в ответе")
                return None
        except Exception as e:
            logger.error(f"Gemini upload start: {e}")
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
                logger.error(f"Gemini upload finalize: нет uri в ответе: {data}")
                return None
            logger.info(f"📤 Gemini Files: загружено {file_name_id} ({file_size} байт)")

            # 3) Дождаться, пока файл станет ACTIVE (для видео идёт препроцессинг)
            poll_url = f"{GEMINI_API_BASE}/{file_name_id}"
            max_wait_time = GEMINI_UPLOAD_MAX_WAIT_SEC
            wait_time = 0
            backoff = GEMINI_UPLOAD_BACKOFF_INITIAL
            
            while wait_time < max_wait_time:
                pr = await client.get(poll_url, headers={"x-goog-api-key": GEMINI_API_KEY})
                if pr.status_code != 200:
                    logger.warning(f"Gemini poll {pr.status_code}: {pr.text[:200]}")
                    await asyncio.sleep(backoff)
                    wait_time += backoff
                    backoff = min(backoff * 1.5, GEMINI_UPLOAD_BACKOFF_MAX)
                    continue
                
                state = pr.json().get("state", "")
                if state == "ACTIVE":
                    return file_uri
                if state == "FAILED":
                    logger.error(f"Gemini обработка файла FAILED: {pr.json()}")
                    return None
                
                await asyncio.sleep(backoff)
                wait_time += backoff
                backoff = min(backoff * 1.5, GEMINI_UPLOAD_BACKOFF_MAX)

            logger.error(f"Gemini upload: файл не стал ACTIVE за {max_wait_time} сек")
            return None
        except Exception as e:
            logger.error(f"Gemini upload finalize: {e}")
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
    """Описывает видео через Gemini API. Не удаляет файл - это ответственность вызывающего кода."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY не задан в .env")
        return ""

    if not os.path.exists(video_path):
        logger.error(f"Файл видео не найден: {video_path}")
        return ""

    file_size = os.path.getsize(video_path)
    user_text = _video_prompt(duration)
    if caption:
        user_text += (
            f"\n\nПользователь добавил подпись к видео: «{caption}»\n"
            "Учитывай её при описании — используй как подсказку для узнавания и контекста."
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
                logger.error(f"Gemini video API {r.status_code}: {r.text[:300]}")
                return ""
            data = r.json()
            description = _gemini_extract_text(data)
            usage = data.get("usageMetadata") or {}
            _log_description(
                f"🎬 [Gemini:{GEMINI_MODEL}, {file_size} байт]",
                description,
                meta=(
                    f"tokens prompt={usage.get('promptTokenCount', '?')}, "
                    f"out={usage.get('candidatesTokenCount', '?')}"
                ),
            )
            return description
    except Exception as e:
        logger.error(f"Gemini video error: {e}")
        return ""
        return ""
    finally:
        # Подчищаем за собой загруженный файл из Gemini Files API
        if file_uri:
            asyncio.create_task(_gemini_delete_file(file_uri))


_AUDIO_PROMPT = (
    "Это голосовое или аудио сообщение из чата. "
    "Транскрибируй РЕЧЬ дословно на языке оригинала — так, как человек это сказал.\n\n"
    
    "Верни ТОЛЬКО текст сказанного:\n"
    "- Без комментариев, без кавычек, без префиксов «Человек сказал:»\n"
    "- Если речи нет (музыка, шум, тишина) — кратко опиши, что слышно, в скобках\n"
    "- Мат, сленг, оговорки — всё как есть, дословно\n"
)


async def _gemini_transcribe_audio(audio_path: str, mime: str, caption: str = "") -> str:
    """Транскрибирует аудио через Gemini. Файл НЕ удаляет — это делает вызывающий код."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY не задан в .env")
        return ""

    if not os.path.exists(audio_path):
        logger.error(f"Аудиофайл не найден: {audio_path}")
        return ""

    file_size = os.path.getsize(audio_path)
    user_text = _AUDIO_PROMPT
    if caption:
        user_text += f"\n\nПользователь добавил подпись: «{caption}»"

    file_uri: str | None = None

    # Маленькие файлы — inline, крупные — через Files API
    if file_size <= GEMINI_INLINE_MAX_BYTES:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        parts = [
            {"text": user_text},
            {"inline_data": {"mime_type": mime, "data": b64}},
        ]
    else:
        file_uri = await _gemini_upload_file(audio_path, mime)
        if not file_uri:
            return ""
        parts = [
            {"text": user_text},
            {"file_data": {"mime_type": mime, "file_uri": file_uri}},
        ]

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        },
    }

    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"Gemini audio API {r.status_code}: {r.text[:300]}")
                return ""
            data = r.json()
            transcript = _gemini_extract_text(data)
            _log_description(f"🎤 [Gemini:{GEMINI_MODEL}, {file_size} байт]", transcript)
            return transcript
    except Exception as e:
        logger.error(f"Gemini audio error: {e}")
        return ""
    finally:
        if file_uri:
            asyncio.create_task(_gemini_delete_file(file_uri))


# ========================== ПУБЛИЧНЫЙ API ==========================

async def describe_image_bytes(image_bytes: bytes, mime: str, caption: str = "") -> str:
    """Возвращает текстовое описание картинки через Gemini."""
    return await _gemini_describe_image(image_bytes, mime, caption)


async def describe_video(
    *,
    video_path: str,
    mime: str = "video/mp4",
    caption: str = "",
    duration: float = 0.0,
) -> str:
    """
    Возвращает текстовое описание видео через Gemini (видео передаётся целиком).
    
    ВАЖНО: Удаление временного файла происходит в finally блоке этой функции.
    """
    if not video_path:
        logger.error("video_path обязателен")
        return ""
    
    try:
        return await _gemini_describe_video(video_path, mime, caption, duration)
    finally:
        # Удаляем временный файл после обработки (или при ошибке)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.debug(f"🗑️ Удалён временный файл: {video_path}")
        except OSError as e:
            logger.debug(f"⚠️ Не удалось удалить временный файл {video_path}: {e}")


async def transcribe_audio(
    *,
    audio_path: str,
    mime: str = "audio/ogg",
    caption: str = "",
) -> str:
    """
    Возвращает транскрипцию голосового/аудио сообщения через Gemini.

    ВАЖНО: Удаление временного файла происходит в finally блоке этой функции.
    """
    if not audio_path:
        logger.error("audio_path обязателен")
        return ""

    try:
        return await _gemini_transcribe_audio(audio_path, mime, caption)
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"🗑️ Удалён временный аудиофайл: {audio_path}")
        except OSError as e:
            logger.debug(f"⚠️ Не удалось удалить временный аудиофайл {audio_path}: {e}")
