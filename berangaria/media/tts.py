"""Fish Audio text-to-speech for Telegram voice notes.

Secrets and readiness live in berangaria.config (FISH_API_KEY, FISH_VOICE_ID,
TTS_*). Call synthesize_speech only from a worker thread (asyncio.to_thread) —
this module uses blocking httpx.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import httpx

from berangaria.config import (
    FISH_API_KEY,
    FISH_VOICE_ID,
    TTS_ENABLED,
    TTS_FORMAT,
    TTS_LATENCY,
    TTS_MAX_CHARS,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
    TTS_TIMEOUT_SECONDS,
    TTS_DEFAULT_EMOTION,
)

logger = logging.getLogger(__name__)

API_URL = "https://api.fish.audio/v1/tts"

# Whitelist only — free-form tags would let the model drive uncanny acting.
# Values are S2 bracket cues (see Fish Emotion Control docs).
VOICE_EMOTIONS: dict[str, str] = {
    "calm": "[calm]",
    "sarcastic": "[sarcastic]",
    "disdainful": "[disdainful]",
    "bored": "[bored]",
    "indifferent": "[indifferent]",
    "confident": "[confident]",
    "sighing": "[sighing]",
    "chuckling": "[chuckling]",
}

# Model-injected control brackets must never reach Fish as free-form stage directions.
_CONTROL_BRACKET_RE = re.compile(r"\[[^\]\n]{1,80}\]")


class TTSError(RuntimeError):
    """Synthesis failed in a way the tool layer should surface to the model."""


def is_tts_ready() -> bool:
    """True when config wants TTS and secrets + voice id are present."""
    return bool(TTS_ENABLED and FISH_API_KEY and FISH_VOICE_ID)


def normalize_emotion(emotion: object) -> Optional[str]:
    """Map untrusted tool arg → whitelist key.

    Returns:
        whitelist key, or None when the caller asked for no tag
        (\"none\") / invalid empty. Omitted emotion is handled separately
        via resolve_emotion_key (applies TTS_DEFAULT_EMOTION).
    """
    if emotion is None:
        return None
    if not isinstance(emotion, str):
        return None
    key = emotion.strip().lower()
    if not key or key in {"none", "null", "-"}:
        return None
    if key in VOICE_EMOTIONS:
        return key
    return None


def resolve_emotion_key(emotion: object = ..., *, apply_default: bool = True) -> Optional[str]:
    """Pick the emotion key actually used for synthesis / history.

    - Ellipsis / missing-style call with apply_default: config default.
    - Explicit \"none\": no tag.
    - Whitelist value: that key.
    - Unknown string: config default when apply_default else None.
    """
    if emotion is ...:
        if apply_default and TTS_DEFAULT_EMOTION:
            return normalize_emotion(TTS_DEFAULT_EMOTION)
        return None
    if isinstance(emotion, str) and emotion.strip().lower() in {"none", "null", "-"}:
        return None
    if emotion is None:
        # Explicit None from dispatch after normalize: treat as default unless
        # the tool forced \"none\" (then dispatch passes emotion=\"none\").
        if apply_default and TTS_DEFAULT_EMOTION:
            return normalize_emotion(TTS_DEFAULT_EMOTION)
        return None
    key = normalize_emotion(emotion)
    if key is not None:
        return key
    if apply_default and TTS_DEFAULT_EMOTION:
        return normalize_emotion(TTS_DEFAULT_EMOTION)
    return None


def sanitize_speech_text(text: object, *, max_chars: int | None = None) -> str:
    """Plain spoken text only: strip control brackets, collapse space, cap length."""
    if not isinstance(text, str):
        return ""
    cleaned = _CONTROL_BRACKET_RE.sub(" ", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    limit = TTS_MAX_CHARS if max_chars is None else max_chars
    if limit > 0 and len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip()
    return cleaned


def build_tts_payload_text(text: str, emotion: object = ...) -> str:
    """Text actually sent to Fish: optional one whitelist tag + spoken words."""
    spoken = sanitize_speech_text(text)
    if not spoken:
        return ""
    tag_key = resolve_emotion_key(emotion)
    tag = VOICE_EMOTIONS.get(tag_key or "")
    if tag:
        return f"{tag} {spoken}"
    return spoken


def synthesize_speech(
    text: str,
    *,
    emotion: object = ...,
    model: Optional[str] = None,
    reference_id: Optional[str] = None,
    fmt: Optional[str] = None,
    latency: Optional[str] = None,
    timeout: Optional[float] = None,
) -> bytes:
    """Blocking Fish TTS call. Returns raw audio bytes.

    Raises:
        TTSError: missing config, empty text, HTTP/API failure.
    """
    if not is_tts_ready() and not (FISH_API_KEY and (reference_id or FISH_VOICE_ID)):
        raise TTSError("TTS выключен или нет FISH_API_KEY / FISH_VOICE_ID.")

    spoken = sanitize_speech_text(text)
    if not spoken:
        raise TTSError("Пустой текст для озвучки.")

    emotion_key = resolve_emotion_key(emotion)
    payload_text = build_tts_payload_text(spoken, emotion)
    voice_id = (reference_id or FISH_VOICE_ID or "").strip()
    if not voice_id:
        raise TTSError("Не задан FISH_VOICE_ID.")
    api_key = FISH_API_KEY
    if not api_key:
        raise TTSError("Не задан FISH_API_KEY.")

    use_model = (model or TTS_MODEL or "s2.1-pro-free").strip()
    use_fmt = (fmt or TTS_FORMAT or "opus").strip().lower()
    use_latency = (latency or TTS_LATENCY or "normal").strip().lower()
    use_timeout = float(timeout if timeout is not None else TTS_TIMEOUT_SECONDS)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": use_model,
    }
    body: dict = {
        "text": payload_text,
        "reference_id": voice_id,
        "format": use_fmt,
        "latency": use_latency,
        "normalize": True,
    }
    if use_fmt == "opus":
        body["sample_rate"] = TTS_SAMPLE_RATE

    try:
        with httpx.Client(timeout=use_timeout) as client:
            res = client.post(API_URL, headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise TTSError(f"TTS таймаут ({use_timeout:g}s).") from exc
    except httpx.HTTPError as exc:
        raise TTSError(f"TTS сеть: {exc}") from exc

    if res.status_code != 200:
        try:
            detail = res.json()
        except Exception:
            detail = (res.text or "")[:300]
        raise TTSError(f"TTS HTTP {res.status_code}: {detail}")

    audio = res.content
    if not audio:
        raise TTSError("TTS вернул пустой аудиоответ.")
    logger.debug(
        "TTS ok model=%s voice=%s chars=%s bytes=%s emotion=%s",
        use_model,
        voice_id[:8],
        len(spoken),
        len(audio),
        emotion_key or "none",
    )
    return audio


def voice_filename(fmt: Optional[str] = None) -> str:
    """Telegram InputFile name hint (extension drives MIME sniffing)."""
    use_fmt = (fmt or TTS_FORMAT or "opus").strip().lower()
    if use_fmt == "opus":
        return "voice.ogg"
    if use_fmt == "mp3":
        return "voice.mp3"
    if use_fmt == "wav":
        return "voice.wav"
    return f"voice.{use_fmt}"
