"""Conversation-history summarization for the chat pipeline."""

import copy
import logging
import re

import httpx

from berangaria.chat.history_rendering import renumber_sids
from berangaria.config import (
    CHAT_API_URL,
    FULL_DEBUG_LOGS,
    MODEL,
    SUMMARY_INTERVAL,
    apply_chat_routing,
    chat_api_headers,
)
from berangaria.core.utils import strip_tiktok_urls

logger = logging.getLogger(__name__)


def _message_reasoning_len(message: dict) -> int:
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if isinstance(reasoning, str):
        return len(reasoning)
    if isinstance(reasoning, list):
        return sum(
            len(part.get("text", ""))
            for part in reasoning
            if isinstance(part, dict)
        )
    return 0


async def summarize_history(history: list) -> list:
    """Compress older entries while leaving the live history untouched on failure."""
    to_summarize = history[:-SUMMARY_INTERVAL]
    keep_recent = copy.deepcopy(history[-SUMMARY_INTERVAL:])
    if not to_summarize:
        return history

    renumber_sids(keep_recent)
    summary_lines: list[str] = []
    for message in to_summarize:
        role = message.get("role") or "assistant"
        raw = message.get("content")
        if isinstance(raw, str) and strip_tiktok_urls(raw).strip():
            summary_lines.append(f"{role}: {strip_tiktok_urls(raw)}")
            continue
        for voice in message.get("voices") or []:
            if not isinstance(voice, dict):
                continue
            spoken = (voice.get("text") or "").strip()
            if spoken:
                summary_lines.append(f"{role}: {strip_tiktok_urls(spoken)}")

    text_to_summarize = "\n".join(summary_lines)
    if not text_to_summarize.strip():
        logger.warning(
            "📝 [yellow]Суммаризация пропущена:[/] "
            "нет текстового содержимого для сжатия"
        )
        return history

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Напиши ТЕХНИЧЕСКОЕ РЕЗЮМЕ диалога на русском:"
                    "Сожми этот диалог в КРАТКОЕ резюме на русском языке. "
                    "Пиши ТОЛЬКО суть, без вводных фраз. "
                    "Обязательно сохрани: имена, цифры, модели (например, RTX 5070 Ti), "
                    "технические характеристики, решения и важные факты. "
                    "НЕ пиши 'Пользователь сказал...', 'Собеседник ответил...' — "
                    "просто перескажи факты."
                ),
            },
            {"role": "user", "content": text_to_summarize},
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.9,
        "reasoning": {"effort": "high"},
    }
    apply_chat_routing(payload)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                CHAT_API_URL, json=payload, headers=chat_api_headers()
            )
            logger.info("Ответ сумморизации: [cyan]%s[/]", response.status_code)
            response.raise_for_status()
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            raw = message.get("content")
            if not isinstance(raw, str) or not raw.strip():
                reasoning_len = _message_reasoning_len(message)
                detail = (
                    f" (есть reasoning, {reasoning_len} символов)"
                    if reasoning_len
                    else ""
                )
                raise ValueError("пустой content в ответе суммаризации" + detail)

            summary = re.sub(
                r"<think>.*?</think>", "", raw, flags=re.DOTALL
            ).strip()
            if not summary:
                raise ValueError("резюме пустое после очистки thinking-тегов")
            logger.info("📝 Резюме истории получено (%s символов)", len(summary))
            if FULL_DEBUG_LOGS:
                logger.debug("Содержание:\n%s", summary)
            return [
                {
                    "role": "user",
                    "content": f"[Previous conversation summary: {summary}]",
                }
            ] + keep_recent
    except Exception as error:
        logger.error("❌ [red]Ошибка суммаризации:[/] %s", error)
        return history
