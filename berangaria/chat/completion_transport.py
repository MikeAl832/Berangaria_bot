"""One chat completion request with optional Telegram streaming preview."""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from berangaria.chat.streaming import (
    IncompleteSSEError,
    StreamedCompletionResponse,
    TelegramStreamPreview,
)

logger = logging.getLogger(__name__)


def _normalize_in_band_error(response: Any) -> Any:
    """Expose OpenRouter's HTTP-200 non-streaming error body as an error status."""
    if getattr(response, "status_code", None) != 200:
        return response
    try:
        data = response.json()
    except Exception:
        return response
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return response
    try:
        status = int(error.get("code", 502))
    except (TypeError, ValueError):
        status = 502
    if not 400 <= status <= 599:
        status = 502
    return StreamedCompletionResponse(
        status_code=status,
        headers=getattr(response, "headers", {}),
        text=json.dumps(data, ensure_ascii=False),
        data=data,
    )


@dataclass(frozen=True)
class CompletionRuntime:
    """Per-turn streaming settings and patchable transport callback."""

    update: Any
    context: Any
    mentioned: bool
    api_url: str
    streaming_enabled: bool
    update_interval_seconds: float
    preview_min_chars: int
    stream_chat_completion: Callable[..., Awaitable[Any]]


async def request_completion(
    client: Any,
    payload: dict,
    headers: dict,
    turn: Any,
    runtime: CompletionRuntime,
) -> Any:
    """POST normally or reconstruct SSE while maintaining the preview status."""
    if not runtime.streaming_enabled:
        response = await client.post(runtime.api_url, json=payload, headers=headers)
        return _normalize_in_band_error(response)

    preview = TelegramStreamPreview(
        runtime.update,
        runtime.context,
        mentioned=runtime.mentioned,
        status_message=turn.status_message,
        interval_seconds=runtime.update_interval_seconds,
        min_chars=runtime.preview_min_chars,
    )
    try:
        try:
            return await runtime.stream_chat_completion(
                client,
                runtime.api_url,
                payload=payload,
                headers=headers,
                on_content=preview.publish,
            )
        except IncompleteSSEError as error:
            # An incomplete 200/SSE body must not be delivered because its text or
            # tool-call arguments may be truncated. Retry this provider round once
            # without SSE so a repeatedly broken stream does not exhaust all outer
            # retries using the same transport mode.
            logger.warning(
                "OpenRouter оборвал SSE без завершающего события; повторяем "
                "запрос без streaming: generation=%s events=%s "
                "partial_chars=%s tool_calls=%s messages=%s payload_chars=%s "
                "max_tokens=%s",
                error.generation_id,
                error.event_count,
                error.content_chars,
                error.tool_call_count,
                len(payload.get("messages") or []),
                len(json.dumps(payload, ensure_ascii=False, default=str)),
                payload.get("max_tokens"),
            )
            response = await client.post(runtime.api_url, json=payload, headers=headers)
            return _normalize_in_band_error(response)
    finally:
        # The final delivery and tool handlers must reuse a preview message that
        # Telegram already created, even when the stream itself fails.
        turn.status_message = preview.status_message
