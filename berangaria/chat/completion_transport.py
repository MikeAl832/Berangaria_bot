"""One OpenRouter completion request with optional Telegram streaming preview."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from berangaria.chat.streaming import TelegramStreamPreview


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
        return await client.post(runtime.api_url, json=payload, headers=headers)

    preview = TelegramStreamPreview(
        runtime.update,
        runtime.context,
        mentioned=runtime.mentioned,
        status_message=turn.status_message,
        interval_seconds=runtime.update_interval_seconds,
        min_chars=runtime.preview_min_chars,
    )
    try:
        return await runtime.stream_chat_completion(
            client,
            runtime.api_url,
            payload=payload,
            headers=headers,
            on_content=preview.publish,
        )
    finally:
        # The final delivery and tool handlers must reuse a preview message that
        # Telegram already created, even when the stream itself fails.
        turn.status_message = preview.status_message
