"""OpenAI-compatible SSE aggregation and throttled Telegram streaming previews."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from berangaria.chat.chat_actions import effective_message_thread_id
from berangaria.core.utils import strip_internal_tags

logger = logging.getLogger(__name__)

ContentCallback = Callable[[str], Awaitable[None]]


def _safe_error_value(value: Any, *, max_chars: int) -> str:
    """Keep provider identifiers useful without allowing multiline log injection."""
    return " ".join(str(value or "unknown").split())[:max_chars]


class IncompleteSSEError(RuntimeError):
    """The provider closed a successful SSE response before its terminal event."""

    def __init__(
        self,
        message: str,
        *,
        generation_id: str = "unknown",
        event_count: int = 0,
        content_chars: int = 0,
        tool_call_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.generation_id = generation_id
        self.event_count = event_count
        self.content_chars = content_chars
        self.tool_call_count = tool_call_count


@dataclass
class StreamedCompletionResponse:
    """Small response adapter matching the fields used by ``send_llm_request``."""

    status_code: int
    headers: Any = field(default_factory=dict)
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return self.data


def _delta_reasoning_text(delta: dict[str, Any]) -> str:
    """Extract reasoning text from a streamed delta without exposing it to preview."""
    reasoning_content = delta.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content
    reasoning = delta.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    if isinstance(reasoning, dict):
        text = reasoning.get("content") or reasoning.get("text") or ""
        return text if isinstance(text, str) else ""
    return ""


def _merge_tool_call(target: dict[str, Any], delta: dict[str, Any]) -> None:
    """Merge one OpenAI-compatible streamed tool-call delta in place."""
    if delta.get("id"):
        target["id"] = delta["id"]
    if delta.get("type"):
        target["type"] = delta["type"]
    function_delta = delta.get("function") or {}
    function = target.setdefault("function", {"name": "", "arguments": ""})
    if function_delta.get("name"):
        function["name"] += function_delta["name"]
    if function_delta.get("arguments"):
        function["arguments"] += function_delta["arguments"]


def _provider_error_status(error: dict[str, Any]) -> int:
    """Return a usable HTTP-equivalent status from an in-band SSE error."""
    try:
        status = int(error.get("code", 502))
    except (TypeError, ValueError):
        return 502
    return status if 400 <= status <= 599 else 502


async def stream_chat_completion(
    client,
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    on_content: ContentCallback | None = None,
) -> StreamedCompletionResponse:
    """Consume OpenAI-compatible SSE and rebuild a normal chat-completion response.

    Only cumulative ``delta.content`` is exposed to ``on_content``. Structured
    reasoning, legacy reasoning text, and tool-call arguments are retained for API
    continuity but never sent to preview.
    """
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    stream_payload["stream_options"] = {"include_usage": True}

    async with client.stream(
        "POST", url, json=stream_payload, headers=headers
    ) as response:
        response_headers = response.headers
        if response.status_code != 200:
            body = await response.aread()
            error_text = body.decode("utf-8", errors="replace")
            return StreamedCompletionResponse(
                status_code=response.status_code,
                headers=response_headers,
                text=error_text,
            )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_details: list[dict[str, Any]] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        done_received = False
        role = "assistant"
        usage: dict[str, Any] = {}
        response_meta: dict[str, Any] = {}
        event_count = 0

        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            raw_data = line[5:].strip()
            if raw_data == "[DONE]":
                done_received = True
                break
            try:
                event = json.loads(raw_data)
            except json.JSONDecodeError:
                logger.warning("Пропущена некорректная SSE-строка чат-модели: %r", raw_data[:200])
                continue
            event_count += 1

            if event.get("usage"):
                usage = event["usage"]
            for field in ("id", "model", "provider", "service_tier"):
                if event.get(field) is not None:
                    response_meta[field] = event[field]
            if isinstance(event.get("openrouter_metadata"), dict):
                response_meta["openrouter_metadata"] = event["openrouter_metadata"]

            provider_error = event.get("error")
            if isinstance(provider_error, dict):
                metadata = provider_error.get("metadata") or {}
                error_type = _safe_error_value((
                    metadata.get("error_type")
                    if isinstance(metadata, dict)
                    else None
                ), max_chars=64)
                provider_code = _safe_error_value((
                    metadata.get("provider_code")
                    if isinstance(metadata, dict)
                    else None
                ), max_chars=64)
                error_message = _safe_error_value(
                    provider_error.get("message"), max_chars=160
                )
                generation_id = _safe_error_value(
                    event.get("id")
                    or response_meta.get("id")
                    or response_headers.get("x-generation-id")
                    or "unknown",
                    max_chars=96,
                )
                logger.warning(
                    "OpenRouter SSE error: generation=%s code=%s type=%s "
                    "provider_code=%s message=%r events=%s partial_chars=%s tool_calls=%s",
                    generation_id,
                    provider_error.get("code", "unknown"),
                    error_type,
                    provider_code,
                    error_message,
                    event_count,
                    sum(len(part) for part in content_parts),
                    len(tool_calls),
                )
                return StreamedCompletionResponse(
                    status_code=_provider_error_status(provider_error),
                    headers=response_headers,
                    text=json.dumps(event, ensure_ascii=False),
                    data=event,
                )

            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("role"):
                role = delta["role"]
            reasoning_text = _delta_reasoning_text(delta)
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
            detail_chunks = delta.get("reasoning_details")
            if isinstance(detail_chunks, list):
                # Reasoning models require the complete structured sequence to be
                # echoed unmodified during tool use. Streaming chunks are already
                # in provider order, so concatenate them instead of merging by index.
                reasoning_details.extend(
                    detail for detail in detail_chunks if isinstance(detail, dict)
                )
            if delta.get("content"):
                content_parts.append(delta["content"])
                if on_content is not None:
                    try:
                        await on_content("".join(content_parts))
                    except Exception as exc:
                        # Preview is best-effort and must never abort the LLM request.
                        logger.warning("Не удалось обновить streaming preview: %s", exc)
            for tool_delta in delta.get("tool_calls") or []:
                index = int(tool_delta.get("index", 0))
                target = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                _merge_tool_call(target, tool_delta)

        if not done_received and not finish_reason:
            generation_id = str(
                response_meta.get("id")
                or response_headers.get("x-generation-id")
                or "unknown"
            )
            raise IncompleteSSEError(
                "Chat completion SSE завершился без [DONE] и finish_reason",
                generation_id=generation_id,
                event_count=event_count,
                content_chars=sum(len(part) for part in content_parts),
                tool_call_count=len(tool_calls),
            )

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts),
        }
        if reasoning_details:
            # Special summarized/encrypted reasoning must keep its structured form.
            # Do not duplicate the flattened legacy string in the next tool request.
            message["reasoning_details"] = reasoning_details
        elif reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

        return StreamedCompletionResponse(
            status_code=200,
            headers=response_headers,
            data={
                **response_meta,
                "choices": [{"finish_reason": finish_reason, "message": message}],
                "usage": usage,
            },
        )


class TelegramStreamPreview:
    """Publish private-chat drafts without persisting them in chat history."""

    def __init__(
        self,
        update,
        context,
        *,
        mentioned: bool,
        status_message=None,
        interval_seconds: float = 0.8,
        min_chars: int = 12,
    ) -> None:
        self.update = update
        self.context = context
        self.mentioned = mentioned
        self.status_message = status_message
        self.interval_seconds = max(0.0, interval_seconds)
        self.min_chars = max(1, min_chars)
        self.last_update = 0.0
        self.last_text = ""
        self.disabled = False
        message_id = getattr(update.message, "message_id", 1) or 1
        self.draft_id = int(message_id)

    @staticmethod
    def _bounded(text: str) -> str:
        if len(text) <= 4096:
            return text
        return "…" + text[-4095:]

    async def publish(self, full_text: str) -> None:
        if self.disabled or not full_text:
            return
        # Служебные теги вычищаются только из финального ответа, поэтому в
        # превью пользователь наблюдал, как «печатаются» [#26] и
        # [Context from memory: ...]. Полную очистку финала здесь применять
        # нельзя — срез точки и правило молчания относятся к готовому ответу.
        preview = self._bounded(strip_internal_tags(full_text))
        if not preview:
            return
        if not self.last_text and len(preview) < self.min_chars:
            return
        now = time.monotonic()
        if self.last_text and now - self.last_update < self.interval_seconds:
            return
        if preview == self.last_text:
            return

        is_private = self.update.effective_chat.type == "private"
        try:
            # status_message в группе — обычное сообщение чата: переписывать его
            # кусками ответа значит завести ровно то персистентное превью,
            # которое запрещено для групп (см. ветку else ниже).
            if self.status_message is not None and is_private:
                await self.status_message.edit_text(preview)
            elif is_private:
                kwargs = {
                    "chat_id": self.update.effective_chat.id,
                    "draft_id": self.draft_id,
                    "text": preview,
                }
                thread_id = effective_message_thread_id(self.update.message)
                if thread_id is not None:
                    kwargs["message_thread_id"] = thread_id
                await self.context.bot.send_message_draft(**kwargs)
            else:
                # A timed-out group send may still have reached Telegram without
                # returning its message ID. It cannot then be edited or deleted and
                # final delivery would leave a duplicate partial message. Groups wait
                # for the single final response; private chats use native drafts.
                return
        except Exception as exc:
            logger.warning("Telegram streaming preview отключён для текущего хода: %s", exc)
            self.disabled = True
            return

        self.last_text = preview
        self.last_update = now
