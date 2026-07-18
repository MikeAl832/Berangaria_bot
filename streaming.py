"""DeepSeek SSE aggregation and throttled Telegram streaming previews."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ContentCallback = Callable[[str], Awaitable[None]]


@dataclass
class StreamedCompletionResponse:
    """Small response adapter matching the fields used by ``send_llm_request``."""

    status_code: int
    headers: Any = field(default_factory=dict)
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return self.data


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


async def stream_chat_completion(
    client,
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    on_content: ContentCallback | None = None,
) -> StreamedCompletionResponse:
    """Consume DeepSeek SSE and rebuild a normal chat-completion response.

    Only cumulative ``delta.content`` is exposed to ``on_content``. Reasoning and
    tool-call arguments are retained for API continuity but never sent to preview.
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
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        done_received = False
        role = "assistant"
        usage: dict[str, Any] = {}

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
                logger.warning("Пропущена некорректная SSE-строка DeepSeek: %r", raw_data[:200])
                continue

            if event.get("usage"):
                usage = event["usage"]

            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("role"):
                role = delta["role"]
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
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
            raise RuntimeError("DeepSeek SSE завершился без [DONE] и finish_reason")

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

        return StreamedCompletionResponse(
            status_code=200,
            headers=response_headers,
            data={
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
        preview = self._bounded(full_text)
        if not self.last_text and len(preview) < self.min_chars:
            return
        now = time.monotonic()
        if self.last_text and now - self.last_update < self.interval_seconds:
            return
        if preview == self.last_text:
            return

        try:
            if self.status_message is not None:
                await self.status_message.edit_text(preview)
            elif self.update.effective_chat.type == "private":
                kwargs = {
                    "chat_id": self.update.effective_chat.id,
                    "draft_id": self.draft_id,
                    "text": preview,
                }
                thread_id = getattr(self.update.message, "message_thread_id", None)
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
