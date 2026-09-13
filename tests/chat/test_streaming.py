import asyncio
import json

import pytest

from berangaria.chat.streaming import (
    IncompleteSSEError,
    TelegramStreamPreview,
    stream_chat_completion,
)


def _event(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False)


class _StreamResponse:
    def __init__(self, lines, status_code=200, body=b""):
        self.lines = lines
        self.status_code = status_code
        self.headers = {"Retry-After": "7"}
        self.body = body

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aread(self):
        return self.body


class _StreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        return None


class _Client:
    def __init__(self, response):
        self.response = response
        self.request = None

    def stream(self, method, url, **kwargs):
        self.request = (method, url, kwargs)
        return _StreamContext(self.response)


def test_stream_aggregates_openai_reasoning_without_previewing_it():
    response = _StreamResponse([
        ": keep-alive",
        _event({"choices": [{"delta": {"role": "assistant", "reasoning": "секрет"}}]}),
        _event({"choices": [{"delta": {"content": "При"}}]}),
        _event({"choices": [{"delta": {"content": "вет"}, "finish_reason": "stop"}]}),
        _event({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}),
        "data: [DONE]",
    ])
    client = _Client(response)
    previews = []

    async def on_content(text):
        previews.append(text)

    result = asyncio.run(stream_chat_completion(
        client,
        "https://openrouter.ai/api/v1/chat/completions",
        payload={"model": "openai/gpt-5.6-luna", "messages": []},
        headers={"Authorization": "Bearer test"},
        on_content=on_content,
    ))

    data = result.json()
    assert data["choices"][0]["message"]["content"] == "Привет"
    assert data["choices"][0]["message"]["reasoning_content"] == "секрет"
    assert previews == ["При", "Привет"]
    assert all("секрет" not in preview for preview in previews)


def test_stream_aggregates_content_without_previewing_reasoning():
    response = _StreamResponse([
        ": keep-alive",
        _event({"choices": [{"delta": {"role": "assistant", "reasoning_content": "секрет"}}]}),
        _event({"choices": [{"delta": {"content": "При"}}]}),
        _event({"choices": [{"delta": {"content": "вет"}, "finish_reason": "stop"}]}),
        _event({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}),
        "data: [DONE]",
    ])
    client = _Client(response)
    previews = []

    async def on_content(text):
        previews.append(text)

    result = asyncio.run(stream_chat_completion(
        client,
        "https://api.example/chat",
        payload={"model": "test", "messages": []},
        headers={"Authorization": "Bearer test"},
        on_content=on_content,
    ))

    data = result.json()
    assert data["choices"][0]["message"]["content"] == "Привет"
    assert data["choices"][0]["message"]["reasoning_content"] == "секрет"
    assert data["usage"]["prompt_tokens"] == 10
    assert previews == ["При", "Привет"]
    assert all("секрет" not in preview for preview in previews)
    assert client.request[2]["json"]["stream"] is True
    assert client.request[2]["json"]["stream_options"] == {"include_usage": True}


def test_stream_preserves_structured_reasoning_for_tool_continuity():
    detail_chunks = [
        {
            "type": "reasoning.summary",
            "summary": "Нужно проверить актуальный курс.",
            "id": "reasoning-summary-1",
            "format": "xai-responses-v1",
            "index": 0,
        },
        {
            "type": "reasoning.encrypted",
            "data": "encrypted-part",
            "id": "reasoning-encrypted-1",
            "format": "xai-responses-v1",
            "index": 1,
        },
        {
            "type": "reasoning.text",
            "text": "Вызываю поиск.",
            "signature": "sig-1",
            "id": "reasoning-text-1",
            "format": "xai-responses-v1",
            "index": 2,
        },
    ]
    response = _StreamResponse([
        _event({
            "id": "generation-1",
            "model": "openai/gpt-5.6-sol",
            "provider": "OpenAI",
            "service_tier": "flex",
            "choices": [{"delta": {
            "role": "assistant",
            "reasoning": "плоская копия, которую не надо дублировать",
            "reasoning_details": detail_chunks[:2],
        }}]}),
        _event({"choices": [{"delta": {
            "reasoning_details": detail_chunks[2:],
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query":"курс"}'},
            }],
        }, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ])
    previews = []

    async def on_content(text):
        previews.append(text)

    result = asyncio.run(stream_chat_completion(
        _Client(response),
        "https://openrouter.ai/api/v1/chat/completions",
        payload={"model": "openai/gpt-5.6-sol", "messages": []},
        headers={"Authorization": "Bearer test"},
        on_content=on_content,
    ))

    message = result.json()["choices"][0]["message"]
    assert result.json()["id"] == "generation-1"
    assert result.json()["model"] == "openai/gpt-5.6-sol"
    assert result.json()["provider"] == "OpenAI"
    assert result.json()["service_tier"] == "flex"
    assert message["reasoning_details"] == detail_chunks
    assert "reasoning_content" not in message
    assert message["tool_calls"][0]["function"]["name"] == "web_search"
    assert previews == []


def test_stream_preserves_openrouter_metadata_from_terminal_chunk():
    metadata = {
        "strategy": "direct",
        "region": "iad",
        "attempt": 1,
        "endpoints": {
            "available": [{
                "provider": "OpenAI",
                "model": "openai/gpt-5.6-sol",
                "selected": True,
            }],
        },
    }
    response = _StreamResponse([
        _event({
            "id": "generation-1",
            "choices": [{
                "delta": {"role": "assistant", "content": "готово"},
                "finish_reason": "stop",
            }],
        }),
        _event({"choices": [], "openrouter_metadata": metadata}),
        "data: [DONE]",
    ])

    result = asyncio.run(stream_chat_completion(
        _Client(response),
        "https://openrouter.ai/api/v1/chat/completions",
        payload={"model": "openai/gpt-5.6-sol", "messages": []},
        headers={"X-OpenRouter-Metadata": "enabled"},
    ))

    assert result.json()["openrouter_metadata"] == metadata


def test_stream_reassembles_tool_call_arguments():
    response = _StreamResponse([
        _event({"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_", "arguments": '{"q":'},
        }]}}]}),
        _event({"choices": [{
            "delta": {"tool_calls": [{
                "index": 0,
                "function": {"name": "search", "arguments": '"кот"}'},
            }]},
            "finish_reason": "tool_calls",
        }]}),
        "data: [DONE]",
    ])

    result = asyncio.run(stream_chat_completion(
        _Client(response),
        "https://api.example/chat",
        payload={},
        headers={},
    ))

    choice = result.json()["choices"][0]
    call = choice["message"]["tool_calls"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert call["id"] == "call_1"
    assert call["function"] == {"name": "web_search", "arguments": '{"q":"кот"}'}


def test_stream_preserves_http_error_for_existing_retry_logic():
    response = _StreamResponse([], status_code=429, body=b"rate limited")

    result = asyncio.run(stream_chat_completion(
        _Client(response),
        "https://api.example/chat",
        payload={},
        headers={},
    ))

    assert result.status_code == 429
    assert result.headers["Retry-After"] == "7"
    assert result.text == "rate limited"


def test_stream_surfaces_error_without_accepting_partial_as_completion():
    response = _StreamResponse([
        _event({
            "id": "gen-context-1",
            "model": "openai/gpt-5.6-sol",
            "provider": "OpenAI",
            "choices": [{"delta": {"content": "частичный текст"}}],
        }),
        _event({
            "id": "gen-context-1",
            "model": "openai/gpt-5.6-sol",
            "provider": "OpenAI",
            "error": {
                "code": 400,
                "message": "Context length exceeded",
                "metadata": {"error_type": "context_length_exceeded"},
            },
            "choices": [{
                "index": 0,
                "delta": {"content": ""},
                "finish_reason": "error",
            }],
        }),
    ])
    previews = []

    async def on_content(text):
        previews.append(text)

    result = asyncio.run(stream_chat_completion(
        _Client(response),
        "https://openrouter.ai/api/v1/chat/completions",
        payload={},
        headers={},
        on_content=on_content,
    ))

    assert result.status_code == 400
    assert result.json()["error"]["metadata"]["error_type"] == (
        "context_length_exceeded"
    )
    assert result.json()["id"] == "gen-context-1"
    assert previews == ["частичный текст"]
    assert "частичный текст" not in result.text


def test_stream_rejects_truncated_success_response():
    response = _StreamResponse([
        _event({
            "id": "gen-truncated-1",
            "choices": [{"delta": {"content": "оборванный ответ"}}],
        }),
    ])

    with pytest.raises(IncompleteSSEError, match="без \\[DONE\\]") as captured:
        asyncio.run(stream_chat_completion(
            _Client(response),
            "https://api.example/chat",
            payload={},
            headers={},
        ))

    assert captured.value.generation_id == "gen-truncated-1"
    assert captured.value.event_count == 1
    assert captured.value.content_chars == len("оборванный ответ")
    assert captured.value.tool_call_count == 0


class _StatusMessage:
    def __init__(self, message_id=500):
        self.message_id = message_id
        self.edits = []

    async def edit_text(self, text, **kwargs):
        self.edits.append(text)


class _Message:
    message_id = 77
    message_thread_id = None

    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        status = _StatusMessage()
        status.initial_text = text
        self.replies.append(status)
        return status


class _Chat:
    def __init__(self, chat_type):
        self.id = 42
        self.type = chat_type


class _Update:
    def __init__(self, chat_type):
        self.message = _Message()
        self.effective_chat = _Chat(chat_type)


class _Bot:
    def __init__(self):
        self.drafts = []

    async def send_message_draft(self, **kwargs):
        self.drafts.append(kwargs)


class _Context:
    def __init__(self):
        self.bot = _Bot()


def test_private_preview_uses_native_draft_and_minimum_length():
    update, context = _Update("private"), _Context()
    preview = TelegramStreamPreview(
        update, context, mentioned=True, interval_seconds=0, min_chars=5,
    )

    asyncio.run(preview.publish("При"))
    asyncio.run(preview.publish("Привет"))

    assert update.message.replies == []
    assert context.bot.drafts == [{
        "chat_id": 42,
        "draft_id": 77,
        "text": "Привет",
    }]


def test_direct_group_turn_waits_for_final_delivery():
    update, context = _Update("supergroup"), _Context()
    preview = TelegramStreamPreview(
        update, context, mentioned=True, interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("Первый кусок"))
    asyncio.run(preview.publish("Первый кусок ответа"))

    assert update.message.replies == []
    assert preview.status_message is None


def test_ambient_group_turn_does_not_create_preview_message():
    update, context = _Update("supergroup"), _Context()
    preview = TelegramStreamPreview(
        update, context, mentioned=False, interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("Незаметный ambient ответ"))

    assert update.message.replies == []
    assert context.bot.drafts == []


def test_group_status_message_is_not_turned_into_a_preview():
    """Статусная плашка в группе — обычное сообщение чата.

    Переписывать её кусками ответа значит завести персистентное превью,
    запрещённое для групп: неоднозначный таймаут теряет message_id, и в чате
    остаётся неудаляемый обрывок рядом с финальным ответом.
    """
    update, context = _Update("supergroup"), _Context()
    status = _StatusMessage()
    preview = TelegramStreamPreview(
        update, context, mentioned=True, status_message=status,
        interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("Первый кусок"))
    asyncio.run(preview.publish("Первый кусок ответа"))

    assert status.edits == []
    assert context.bot.drafts == []


def test_private_status_message_still_receives_preview():
    update, context = _Update("private"), _Context()
    status = _StatusMessage()
    preview = TelegramStreamPreview(
        update, context, mentioned=True, status_message=status,
        interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("Ответ по ходу"))

    assert status.edits == ["Ответ по ходу"]


def test_preview_strips_internal_handles_and_memory_tag():
    """Пользователь не должен видеть, как «печатаются» служебные теги."""
    update, context = _Update("private"), _Context()
    preview = TelegramStreamPreview(
        update, context, mentioned=True, interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("Как ты писал в [#26], я помню [Context from memory: X]"))

    text = context.bot.drafts[-1]["text"]
    assert "[#26]" not in text
    assert "[Context from memory:" not in text


def test_preview_skips_update_that_is_only_internal_tags():
    update, context = _Update("private"), _Context()
    preview = TelegramStreamPreview(
        update, context, mentioned=True, interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("[#26]"))

    assert context.bot.drafts == []
