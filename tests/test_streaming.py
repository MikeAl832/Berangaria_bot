import asyncio
import json

import pytest

from streaming import TelegramStreamPreview, stream_chat_completion


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


def test_stream_rejects_truncated_success_response():
    response = _StreamResponse([
        _event({"choices": [{"delta": {"content": "оборванный ответ"}}]}),
    ])

    with pytest.raises(RuntimeError, match="без \\[DONE\\]"):
        asyncio.run(stream_chat_completion(
            _Client(response),
            "https://api.example/chat",
            payload={},
            headers={},
        ))


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


def test_direct_group_preview_creates_then_edits_status_message():
    update, context = _Update("supergroup"), _Context()
    preview = TelegramStreamPreview(
        update, context, mentioned=True, interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("Первый кусок"))
    asyncio.run(preview.publish("Первый кусок ответа"))

    assert len(update.message.replies) == 1
    status = update.message.replies[0]
    assert status.initial_text == "Первый кусок"
    assert status.edits == ["Первый кусок ответа"]
    assert preview.status_message is status


def test_ambient_group_turn_does_not_create_preview_message():
    update, context = _Update("supergroup"), _Context()
    preview = TelegramStreamPreview(
        update, context, mentioned=False, interval_seconds=0, min_chars=1,
    )

    asyncio.run(preview.publish("Незаметный ambient ответ"))

    assert update.message.replies == []
    assert context.bot.drafts == []
