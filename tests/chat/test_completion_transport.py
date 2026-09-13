import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from berangaria.chat.completion_transport import CompletionRuntime, request_completion
from berangaria.chat.streaming import IncompleteSSEError


class _Client:
    def __init__(self, response):
        self.response = response
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.response


class _JsonResponse:
    status_code = 200
    headers = {"x-generation-id": "gen-nonstream-error"}

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def _runtime(stream_callback):
    update = SimpleNamespace(message=SimpleNamespace(message_id=17))
    return CompletionRuntime(
        update=update,
        context=SimpleNamespace(),
        mentioned=False,
        api_url="https://openrouter.test/chat/completions",
        streaming_enabled=True,
        update_interval_seconds=0.8,
        preview_min_chars=12,
        stream_chat_completion=stream_callback,
    )


def test_incomplete_sse_falls_back_to_non_streaming_post():
    fallback_response = object()
    client = _Client(fallback_response)
    payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    headers = {"Authorization": "Bearer test"}
    turn = SimpleNamespace(status_message=None)
    stream_calls = []

    async def incomplete_stream(*args, **kwargs):
        stream_calls.append((args, kwargs))
        raise IncompleteSSEError("truncated")

    result = asyncio.run(
        request_completion(client, payload, headers, turn, _runtime(incomplete_stream))
    )

    assert result is fallback_response
    assert len(stream_calls) == 1
    assert client.posts == [(
        "https://openrouter.test/chat/completions",
        {"json": payload, "headers": headers},
    )]
    assert "stream" not in payload


def test_unrelated_stream_failure_is_not_hidden_by_fallback():
    client = _Client(object())
    turn = SimpleNamespace(status_message=None)

    async def broken_stream(*args, **kwargs):
        raise RuntimeError("parser bug")

    with pytest.raises(RuntimeError, match="parser bug"):
        asyncio.run(
            request_completion(client, {}, {}, turn, _runtime(broken_stream))
        )

    assert client.posts == []


def test_nonstreaming_http_200_error_body_is_exposed_as_failure():
    response = _JsonResponse({
        "id": "gen-nonstream-error",
        "error": {
            "code": 429,
            "message": "Provider overloaded",
            "metadata": {"error_type": "provider_overloaded"},
        },
    })
    client = _Client(response)

    async def unused_stream(*args, **kwargs):
        raise AssertionError("streaming must stay disabled")

    runtime = replace(_runtime(unused_stream), streaming_enabled=False)

    result = asyncio.run(request_completion(
        client,
        {"model": "test"},
        {},
        SimpleNamespace(status_message=None),
        runtime,
    ))

    assert result.status_code == 429
    assert result.json()["error"]["metadata"]["error_type"] == (
        "provider_overloaded"
    )


def test_explicit_stream_error_does_not_trigger_nonstreaming_fallback():
    client = _Client(object())
    stream_error = _JsonResponse({
        "error": {
            "code": 429,
            "message": "Provider overloaded",
            "metadata": {"error_type": "provider_overloaded"},
        },
    })
    stream_error.status_code = 429

    async def explicit_error(*args, **kwargs):
        return stream_error

    result = asyncio.run(request_completion(
        client,
        {"model": "test"},
        {},
        SimpleNamespace(status_message=None),
        _runtime(explicit_error),
    ))

    assert result is stream_error
    assert client.posts == []
