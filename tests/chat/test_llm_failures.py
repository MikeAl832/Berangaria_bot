import asyncio
import copy
import json
import logging
from types import SimpleNamespace

from telegram.error import BadRequest, TimedOut

from berangaria.chat import llm_client
from berangaria.chat.chat_actions import CHAT_ACTION_REFRESH_SECONDS
from berangaria.chat.turn_outcome import TurnOutcome
from berangaria.memory import store as memory_store
import pytest
from berangaria.core import state
from berangaria.chat.streaming import StreamedCompletionResponse
from berangaria.analytics import store as analytics_store


def _record_transport_retry_sleeps(sleeps, *, budget=3, message="transport retry budget was reset"):
    """Count LLM backoff sleeps without treating typing-heartbeat pauses as retries.

    ``monkeypatch.setattr(llm_client.asyncio, "sleep", ...)`` replaces the
    process-wide ``asyncio.sleep``. ChatActionHeartbeat uses the same function
    for its refresh interval; parking that wait until cancel keeps the
    heartbeat from looking like a third transport retry.
    """

    async def fake_sleep(seconds):
        if seconds == CHAT_ACTION_REFRESH_SECONDS:
            await asyncio.Event().wait()
            return
        sleeps.append(seconds)
        assert len(sleeps) < budget, message

    return fake_sleep


class _Response:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


def _client_returning(response):
    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return response

    return Client


def test_rate_limit_backoff_honors_header_and_grows_without_one(monkeypatch):
    monkeypatch.setattr(llm_client.random, "uniform", lambda low, high: 1.0)

    assert llm_client._rate_limit_retry_delay(
        _Response(429, headers={"Retry-After": "17"}), 4
    ) == (17.0, "retry-after")
    assert [
        llm_client._rate_limit_retry_delay(_Response(429), failure)[0]
        for failure in range(1, 5)
    ] == [5.0, 10.0, 20.0, 30.0]


def test_provider_error_summary_is_compact_and_structured():
    response = _Response(429, {
        "id": "gen-rate-1",
        "error": {
            "message": "Rate limit\nexceeded " + ("x" * 300),
            "metadata": {
                "error_type": "rate_limit_exceeded",
                "provider_code": "rate_limited",
            },
        },
    })

    summary = llm_client._provider_error_summary(response)

    assert summary["generation_id"] == "gen-rate-1"
    assert summary["error_type"] == "rate_limit_exceeded"
    assert summary["provider_code"] == "rate_limited"
    assert "\n" not in summary["message"]
    assert len(summary["message"]) == 160


class _Message:
    message_id = 10
    message_thread_id = None

    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class _Chat:
    id = 100
    type = "private"


class _Update:
    def __init__(self):
        self.message = _Message()
        self.effective_chat = _Chat()


class _FailingBot:
    async def send_message(self, **kwargs):
        raise RuntimeError("telegram unavailable")


class _SuccessfulBot:
    def __init__(self):
        self.drafts = []
        self.messages = []

    async def send_message_draft(self, **kwargs):
        self.drafts.append(kwargs)

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return type("SentMessage", (), {"message_id": 99})()


class _Context:
    def __init__(self, bot):
        self.bot = bot


def test_ambient_empty_reply_stays_silent_without_retry(monkeypatch, tmp_path):
    posts = []
    empty = _Response(200, {
        "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
        "usage": {},
    })
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient", _sequenced_client(posts, [empty]),
    )
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "group_-1"
    history = [{"role": "user", "content": "[Message: привет всем]", "sid": 1, "mid": 10}]
    state.histories[key] = history
    bot = _SuccessfulBot()

    outcome = asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, False,
    ))

    assert outcome is TurnOutcome.SILENT
    assert len(posts) == 1
    assert posts[0]["messages"][-1]["role"] != "system"
    assert [entry["role"] for entry in history] == ["user"]
    assert bot.messages == []


def test_failed_delivery_does_not_create_ghost_assistant(monkeypatch):
    response = _Response(200, {
        "choices": [{"finish_reason": "stop", "message": {"content": "ответ"}}],
        "usage": {},
    })
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _client_returning(response))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    key = "private_1"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories[key] = history
    state.chat_tokens.pop(key, None)

    with pytest.raises(llm_client.ReplyDeliveryError):
        asyncio.run(llm_client.send_llm_request(
            _Update(), _Context(_FailingBot()), key, history, "Миша", 1, True,
        ))

    assert [entry["role"] for entry in history] == ["user"]


def test_api_400_preserves_persisted_history(monkeypatch, tmp_path):
    response = _Response(400, text="bad context")
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _client_returning(response))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    key = "private_1"
    history = [{"role": "user", "content": "[Message: сломано]", "sid": 1, "mid": 10}]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)
    state.init_db()
    state.save_history(key)

    outcome = asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_FailingBot()), key, history, "Миша", 1, True,
    ))

    assert outcome is TurnOutcome.FAILED
    assert [entry["role"] for entry in history] == ["user"]
    state.histories.clear()
    state.load_all_histories()
    assert state.histories[key] == history


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_terminal_http_error_returns_failed_without_assistant(
    monkeypatch, isolated_db, status_code,
):
    posts = []
    sleeps = []

    monkeypatch.setattr(llm_client, "MAX_API_RETRIES", 3)
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(
        llm_client.asyncio,
        "sleep",
        _record_transport_retry_sleeps(sleeps),
    )
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient",
        _sequenced_client(posts, [_Response(status_code)]),
    )
    key = "private_1"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories[key] = history
    update = _Update()

    outcome = asyncio.run(llm_client.send_llm_request(
        update, _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
    ))

    assert outcome is TurnOutcome.FAILED
    assert len(posts) == 3
    assert len(sleeps) == 2
    assert len(update.message.replies) == 1
    assert [entry["role"] for entry in history] == ["user"]


@pytest.mark.parametrize("body", [
    "not JSON", "null", "[]", "42", "{}",
    '{"choices": []}', '{"choices": {"0": {}}}',
    '{"choices": [null]}', '{"choices": [[]]}',
    '{"choices": [{}]}', '{"choices": [{"message": null}]}',
    '{"choices": [{"message": []}]}',
])
def test_malformed_http_200_exhausts_retry_budget(monkeypatch, isolated_db, body):
    posts = []
    sleeps = []

    monkeypatch.setattr(llm_client, "MAX_API_RETRIES", 3)
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(
        llm_client.asyncio,
        "sleep",
        _record_transport_retry_sleeps(
            sleeps,
            message="malformed response reset the retry budget",
        ),
    )
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient",
        _sequenced_client(posts, [llm_client.httpx.Response(200, text=body)]),
    )
    key = "private_1"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories[key] = history
    update = _Update()

    outcome = asyncio.run(llm_client.send_llm_request(
        update, _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
    ))

    assert outcome is TurnOutcome.FAILED
    assert len(posts) == 3
    assert sleeps == [1, 2]
    assert len(update.message.replies) == 1
    assert [entry["role"] for entry in history] == ["user"]


def test_direct_empty_reply_returns_failed(monkeypatch, isolated_db):
    posts = []
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient",
        _sequenced_client(posts, [_Response(200, {
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
        })]),
    )
    key = "private_1"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories[key] = history
    bot = _SuccessfulBot()

    outcome = asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, True,
    ))

    assert outcome is TurnOutcome.FAILED
    assert len(posts) == 1
    assert bot.messages == []
    assert [entry["role"] for entry in history] == ["user"]


def test_streaming_preview_finishes_with_persisted_delivery(monkeypatch, tmp_path):
    reasoning_details = [{
        "type": "reasoning.encrypted",
        "data": "opaque-state",
        "id": "reasoning-1",
        "format": "xai-responses-v1",
        "index": 0,
    }]
    payloads = []
    captured_headers = []

    async def fake_stream(client, url, *, payload, headers, on_content):
        payloads.append(copy.deepcopy(payload))
        captured_headers.append(dict(headers))
        await on_content("потоковый ответ")
        return StreamedCompletionResponse(
            status_code=200,
            data={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "потоковый ответ",
                        "reasoning_details": reasoning_details,
                    },
                }],
                "usage": {},
            },
        )

    response = _Response(500)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _client_returning(response))
    monkeypatch.setattr(llm_client, "stream_chat_completion", fake_stream)
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", True)
    monkeypatch.setattr(llm_client, "STREAM_UPDATE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(llm_client, "STREAM_PREVIEW_MIN_CHARS", 1)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{
        "role": "user",
        "content": "[Message: привет]",
        "sid": 1,
        "mid": 10,
        "provider_sent": False,
    }]
    state.histories[key] = history
    state.chat_tokens.pop(key, None)
    bot = _SuccessfulBot()

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, True,
    ))

    assert bot.drafts[0]["text"] == "потоковый ответ"
    assert bot.messages[0]["text"] == "потоковый ответ"
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "потоковый ответ"
    assert history[-1]["reasoning_details"] == reasoning_details
    assert history[-1]["reasoning_details"] is not reasoning_details
    assert history[-1]["mid"] == 99
    assert history[0]["provider_sent"] is True
    assert history[-1]["provider_sent"] is False
    assert payloads[0]["session_id"] == llm_client._chat_session_id(key)
    assert payloads[0]["provider"] == {
        "only": ["meta"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert "service_tier" not in payloads[0]
    assert captured_headers[0]["x-session-id"] == llm_client._chat_session_id(key)
    assert "x-grok-conv-id" not in captured_headers[0]

    state.histories.clear()
    state.load_all_histories()
    restored = state.histories[key][-1]
    assert restored["reasoning_details"] == reasoning_details
    assert llm_client._render_history_for_api([restored]) == [{
        "role": "assistant",
        "content": "потоковый ответ",
        "reasoning_details": reasoning_details,
    }]


def test_confirmed_reply_and_usage_are_recorded(monkeypatch, tmp_path, caplog):
    response = _Response(200, {
        "id": "gen-meta-1",
        "provider": "Meta",
        "model": "meta/muse-spark-1.3-20260902",
        "choices": [{"finish_reason": "stop", "message": {"content": "ответ"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 10,
            },
            "cost": 0.000321,
        },
    })
    caplog.set_level(logging.INFO, logger="berangaria.chat.llm_client")
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _client_returning(response))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{
        "role": "user",
        "content": "[Message: привет]",
        "sid": 1,
        "mid": 10,
        "author_id": 1,
        "author_name": "Миша",
    }]
    state.histories[key] = history

    outcome = asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
    ))

    assert outcome is TurnOutcome.DELIVERED
    overview = analytics_store.get_overview("all", chat_id=100)
    assert overview["requests"] == 1
    assert overview["cost_microusd"] == 321
    assert overview["assistant_replies"] == 1
    leaders = analytics_store.get_leaderboards("all", chat_id=100)
    assert leaders["replies"][0]["user_id"] == 1
    assert leaders["cost"][0]["value"] == 321
    assert "provider=Meta model=meta/muse-spark-1.3-20260902" in caplog.text


def test_unexpected_provider_alerts_owner(monkeypatch, tmp_path):
    response = _Response(200, {
        "id": "gen-wrong-provider-1",
        "provider": "Other",
        "model": "meta/muse-spark-1.3-20260902",
        "choices": [{"finish_reason": "stop", "message": {"content": "ответ"}}],
        "usage": {},
    })
    alerts = []

    async def fake_notify_owner(bot, *, category, message, error=None):
        alerts.append({"category": category, "message": message, "error": error})
        return True

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _client_returning(response))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(llm_client.alerts, "notify_owner", fake_notify_owner)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{
        "role": "user",
        "content": "[Message: привет]",
        "sid": 1,
        "mid": 10,
    }]
    state.histories[key] = history

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
    ))

    assert len(alerts) == 1
    assert alerts[0]["category"] == "LLM routing provider"
    assert "provider=Other" in alerts[0]["message"]
    assert "ожидался meta" in alerts[0]["message"]
    assert "generation=gen-wrong-provider-1" in alerts[0]["message"]


def test_telegram_cleanup_does_not_change_provider_history(monkeypatch, tmp_path):
    response = _Response(200, {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Сырая точка."},
        }],
        "usage": {},
    })
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _client_returning(response))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{
        "role": "user",
        "content": "[Message: привет]",
        "sid": 1,
        "mid": 10,
    }]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)
    bot = _SuccessfulBot()

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, True,
    ))

    assert bot.messages[0]["text"] == "Сырая точка"
    assert history[-1]["content"] == "Сырая точка"
    assert history[-1]["provider_messages"] == [{
        "role": "assistant",
        "content": "Сырая точка.",
    }]
    assert llm_client._render_history_for_api(history)[-1] == {
        "role": "assistant",
        "content": "Сырая точка.",
    }

    state.histories.clear()
    state.load_all_histories()
    restored = state.histories[key][-1]
    assert restored["content"] == "Сырая точка"
    assert restored["provider_messages"][0]["content"] == "Сырая точка."


def test_terminal_reply_failure_does_not_resend_unanswered_tool_calls(
    monkeypatch, tmp_path
):
    """reply_to_message терминальный: payload уже содержит tool_calls без ответов.

    Сбой при подготовке текста не должен уйти в общий retry — иначе такой payload
    переотправляется, DeepSeek отвечает 400, и ветка очистки стирает историю чата.
    """
    posts = []

    class CountingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            posts.append(kwargs.get("json"))
            return _Response(200, {
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "reply_to_message",
                                "arguments": '{"id": 1, "text": "ответ"}',
                            },
                        }],
                    },
                }],
                "usage": {},
            })

    def exploding_clean_reply(text):
        raise TypeError("expected string or bytes-like object")

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", CountingClient)
    monkeypatch.setattr(llm_client, "_clean_reply", exploding_clean_reply)
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
    ))

    # Ровно один запрос: ход завершён на месте, а не отправлен в retry-цикл.
    assert len(posts) == 1
    # История цела — ни призрачного assistant, ни очистки.
    assert [entry["role"] for entry in history] == ["user"]
    assert state.histories[key] == history


def test_non_string_tool_reply_text_keeps_history(monkeypatch, tmp_path):
    """Сквозная проверка: нестроковый `text` от модели не роняет ход."""
    posts = []

    class CountingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            posts.append(kwargs.get("json"))
            return _Response(200, {
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "reply_to_message",
                                "arguments": '{"id": 1, "text": 42}',
                            },
                        }],
                    },
                }],
                "usage": {},
            })

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", CountingClient)
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)
    bot = _SuccessfulBot()

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, True,
    ))

    assert len(posts) == 1
    # Пустой текст — отправлять нечего, «42» в чат не уходит.
    assert bot.messages == []
    assert [entry["role"] for entry in history] == ["user"]


def test_group_streaming_does_not_leave_partial_message_after_ambiguous_timeout(
    monkeypatch, tmp_path
):
    async def fake_stream(client, url, *, payload, headers, on_content):
        await on_content("оборванный preview")
        return StreamedCompletionResponse(
            status_code=200,
            data={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "полный ответ"},
                }],
                "usage": {},
            },
        )

    class AmbiguousTimeoutMessage(_Message):
        async def reply_text(self, text, **kwargs):
            self.replies.append(text)
            raise TimeoutError("Telegram accepted the message but timed out")

    response = _Response(500)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _client_returning(response))
    monkeypatch.setattr(llm_client, "stream_chat_completion", fake_stream)
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", True)
    monkeypatch.setattr(llm_client, "STREAM_UPDATE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(llm_client, "STREAM_PREVIEW_MIN_CHARS", 1)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "group_-100"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories[key] = history
    state.chat_tokens.pop(key, None)
    bot = _SuccessfulBot()
    update = _Update()
    update.effective_chat.type = "supergroup"
    update.message = AmbiguousTimeoutMessage()

    asyncio.run(llm_client.send_llm_request(
        update, _Context(bot), key, history, "Миша", 1, True,
    ))

    assert update.message.replies == []
    assert [message["text"] for message in bot.messages] == ["полный ответ"]
    assert history[-1]["content"] == "полный ответ"


def _sequenced_client(posts, responses):
    """A client that returns canned responses in order and records the payloads."""

    class SequencedClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            posts.append(kwargs.get("json"))
            return responses[min(len(posts) - 1, len(responses) - 1)]

    return SequencedClient


def _tool_call_response(name, arguments):
    return _Response(200, {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            },
        }],
        "usage": {},
    })


class _ReactingBot(_SuccessfulBot):
    def __init__(self):
        super().__init__()
        self.reactions = []

    async def set_message_reaction(self, **kwargs):
        self.reactions.append(kwargs)


def _run_turn_with_tool(monkeypatch, tmp_path, name, arguments):
    posts = []
    responses = [
        _tool_call_response(name, arguments),
        _Response(200, {
            "choices": [{"finish_reason": "stop", "message": {"content": "ответ"}}],
            "usage": {},
        }),
    ]
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _sequenced_client(posts, responses))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_ReactingBot()), key, history, "Миша", 1, True,
    ))
    assert len(posts) == 2, posts
    return posts, history


def test_reaction_round_keeps_the_warm_temperature(monkeypatch, tmp_path):
    """Tool rounds retain the shipped Muse sampling parameters."""
    posts, _ = _run_turn_with_tool(
        monkeypatch, tmp_path, "react_to_message", '{"emoji": "\\ud83d\\udd25"}'
    )
    assert posts[1]["temperature"] == llm_client.GENERATION_PARAMS["temperature"]


def test_search_round_keeps_temperature_when_reasoning_is_on(monkeypatch, tmp_path):
    """Grounded Muse rounds keep the same temperature and reasoning baseline."""
    from berangaria.tools import dispatch as tool_handlers

    monkeypatch.setattr(
        tool_handlers, "web_search",
        lambda query, max_results=5, timelimit=None, region="ru-ru": "1. факт\nтекст\nhttps://e.com",
    )
    posts, history = _run_turn_with_tool(
        monkeypatch, tmp_path, "web_search", '{"query": "курс евро"}'
    )
    assert posts[1]["temperature"] == llm_client.GENERATION_PARAMS["temperature"]
    assert posts[1]["reasoning"] == {"effort": "low"}
    provider_messages = history[-1]["provider_messages"]
    assert [message["role"] for message in provider_messages] == [
        "assistant", "tool", "assistant",
    ]
    assert provider_messages[0]["tool_calls"][0]["function"]["name"] == "web_search"
    assert provider_messages[1]["tool_call_id"] == "call_1"
    assert provider_messages[-1]["content"] == "ответ"

def test_exhausted_read_url_is_removed_from_following_provider_round(monkeypatch, tmp_path):
    from berangaria.tools import dispatch as tool_handlers

    monkeypatch.setattr(tool_handlers, "READ_URL_MAX_PER_TURN", 1)
    monkeypatch.setattr(tool_handlers, "WEB_TOOL_MAX_PER_TURN", 6)
    monkeypatch.setattr(tool_handlers, "read_url", lambda url: "page")

    posts, _ = _run_turn_with_tool(
        monkeypatch, tmp_path, "read_url", '{"url": "https://e.com"}'
    )

    first_names = {
        tool["function"]["name"] for tool in posts[0]["tools"]
    }
    continuation_names = {
        tool["function"]["name"] for tool in posts[1]["tools"]
    }
    assert "read_url" in first_names
    assert "read_url" not in continuation_names
    assert "web_search" in continuation_names


def test_exhausted_combined_web_budget_preserves_addressed_terminal_tools(monkeypatch, tmp_path):
    from berangaria.tools import dispatch as tool_handlers

    posts = []
    responses = [
        _tool_call_response("web_search", '{"query": "курс евро"}'),
        _tool_call_response("read_url", '{"url": "https://e.com"}'),
        _tool_call_response(
            "send_messages", '{"messages": [{"text": "ответ", "reply_to": 1}]}',
        ),
    ]
    monkeypatch.setattr(tool_handlers, "WEB_SEARCH_MAX_PER_TURN", 1)
    monkeypatch.setattr(tool_handlers, "READ_URL_MAX_PER_TURN", 1)
    monkeypatch.setattr(tool_handlers, "WEB_TOOL_MAX_PER_TURN", 2)
    monkeypatch.setattr(
        tool_handlers,
        "web_search",
        lambda query, max_results=5, timelimit=None, region="ru-ru": "1. факт",
    )
    monkeypatch.setattr(tool_handlers, "read_url", lambda url: "page")
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _sequenced_client(posts, responses))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{"role": "user", "content": "[Message: курс евро]", "sid": 1, "mid": 10}]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)

    bot = _SuccessfulBot()
    outcome = asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, True,
    ))

    assert outcome is TurnOutcome.DELIVERED
    assert len(posts) == 3
    assert all(post.get("tool_choice") != "none" for post in posts)
    final_tool_names = {
        tool["function"]["name"] for tool in posts[2]["tools"]
    }
    initial_tool_names = {tool["function"]["name"] for tool in posts[0]["tools"]}
    assert final_tool_names == initial_tool_names - {"web_search", "read_url"}
    assert {"send_messages", "reply_to_message"} <= final_tool_names
    assert [message["text"] for message in bot.messages] == ["ответ"]
    assert bot.messages[0]["reply_to_message_id"] == 10
    assert history[-1]["content"] == "ответ"
    assert history[-1]["telegram_messages"] == [
        {"text": "ответ", "mid": 99, "reply_mid": 10, "reply_sid": 1},
    ]


class _AddressedBot(_SuccessfulBot):
    def __init__(self, *, failure=None):
        super().__init__()
        self.failure = failure

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        if self.failure is not None and len(self.messages) == 2:
            raise self.failure
        return SimpleNamespace(message_id=700 + len(self.messages))


@pytest.fixture
def addressed_turn(monkeypatch, isolated_db):
    """Exercise real dispatch, delivery and SQLite persistence with fake I/O."""
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(llm_client, "_multi_message_delay_seconds", lambda *a, **kw: 0)
    monkeypatch.setattr(memory_store, "memory", None)
    key = "group_100"
    history = [
        {"role": "user", "content": "[Message: Первый вопрос]", "sid": 1, "mid": 42},
        {"role": "user", "content": "[Message: Второй вопрос]", "sid": 2, "mid": 43},
        {"role": "user", "content": "[Message: Беран, ответь]", "sid": 3, "mid": 10},
    ]
    state.histories[key] = history
    update = _Update()
    update.effective_chat.type = "supergroup"

    def run(batches, *, bot=None):
        posts = []
        responses = []
        for index, batch in enumerate(batches):
            response = _tool_call_response(
                "send_messages", json.dumps({"messages": batch}, ensure_ascii=False),
            )
            response.json()["choices"][0]["message"]["tool_calls"][0]["id"] = f"batch_{index}"
            responses.append(response)
        monkeypatch.setattr(
            llm_client.httpx, "AsyncClient", _sequenced_client(posts, responses),
        )
        bot = bot if bot is not None else _AddressedBot()
        outcome = asyncio.run(llm_client.send_llm_request(
            update, _Context(bot), key, history, "Миша", 1, True,
        ))
        assert outcome is TurnOutcome.DELIVERED
        assert len(posts) == len(responses), "terminal delivery must not retry the LLM turn"
        assert len(history) == 4
        entry = copy.deepcopy(history[-1])
        assert entry["role"] == "assistant"
        state.histories.clear()
        state.load_all_histories()
        assert state.histories[key][-1] == entry
        return posts, bot, entry

    return run


@pytest.mark.parametrize("count", [1, 2, 5])
def test_structured_send_messages_preserves_independent_targets(addressed_turn, count):
    batch = [
        {"text": f"Ответ {index}.", "reply_to": 1 + index % 2}
        for index in range(count)
    ]

    _, bot, entry = addressed_turn([batch])

    texts = [f"Ответ {index}" for index in range(count)]
    targets = [42 + index % 2 for index in range(count)]
    assert [message["text"] for message in bot.messages] == texts
    assert [message["reply_to_message_id"] for message in bot.messages] == targets
    assert all(message["allow_sending_without_reply"] is True for message in bot.messages)
    assert entry["content"] == "\n".join(texts)
    assert entry["mid"] == 701
    assert entry["telegram_messages"] == [
        {"text": text, "mid": 701 + index, "reply_mid": targets[index],
         "reply_sid": batch[index]["reply_to"]}
        for index, text in enumerate(texts)
    ]
    trace = entry["provider_messages"]
    assert [message["role"] for message in trace] == ["assistant", "tool"]
    assert json.loads(trace[0]["tool_calls"][0]["function"]["arguments"])["messages"] == batch
    assert trace[-1]["tool_call_id"] == "batch_0"
    assert f"подтвердил первые {count} из {count}" in trace[-1]["content"]
    assert "Не подтверждено: 0" in trace[-1]["content"]
    assert llm_client._render_history_for_api([entry]) == trace
    overview = analytics_store.get_overview("all", chat_id=100)
    assert overview["assistant_replies"] == 1, "one delivered turn, not one per bubble"


def test_structured_send_messages_omitted_target_stays_standalone_when_mentioned(addressed_turn):
    _, bot, entry = addressed_turn([[
        {"text": "Общее начало"},
        {"text": "Первому", "reply_to": 1},
        {"text": "Общий конец"},
    ]])

    assert [message["text"] for message in bot.messages] == [
        "Общее начало", "Первому", "Общий конец",
    ]
    for index in (0, 2):
        assert "reply_to_message_id" not in bot.messages[index]
        assert "allow_sending_without_reply" not in bot.messages[index]
        assert entry["telegram_messages"][index]["reply_mid"] is None
        assert entry["telegram_messages"][index]["reply_sid"] is None
    assert bot.messages[1]["reply_to_message_id"] == 42
    assert entry["telegram_messages"][1]["reply_sid"] == 1


@pytest.mark.parametrize("error", [TimedOut(), BadRequest("Message to be replied not found")])
def test_structured_send_messages_partial_failure_persists_only_confirmed(addressed_turn, error):
    batch = [
        {"text": "Подтверждённый ответ.", "reply_to": 1},
        {"text": "Неподтверждённый ответ", "reply_to": 2},
        {"text": "Не отправлять"},
    ]
    _, bot, entry = addressed_turn([batch], bot=_AddressedBot(failure=error))

    assert len(bot.messages) == 2
    assert [message["reply_to_message_id"] for message in bot.messages] == [42, 43]
    assert entry["content"] == "Подтверждённый ответ"
    assert entry["mid"] == 701
    assert entry["telegram_messages"] == [
        {"text": "Подтверждённый ответ", "mid": 701, "reply_mid": 42, "reply_sid": 1},
    ]
    trace = entry["provider_messages"]
    assert [message["role"] for message in trace] == ["assistant", "tool"]
    # The exact attempted batch remains provider state, not confirmed display text.
    assert json.loads(trace[0]["tool_calls"][0]["function"]["arguments"])["messages"] == batch
    result = trace[-1]
    assert result["tool_call_id"] == "batch_0"
    assert "подтвердил первые 1 из 3" in result["content"]
    assert "Не подтверждено: 2" in result["content"]
    assert "Сообщения доставлены в Telegram" not in result["content"]
    assert llm_client._render_history_for_api([entry]) == trace


def test_bubble_cleaned_to_nothing_does_not_break_the_rest(
    addressed_turn, monkeypatch,
):
    original_clean = llm_client._clean_reply

    def drop_first(text):
        cleaned = original_clean(text)
        return "" if cleaned == "Пустой" else cleaned

    monkeypatch.setattr(llm_client, "_clean_reply", drop_first)

    _, bot, entry = addressed_turn([[
        {"text": "Пустой"},
        {"text": "Второй ответ", "reply_to": 2},
    ]])

    assert [message["text"] for message in bot.messages] == ["Второй ответ"]
    assert bot.messages[0]["reply_to_message_id"] == 43
    assert entry["telegram_messages"] == [
        {"text": "Второй ответ", "mid": 701, "reply_mid": 43, "reply_sid": 2},
    ]


def test_structured_send_messages_invalid_target_errors_before_send_then_corrects(
    addressed_turn, monkeypatch,
):
    posts_at_send = []
    original_dispatch = llm_client.dispatch_tool_call

    async def checked_dispatch(turn, payload_messages, update, context, tool_call, *args):
        if tool_call["id"] == "batch_1":
            assert context.bot.messages == [], "invalid batch must not partially send"
            error = payload_messages[-2]
            assert error["role"] == "tool"
            assert error["tool_call_id"] == "batch_0"
            assert "reply_to" in error["content"]
            posts_at_send.append("correction")
        return await original_dispatch(turn, payload_messages, update, context, tool_call, *args)

    monkeypatch.setattr(llm_client, "dispatch_tool_call", checked_dispatch)
    invalid = [{"text": "Не отправлять", "reply_to": 1},
               {"text": "Неверная цель", "reply_to": 999}]
    corrected = [{"text": "Исправлено", "reply_to": 2}]

    _, bot, entry = addressed_turn([invalid, corrected])

    assert posts_at_send == ["correction"]
    assert [message["text"] for message in bot.messages] == ["Исправлено"]
    assert bot.messages[0]["reply_to_message_id"] == 43
    assert entry["content"] == "Исправлено"
    assert entry["telegram_messages"] == [
        {"text": "Исправлено", "mid": 701, "reply_mid": 43, "reply_sid": 2},
    ]
    trace = entry["provider_messages"]
    assert [message["role"] for message in trace] == ["assistant", "tool", "assistant", "tool"]
    assert trace[1]["tool_call_id"] == "batch_0"
    assert "reply_to" in trace[1]["content"]
    assert trace[3]["tool_call_id"] == "batch_1"
    assert "Не подтверждено: 0" in trace[3]["content"]
    assert llm_client._render_history_for_api([entry]) == trace


def test_successful_tool_round_resets_transport_retry_budget(monkeypatch, tmp_path, caplog):
    from berangaria.tools import dispatch as tool_handlers

    caplog.set_level(logging.WARNING, logger="berangaria.chat.llm_client")
    posts = []
    rate_limits = [
        _Response(429, {
            "id": f"gen-rate-{index}",
            "error": {
                "code": 429,
                "message": "Rate limit exceeded",
                "metadata": {
                    "error_type": "rate_limit_exceeded",
                    "provider_code": "rate_limited",
                },
            },
        })
        for index in range(5)
    ]
    responses = [
        rate_limits[0],
        _tool_call_response("web_search", '{"query": "курс евро"}'),
        *rate_limits[1:],
        _Response(200, {
            "choices": [{"finish_reason": "stop", "message": {"content": "ответ"}}],
            "usage": {},
        }),
    ]
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(
        tool_handlers,
        "web_search",
        lambda query, max_results=5, timelimit=None, region="ru-ru": "1. факт",
    )
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _sequenced_client(posts, responses))
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_client.random, "uniform", lambda low, high: 1.0)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{"role": "user", "content": "[Message: курс евро]", "sid": 1, "mid": 10}]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)

    asyncio.run(llm_client._run_llm_turn(
        _Update(), _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
        chat_actions=None,
    ))

    assert len(posts) == 7
    assert sleeps == [5.0, 5.0, 10.0, 20.0, 30.0]
    assert history[-1]["content"] == "ответ"
    assert "provider_code=rate_limited" in caplog.text
    assert "tool_rounds=1 web_search=1 read_url=0" in caplog.text
    assert "payload_chars=" in caplog.text


def test_terminal_reply_persists_valid_exact_provider_trace(monkeypatch, tmp_path):
    posts = []
    response = _tool_call_response(
        "reply_to_message",
        '{"id":1,"text":"Адресный ответ."}',
    )
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        _sequenced_client(posts, [response]),
    )
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", False)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{
        "role": "user",
        "content": "[Message: привет]",
        "sid": 1,
        "mid": 10,
    }]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)
    bot = _SuccessfulBot()

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, True,
    ))

    assert len(posts) == 1
    assert bot.messages[0]["text"] == "Адресный ответ"
    assert history[-1]["content"] == "Адресный ответ"
    provider_messages = history[-1]["provider_messages"]
    assert [message["role"] for message in provider_messages] == ["assistant", "tool"]
    assert provider_messages[0]["tool_calls"][0]["function"]["arguments"].endswith(
        '"Адресный ответ."}'
    )
    assert provider_messages[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Ответ доставлен в Telegram. Ход завершён.",
    }


def test_streamed_reasoning_details_are_echoed_after_search(monkeypatch, tmp_path):
    from berangaria.tools import dispatch as tool_handlers

    reasoning_details = [
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
    ]
    responses = [
        StreamedCompletionResponse(
            status_code=200,
            data={
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_details": reasoning_details,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"курс евро"}',
                            },
                        }],
                    },
                }],
                "usage": {},
            },
        ),
        StreamedCompletionResponse(
            status_code=200,
            data={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ответ"},
                }],
                "usage": {},
            },
        ),
    ]
    payloads = []

    async def fake_stream(client, url, *, payload, headers, on_content=None):
        payloads.append(copy.deepcopy(payload))
        return responses[len(payloads) - 1]

    monkeypatch.setattr(
        tool_handlers,
        "web_search",
        lambda query, max_results=5, timelimit=None, region="ru-ru": (
            "1. факт\nтекст\nhttps://e.com"
        ),
    )
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        _client_returning(_Response(500)),
    )
    monkeypatch.setattr(llm_client, "stream_chat_completion", fake_stream)
    monkeypatch.setattr(llm_client, "STREAMING_ENABLED", True)
    monkeypatch.setattr(memory_store, "memory", None)
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "state.db"))
    state.init_db()
    key = "private_1"
    history = [{"role": "user", "content": "[Message: курс евро]", "sid": 1, "mid": 10}]
    state.histories.clear()
    state.histories[key] = history
    state.chat_tokens.pop(key, None)

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
    ))

    assert len(payloads) == 2
    continuation = payloads[1]["messages"]
    assistant = next(
        message
        for message in continuation
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant["reasoning_details"] == reasoning_details
    assert "reasoning_content" not in assistant
    assert any(message.get("role") == "tool" for message in continuation)
    assert payloads[1]["temperature"] == llm_client.GENERATION_PARAMS["temperature"]
