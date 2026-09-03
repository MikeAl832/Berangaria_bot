import asyncio
import copy

from berangaria.chat import llm_client
from berangaria.memory import store as memory_store
import pytest
from berangaria.core import state
from berangaria.chat.streaming import StreamedCompletionResponse
from berangaria.analytics import store as analytics_store


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = {}

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

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(bot), key, history, "Миша", 1, False,
    ))

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


def test_api_400_persists_cleared_history(monkeypatch, tmp_path):
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

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_FailingBot()), key, history, "Миша", 1, True,
    ))

    state.histories.clear()
    state.load_all_histories()
    assert state.histories[key] == []


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
        "only": ["openai/flex"],
        "allow_fallbacks": False,
    }
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


def test_confirmed_reply_and_usage_are_recorded(monkeypatch, tmp_path):
    response = _Response(200, {
        "choices": [{"finish_reason": "stop", "message": {"content": "ответ"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
            "cost": 0.000321,
        },
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
        "author_id": 1,
        "author_name": "Миша",
    }]
    state.histories[key] = history

    asyncio.run(llm_client.send_llm_request(
        _Update(), _Context(_SuccessfulBot()), key, history, "Миша", 1, True,
    ))

    overview = analytics_store.get_overview("all", chat_id=100)
    assert overview["requests"] == 1
    assert overview["cost_microusd"] == 321
    assert overview["assistant_replies"] == 1
    leaders = analytics_store.get_leaderboards("all", chat_id=100)
    assert leaders["replies"][0]["user_id"] == 1
    assert leaders["cost"][0]["value"] == 321


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
    """The cold temperature is the price of retelling looked-up facts, not of any
    tool at all. Reactions and stickers must not flatten the rest of the turn."""
    posts, _ = _run_turn_with_tool(
        monkeypatch, tmp_path, "react_to_message", '{"emoji": "\\ud83d\\udd25"}'
    )
    assert posts[1]["temperature"] == llm_client.GENERATION_PARAMS["temperature"]
    assert posts[1]["temperature"] != llm_client.FACTUAL_TEMPERATURE


def test_search_round_keeps_temperature_when_reasoning_is_on(monkeypatch, tmp_path):
    """With effort other than none, OpenAI rejects a cooled temperature on Completions."""
    from berangaria.tools import dispatch as tool_handlers

    monkeypatch.setattr(
        tool_handlers, "web_search",
        lambda query, max_results=5, timelimit=None, region="ru-ru": "1. факт\nтекст\nhttps://e.com",
    )
    posts, history = _run_turn_with_tool(
        monkeypatch, tmp_path, "web_search", '{"query": "курс евро"}'
    )
    assert posts[1]["temperature"] == llm_client.GENERATION_PARAMS["temperature"]
    assert posts[1]["temperature"] != llm_client.FACTUAL_TEMPERATURE
    provider_messages = history[-1]["provider_messages"]
    assert [message["role"] for message in provider_messages] == [
        "assistant", "tool", "assistant",
    ]
    assert provider_messages[0]["tool_calls"][0]["function"]["name"] == "web_search"
    assert provider_messages[1]["tool_call_id"] == "call_1"
    assert provider_messages[-1]["content"] == "ответ"


def test_search_round_cools_when_reasoning_is_off(monkeypatch, tmp_path):
    from berangaria.tools import dispatch as tool_handlers

    monkeypatch.setattr(
        llm_client,
        "GENERATION_PARAMS",
        {**llm_client.GENERATION_PARAMS, "reasoning": {"effort": "none"}},
    )
    monkeypatch.setattr(
        tool_handlers, "web_search",
        lambda query, max_results=5, timelimit=None, region="ru-ru": "1. факт\nтекст\nhttps://e.com",
    )
    posts, _ = _run_turn_with_tool(
        monkeypatch, tmp_path, "web_search", '{"query": "курс евро"}'
    )
    assert posts[1]["temperature"] == llm_client.FACTUAL_TEMPERATURE


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
