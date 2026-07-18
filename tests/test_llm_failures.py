import asyncio

import llm_client
import memory_store
import pytest
import state
from streaming import StreamedCompletionResponse


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
    async def fake_stream(client, url, *, payload, headers, on_content):
        await on_content("потоковый ответ")
        return StreamedCompletionResponse(
            status_code=200,
            data={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "потоковый ответ"},
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
    history = [{"role": "user", "content": "[Message: привет]", "sid": 1, "mid": 10}]
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
    assert history[-1]["mid"] == 99
