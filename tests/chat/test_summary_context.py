import asyncio
import copy
import json

import pytest

from berangaria.chat import summarization


@pytest.mark.parametrize("line", [
    "account: person@example.invalid | fake-password | SYNTHETIC2FASECRET",
    "password: synthetic-password",
    "api_key=synthetic-key",
    "секрет 2FA: synthetic-secret",
    "otpauth://totp/test?secret=SYNTHETIC",
])
def test_credential_lines_are_removed(line):
    assert summarization._redact_credentials("тема\n" + line + "\nрешение") == (
        "тема\n[данные доступа удалены]\nрешение"
    )


def test_summary_request_separates_recent_context_and_filters_secrets(monkeypatch, caplog):
    captured = {}
    credential = "user@example.invalid | fake-password | SYNTHETICSECRET"
    secret_output = "password: synthetic-output-secret"

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json, headers):
            captured.update(json)
            return summarization.httpx.Response(200, json={
                "choices": [{"message": {"content": "Открытый вопрос\n" + secret_output}}],
            }, request=summarization.httpx.Request("POST", url))

    monkeypatch.setattr(summarization.httpx, "AsyncClient", Client)
    monkeypatch.setattr(summarization, "SUMMARY_INTERVAL", 1)
    history = [
        {"role": "user", "content": "[Previous conversation summary: давняя тема\n" + credential + "]"},
        {"role": "assistant", "content": "старый ответ", "provider_messages": [{"content": "OPAQUE_TOOL"}],
         "reasoning_details": [{"text": "OPAQUE_REASONING"}]},
        {"role": "event", "content": "SERVICE_EVENT"},
        {"role": "user", "content": "актуальный вопрос", "sid": 7},
    ]
    before = copy.deepcopy(history)
    result = asyncio.run(summarization.summarize_history(history))
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"
    assert "fake-password" not in captured["messages"][1]["content"]
    assert "Прежнее резюме можно и нужно сокращать" in captured["messages"][0]["content"]
    body = json.loads(captured["messages"][1]["content"])
    assert "давняя тема" in body["older_history"]
    assert "актуальный вопрос" not in body["older_history"]
    assert "актуальный вопрос" in body["recent_context"]
    assert body["as_of"]
    payload = json.dumps(captured)
    for excluded in ("fake-password", "SYNTHETICSECRET", "OPAQUE_TOOL", "OPAQUE_REASONING", "SERVICE_EVENT"):
        assert excluded not in payload
    assert "synthetic-output-secret" not in result[0]["content"]
    assert "synthetic-output-secret" not in caplog.text
    assert history == before
    assert result[-1]["content"] == history[-1]["content"]
    assert result[-1] is not history[-1]
    assert captured["reasoning"] == {"effort": "high"}
    assert captured["max_tokens"] == 8192


def test_summary_length_cap_preserves_short_text():
    assert summarization._bound_summary("короткое резюме", 100) == "короткое резюме"
    long = "Открытый вопрос\n" + "детали " * 40
    bounded = summarization._bound_summary(long, 100)
    assert len(bounded) <= 100
    assert long.startswith(bounded)
