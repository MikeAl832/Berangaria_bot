import asyncio
from types import SimpleNamespace

import pytest

from berangaria.chat import handlers
from berangaria.chat.turn_outcome import TurnOutcome
from berangaria.core import state


@pytest.mark.parametrize(
    "outcome, expected_status",
    [(TurnOutcome.DELIVERED, "pending"), (TurnOutcome.FAILED, "abandoned"),
     (TurnOutcome.SILENT, "pending"), (None, "abandoned")],
)
def test_queue_releases_delivery_and_intentional_silence(monkeypatch, isolated_db, outcome, expected_status):
    monkeypatch.setattr(handlers, "MESSAGE_DEBOUNCE_SECONDS", 0)
    monkeypatch.setattr(handlers, "_check_access_permissions", lambda *args: True)
    monkeypatch.setattr(handlers, "is_bot_mentioned", lambda *args: (True, ""))
    monkeypatch.setattr(handlers, "release_memory_sources", state.release_memory_sources)
    monkeypatch.setattr(handlers, "abandon_memory_sources", state.abandon_memory_sources)

    async def llm(*args, **kwargs):
        return outcome

    monkeypatch.setattr(handlers, "send_llm_request", llm)
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=-100, type="supergroup"),
        effective_user=SimpleNamespace(id=42, first_name="User"),
        message=SimpleNamespace(
            message_id=901, date=None, forward_origin=None, reply_to_message=None,
        ),
    )

    async def run():
        await handlers.queue_message(
            update, SimpleNamespace(), "Я постоянно использую Fedora"
        )
        await state.message_buffer["-100_42"]["task"]

    asyncio.run(run())
    sources = state.list_memory_sources()
    assert len(sources) == 1
    assert sources[0].status == expected_status
    assert (-100 in state.bot_presence_started_at) is (outcome is TurnOutcome.DELIVERED)
    if expected_status == "abandoned":
        assert sources[0].text == ""