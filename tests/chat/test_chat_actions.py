import asyncio
from types import SimpleNamespace

from berangaria.chat import llm_client
from berangaria.chat.chat_actions import (
    ChatActionHeartbeat,
    effective_message_thread_id,
)


class _Chat:
    id = -100
    type = "supergroup"

    def __init__(self):
        self.actions = []

    async def send_action(self, **kwargs):
        self.actions.append(kwargs)


async def _wait_until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not reached")


def test_heartbeat_refreshes_switches_and_stops():
    chat = _Chat()

    async def run():
        heartbeat = ChatActionHeartbeat(
            chat,
            action="typing",
            message_thread_id=7,
            interval_seconds=0.002,
        )
        async with heartbeat:
            await _wait_until(lambda: len(chat.actions) >= 3)
            await heartbeat.set_action("choose_sticker")
            await _wait_until(
                lambda: sum(
                    item["action"] == "choose_sticker" for item in chat.actions
                ) >= 2
            )
        stopped_at = len(chat.actions)
        await asyncio.sleep(0.008)
        return stopped_at

    stopped_at = asyncio.run(run())

    assert [item["action"] for item in chat.actions[:3]] == [
        "typing",
        "typing",
        "typing",
    ]
    assert chat.actions[-1]["action"] == "choose_sticker"
    assert all(item["message_thread_id"] == 7 for item in chat.actions)
    assert len(chat.actions) == stopped_at


def test_send_llm_request_keeps_typing_for_the_selected_turn(monkeypatch):
    chat = _Chat()
    update = SimpleNamespace(
        message=SimpleNamespace(
            chat=chat,
            message_thread_id=None,
        ),
        effective_chat=chat,
    )
    context = SimpleNamespace()

    monkeypatch.setattr(
        llm_client,
        "ChatActionHeartbeat",
        lambda chat, **kwargs: ChatActionHeartbeat(
            chat, **kwargs, interval_seconds=0.002
        ),
    )

    async def fake_turn(*args, chat_actions, **kwargs):
        assert chat_actions.action == "typing"
        await _wait_until(lambda: len(chat.actions) >= 3)
        return "done"

    monkeypatch.setattr(llm_client, "_run_llm_turn", fake_turn)

    async def run():
        result = await llm_client.send_llm_request(
            update, context, "group_-100", [], "Миша", 42, True
        )
        stopped_at = len(chat.actions)
        await asyncio.sleep(0.008)
        return result, stopped_at

    result, stopped_at = asyncio.run(run())

    assert result == "done"
    assert len(chat.actions) == stopped_at
    assert {item["action"] for item in chat.actions} == {"typing"}


def test_effective_message_thread_id_only_for_forum_topics():
    assert effective_message_thread_id(None) is None
    assert effective_message_thread_id(SimpleNamespace(message_thread_id=None)) is None
    # Non-forum reply chain: Telegram may set message_thread_id without topic flag.
    assert (
        effective_message_thread_id(
            SimpleNamespace(message_thread_id=74910, is_topic_message=None)
        )
        is None
    )
    assert (
        effective_message_thread_id(
            SimpleNamespace(message_thread_id=74910, is_topic_message=False)
        )
        is None
    )
    assert (
        effective_message_thread_id(
            SimpleNamespace(message_thread_id=42, is_topic_message=True)
        )
        == 42
    )


def test_send_llm_request_ignores_non_topic_thread_id(monkeypatch):
    chat = _Chat()
    update = SimpleNamespace(
        message=SimpleNamespace(
            chat=chat,
            message_thread_id=74910,
            is_topic_message=None,
        ),
        effective_chat=chat,
    )
    context = SimpleNamespace()

    captured = {}

    class _CaptureHeartbeat(ChatActionHeartbeat):
        def __init__(self, chat, **kwargs):
            captured["message_thread_id"] = kwargs.get("message_thread_id")
            super().__init__(chat, **kwargs, interval_seconds=0.002)

    monkeypatch.setattr(llm_client, "ChatActionHeartbeat", _CaptureHeartbeat)

    async def fake_turn(*args, chat_actions, **kwargs):
        return "done"

    monkeypatch.setattr(llm_client, "_run_llm_turn", fake_turn)

    result = asyncio.run(
        llm_client.send_llm_request(
            update, context, "group_-100", [], "Миша", 42, True
        )
    )

    assert result == "done"
    assert captured["message_thread_id"] is None
