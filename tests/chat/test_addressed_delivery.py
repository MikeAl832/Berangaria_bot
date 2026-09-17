import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from telegram.error import BadRequest, TimedOut

from berangaria.chat.outgoing_message import DeliveredMessage, OutgoingMessage
from berangaria.chat.reply_delivery import DeliveryRuntime, deliver_addressed
from berangaria.chat.reply_formatting import is_parse_error


def _runtime(*, thread_id=None):
    bot = SimpleNamespace(send_message=AsyncMock())
    bot.send_message.side_effect = [
        SimpleNamespace(message_id=701),
        SimpleNamespace(message_id=702),
        SimpleNamespace(message_id=703),
    ]
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=100),
        message=SimpleNamespace(
            message_id=9,
            message_thread_id=thread_id,
            is_topic_message=thread_id is not None,
            chat=SimpleNamespace(send_action=AsyncMock()),
        ),
    )
    return DeliveryRuntime(
        update=update,
        context=SimpleNamespace(bot=bot),
        clean_reply=Mock(side_effect=AssertionError("Cleanup belongs upstream")),
        is_parse_error=is_parse_error,
        multi_message_delay_seconds=Mock(return_value=0),
    )


def test_two_people_receive_individually_addressed_bubbles():
    runtime = _runtime()
    messages = [OutgoingMessage("**Alice.**", 42, 2), OutgoingMessage("Bob.", 43, 3)]

    receipts = asyncio.run(deliver_addressed(messages, None, runtime))

    assert receipts == [
        DeliveredMessage("**Alice.**", 701, 42, 2),
        DeliveredMessage("Bob.", 702, 43, 3),
    ]
    calls = runtime.context.bot.send_message.call_args_list
    assert [call.kwargs["reply_to_message_id"] for call in calls] == [42, 43]
    assert all(call.kwargs["allow_sending_without_reply"] is True for call in calls)
    assert calls[0].kwargs["text"] == "<b>Alice.</b>"
    runtime.clean_reply.assert_not_called()


def test_unaddressed_bubbles_never_inherit_incoming_or_previous_target():
    runtime = _runtime()
    messages = [OutgoingMessage("First"), OutgoingMessage("Second", 42, 2),
                OutgoingMessage("Third")]

    receipts = asyncio.run(deliver_addressed(messages, None, runtime))

    for index in (0, 2):
        kwargs = runtime.context.bot.send_message.call_args_list[index].kwargs
        assert "reply_to_message_id" not in kwargs
        assert "allow_sending_without_reply" not in kwargs
        assert receipts[index].reply_mid is None
        assert receipts[index].reply_sid is None


@pytest.mark.parametrize("error", [TimedOut(), BadRequest("Message to be replied not found")])
def test_partial_failure_returns_only_exact_confirmed_receipts(error):
    runtime = _runtime()
    runtime.context.bot.send_message.side_effect = [SimpleNamespace(message_id=812), error]
    messages = [OutgoingMessage("**First.**", 42, 2), OutgoingMessage("Second", 43, 3),
                OutgoingMessage("Never sent", 44, 4)]

    receipts = asyncio.run(deliver_addressed(messages, None, runtime))

    assert receipts == [DeliveredMessage("**First.**", 812, 42, 2)]
    assert runtime.context.bot.send_message.await_count == 2


@pytest.mark.parametrize("error", [TimedOut(), BadRequest("Message to be replied not found")])
def test_first_failure_raises_without_retry_or_later_sends(error):
    runtime = _runtime()
    runtime.context.bot.send_message.side_effect = error

    with pytest.raises(type(error)) as caught:
        asyncio.run(deliver_addressed(
            [OutgoingMessage("First", 42), OutgoingMessage("Second", 43)], None, runtime
        ))

    assert caught.value is error
    runtime.context.bot.send_message.assert_awaited_once()


def test_html_fallback_preserves_target_and_forum_thread():
    runtime = _runtime(thread_id=55)
    runtime.context.bot.send_message.side_effect = [
        BadRequest("Can't parse entities"), SimpleNamespace(message_id=801)
    ]

    receipts = asyncio.run(deliver_addressed([OutgoingMessage("**Hello**", 42, 2)],
                                           None, runtime))

    assert receipts == [DeliveredMessage("**Hello**", 801, 42, 2)]
    html, plain = [call.kwargs for call in runtime.context.bot.send_message.call_args_list]
    assert html == dict(chat_id=100, text="<b>Hello</b>", parse_mode="HTML",
                        message_thread_id=55, reply_to_message_id=42,
                        allow_sending_without_reply=True)
    assert plain == dict(chat_id=100, text="Hello", message_thread_id=55,
                         reply_to_message_id=42, allow_sending_without_reply=True)


def test_failed_plain_fallback_retains_prior_receipts_without_retry():
    runtime = _runtime()
    runtime.context.bot.send_message.side_effect = [
        SimpleNamespace(message_id=701), BadRequest("Can't parse entities"), TimedOut()
    ]

    receipts = asyncio.run(deliver_addressed(
        [OutgoingMessage("First", 42, 2), OutgoingMessage("**Second**", 43, 3),
         OutgoingMessage("Never sent")], None, runtime
    ))

    assert receipts == [DeliveredMessage("First", 701, 42, 2)]
    assert runtime.context.bot.send_message.await_count == 3


@pytest.mark.parametrize("messages", [
    [], [OutgoingMessage("")], [OutgoingMessage(" \n ")],
    [OutgoingMessage("Valid", 42), OutgoingMessage("x" * 4097, 43)],
    [OutgoingMessage("Valid", 42), OutgoingMessage("")],
])
def test_invalid_batch_is_rejected_before_any_side_effect(messages):
    runtime = _runtime()
    status = SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())

    with pytest.raises(ValueError):
        asyncio.run(deliver_addressed(messages, status, runtime))

    runtime.context.bot.send_message.assert_not_called()
    runtime.multi_message_delay_seconds.assert_not_called()
    status.delete.assert_not_called()
    status.edit_text.assert_not_called()


def test_status_deleted_before_sending_even_when_target_matches_incoming():
    runtime = _runtime()
    events = []

    async def delete():
        events.append("delete")

    async def send(**kwargs):
        events.append("send")
        return SimpleNamespace(message_id=701)

    status = SimpleNamespace(delete=AsyncMock(side_effect=delete), edit_text=AsyncMock())
    runtime.context.bot.send_message.side_effect = send

    asyncio.run(deliver_addressed([OutgoingMessage("Hello", 9)], status, runtime))

    assert events == ["delete", "send"]
    status.edit_text.assert_not_called()


def test_status_deletion_failure_is_best_effort():
    runtime = _runtime()
    status = SimpleNamespace(delete=AsyncMock(side_effect=TimedOut()))

    receipts = asyncio.run(deliver_addressed([OutgoingMessage("Hello")], status, runtime))

    assert receipts == [DeliveredMessage("Hello", 701)]
    status.delete.assert_awaited_once()


def test_pauses_are_sequential_and_track_total(monkeypatch):
    runtime = _runtime()
    runtime.multi_message_delay_seconds.side_effect = [0.2, 0.3]
    events = []

    async def send(**kwargs):
        events.append(kwargs["text"])
        return SimpleNamespace(message_id=len(events))

    async def sleep(delay):
        events.append(delay)

    runtime.context.bot.send_message.side_effect = send
    monkeypatch.setattr("berangaria.chat.reply_delivery.asyncio.sleep", sleep)

    asyncio.run(deliver_addressed(
        [OutgoingMessage("First"), OutgoingMessage("Second"), OutgoingMessage("Third")],
        None, runtime
    ))

    assert events == ["First", 0.2, "Second", 0.3, "Third"]
    delays = runtime.multi_message_delay_seconds.call_args_list
    assert delays[0].args == ("Second",)
    assert delays[0].kwargs == {"slept_total": 0.0}
    assert delays[1].args == ("Third",)
    assert delays[1].kwargs == {"slept_total": 0.2}
    assert runtime.update.message.chat.send_action.await_count == 2


def test_html_expansion_uses_one_plain_bubble_at_size_boundary():
    runtime = _runtime()
    text = "&" * 4096

    receipts = asyncio.run(deliver_addressed([OutgoingMessage(text, 42, 2)], None, runtime))

    assert receipts == [DeliveredMessage(text, 701, 42, 2)]
    runtime.context.bot.send_message.assert_awaited_once_with(
        chat_id=100, text=text, reply_to_message_id=42, allow_sending_without_reply=True
    )
