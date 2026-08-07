from berangaria.user_bridge.models import BridgeEventMeta
from berangaria.user_bridge.policy import (
    decide_bridge_event,
    message_mentions_bot,
    resolve_bridge_chat_ids,
)


def test_resolve_bridge_chat_ids_inherits_allowed_groups():
    assert resolve_bridge_chat_ids([], [-1001, -1002]) == (-1001, -1002)


def test_resolve_bridge_chat_ids_explicit_overrides():
    assert resolve_bridge_chat_ids([-1009], [-1001, -1002]) == (-1009,)


def test_reject_private_chat():
    d = decide_bridge_event(
        BridgeEventMeta(
            chat_id=1,
            message_id=10,
            is_group=False,
            sender_is_bot=True,
            sender_id=99,
            allowed_chat_ids=(-1001,),
        )
    )
    assert d.accept is False
    assert d.reason == "not_group"


def test_reject_human_sender():
    d = decide_bridge_event(
        BridgeEventMeta(
            chat_id=-1001,
            message_id=10,
            is_group=True,
            sender_is_bot=False,
            sender_id=42,
            allowed_chat_ids=(-1001,),
        )
    )
    assert d.accept is False
    assert d.reason == "not_bot"


def test_reject_self_bot():
    d = decide_bridge_event(
        BridgeEventMeta(
            chat_id=-1001,
            message_id=10,
            is_group=True,
            sender_is_bot=True,
            sender_id=777,
            our_bot_id=777,
            allowed_chat_ids=(-1001,),
        )
    )
    assert d.accept is False
    assert d.reason == "self_bot"


def test_reject_chat_outside_allowlist():
    d = decide_bridge_event(
        BridgeEventMeta(
            chat_id=-999,
            message_id=10,
            is_group=True,
            sender_is_bot=True,
            sender_id=5,
            allowed_chat_ids=(-1001,),
        )
    )
    assert d.accept is False
    assert d.reason == "chat_not_allowed"


def test_reject_empty_allowlist():
    d = decide_bridge_event(
        BridgeEventMeta(
            chat_id=-1001,
            message_id=10,
            is_group=True,
            sender_is_bot=True,
            sender_id=5,
            allowed_chat_ids=(),
        )
    )
    assert d.accept is False
    assert d.reason == "empty_allowlist"


def test_accept_other_bot_in_allowed_group():
    d = decide_bridge_event(
        BridgeEventMeta(
            chat_id=-1001,
            message_id=10,
            is_group=True,
            sender_is_bot=True,
            sender_id=5,
            our_bot_id=777,
            allowed_chat_ids=(-1001,),
        )
    )
    assert d.accept is True


def test_mention_by_username():
    assert message_mentions_bot(
        "hey @berangaria_bot hello",
        bot_id=1,
        bot_username="berangaria_bot",
        bot_first_name="Ber",
        reply_to_user_id=None,
        bot_names=["Бер"],
    )


def test_mention_by_reply_to_bot():
    assert message_mentions_bot(
        "ok",
        bot_id=42,
        bot_username="x",
        bot_first_name="Ber",
        reply_to_user_id=42,
        bot_names=[],
    )


def test_no_mention():
    assert not message_mentions_bot(
        "hello world",
        bot_id=1,
        bot_username="berangaria_bot",
        bot_first_name="Ber",
        reply_to_user_id=99,
        bot_names=["Бер"],
    )
