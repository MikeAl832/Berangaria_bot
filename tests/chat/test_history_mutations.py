from berangaria.chat.history_mutations import (
    apply_user_message_delete,
    apply_user_message_edit,
    is_history_row_mutable,
)


def test_edit_updates_unsent_history_row():
    history = [{
        "role": "user",
        "content": "[Owner: Миша] [Time: 01:00] [Message: старый текст]",
        "mid": 10,
        "provider_sent": False,
        "telegram_messages": [{"mid": 10, "text": "старый текст"}],
    }]

    assert apply_user_message_edit(
        history, message_id=10, new_text="новый текст", is_group=True
    ) == "updated"
    assert history[0]["telegram_messages"][0]["text"] == "новый текст"
    assert "[Message: новый текст]" in history[0]["content"]
    assert is_history_row_mutable(history[0])


def test_edit_frozen_after_provider_send():
    history = [{
        "role": "user",
        "content": "[Owner: Миша] [Time: 01:00] [Message: старый]",
        "mid": 10,
        "provider_sent": True,
        "telegram_messages": [{"mid": 10, "text": "старый"}],
    }]
    before = history[0]["content"]
    assert apply_user_message_edit(
        history, message_id=10, new_text="новый", is_group=True
    ) == "frozen"
    assert history[0]["content"] == before


def test_edit_missing_legacy_sent_row_is_frozen():
    history = [{
        "role": "user",
        "content": "[Owner: Миша] [Time: 01:00] [Message: старый]",
        "mid": 10,
        "telegram_messages": [{"mid": 10, "text": "старый"}],
    }]
    assert apply_user_message_edit(
        history, message_id=10, new_text="новый", is_group=True
    ) == "frozen"


def test_delete_removes_unsent_single_message_row():
    history = [{
        "role": "user",
        "content": "[Owner: Миша] [Time: 01:00] [Message: bye]",
        "mid": 10,
        "provider_sent": False,
        "telegram_messages": [{"mid": 10, "text": "bye"}],
    }]
    assert apply_user_message_delete(
        history, message_id=10, is_group=True
    ) == "removed"
    assert history == []


def test_delete_updates_merged_buffer_row():
    history = [{
        "role": "user",
        "content": "[Owner: Миша] [Time: 01:00] [Message: one\ntwo]",
        "mid": 11,
        "provider_sent": False,
        "telegram_messages": [
            {"mid": 10, "text": "one"},
            {"mid": 11, "text": "two"},
        ],
    }]
    assert apply_user_message_delete(
        history, message_id=10, is_group=True
    ) == "updated"
    assert history[0]["telegram_messages"] == [{"mid": 11, "text": "two"}]
    assert history[0]["mid"] == 11
    assert "[Message: two]" in history[0]["content"]


def test_delete_unsent_assistant_row():
    from berangaria.chat.history_mutations import apply_history_message_delete

    history = [{
        "role": "assistant",
        "content": "ответ",
        "mid": 55,
        "provider_sent": False,
    }]
    assert apply_history_message_delete(
        history, message_id=55, is_group=True
    ) == "removed"
    assert history == []


def test_delete_frozen_assistant_row():
    from berangaria.chat.history_mutations import apply_history_message_delete

    history = [{
        "role": "assistant",
        "content": "ответ",
        "mid": 55,
        "provider_sent": True,
    }]
    assert apply_history_message_delete(
        history, message_id=55, is_group=True
    ) == "frozen"
    assert len(history) == 1


def _assistant_burst(*, provider_sent=False):
    return {
        "role": "assistant",
        "content": "первый\nвторой\nтретий",
        "mid": 55,
        "provider_sent": provider_sent,
        "telegram_messages": [
            {"mid": 55, "text": "первый", "reply_mid": None, "reply_sid": None},
            {"mid": 56, "text": "второй", "reply_mid": 42, "reply_sid": 1},
            {"mid": 57, "text": "третий", "reply_mid": None, "reply_sid": None},
        ],
    }


def test_deleting_one_burst_bubble_keeps_the_others():
    from berangaria.chat.history_mutations import apply_history_message_delete

    history = [_assistant_burst()]

    assert apply_history_message_delete(
        history, message_id=56, is_group=True
    ) == "updated"
    assert history[0]["content"] == "первый\nтретий"
    assert [item["mid"] for item in history[0]["telegram_messages"]] == [55, 57]


def test_deleting_the_row_anchor_bubble_keeps_the_rest():
    from berangaria.chat.history_mutations import apply_history_message_delete

    history = [_assistant_burst()]

    assert apply_history_message_delete(
        history, message_id=55, is_group=True
    ) == "updated"
    assert history[0]["mid"] == 56
    assert history[0]["content"] == "второй\nтретий"


def test_deleting_the_last_burst_bubble_removes_the_row():
    from berangaria.chat.history_mutations import apply_history_message_delete

    history = [{
        "role": "assistant",
        "content": "единственный",
        "mid": 55,
        "provider_sent": False,
        "telegram_messages": [{"mid": 55, "text": "единственный"}],
    }]

    assert apply_history_message_delete(
        history, message_id=55, is_group=True
    ) == "removed"
    assert history == []


def test_sent_burst_bubble_is_frozen_before_rewrite():
    from berangaria.chat.history_mutations import apply_history_message_delete

    history = [_assistant_burst(provider_sent=True)]
    before = [dict(item) for item in history[0]["telegram_messages"]]

    assert apply_history_message_delete(
        history, message_id=56, is_group=True
    ) == "frozen"
    assert history[0]["telegram_messages"] == before
    assert history[0]["content"] == "первый\nвторой\nтретий"
