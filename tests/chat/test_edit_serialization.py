import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from berangaria import app
from berangaria.chat import handlers
from berangaria.core import state


def make_update():
    return SimpleNamespace(edited_message=SimpleNamespace(
        chat=SimpleNamespace(id=-100, type="supergroup"),
        from_user=SimpleNamespace(id=42),
        message_id=10,
        text="новый текст",
        caption=None,
    ))


@pytest.mark.parametrize("freeze", [True, False])
def test_history_edit_waits_for_turn_and_rechecks_sent_flag(monkeypatch, freeze):
    key = state.get_history_key(-100, False, 42)
    row = {
        "role": "user",
        "content": "[User: Миша] [Message: старый текст]",
        "mid": 10,
        "provider_sent": False,
        "telegram_messages": [{"mid": 10, "text": "старый текст"}],
    }
    handlers.histories[key] = [row]
    save = Mock(return_value=True)
    update_source = Mock()
    monkeypatch.setattr(handlers, "save_history", save)
    monkeypatch.setattr(state, "update_memory_source_text", update_source)
    context = SimpleNamespace(bot=SimpleNamespace(id=999))

    async def run():
        lock = state.get_turn_lock(key)
        await lock.acquire()
        snapshot = copy.deepcopy(row)
        task = asyncio.create_task(handlers.handle_edited_message(make_update(), context))
        try:
            await asyncio.sleep(0)
            assert not task.done()
            assert row == snapshot
            save.assert_not_called()
            row["provider_sent"] = freeze
        finally:
            lock.release()
            await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())
    if freeze:
        assert row["telegram_messages"][0]["text"] == "старый текст"
        save.assert_not_called()
        update_source.assert_not_called()
    else:
        assert row["telegram_messages"][0]["text"] == "новый текст"
        save.assert_called_once_with(key)
        update_source.assert_called_once_with(
            scope=key, message_id=10, text="новый текст"
        )


def test_buffer_edit_does_not_wait_for_active_turn():
    key = state.get_history_key(-100, False, 42)
    message = {"message_id": 10, "text": "старый", "telegram_text": "старый"}
    handlers.message_buffer["-100_42"] = {"messages": [message]}
    context = SimpleNamespace(bot=SimpleNamespace(id=999))

    async def run():
        async with state.get_turn_lock(key):
            await asyncio.wait_for(
                handlers.handle_edited_message(make_update(), context), timeout=1
            )

    asyncio.run(run())
    assert message["text"] == "новый текст"
    assert message["telegram_text"] == "новый текст"


def test_edit_handler_does_not_block_update_processor():
    application = Mock()
    app.register_handlers(application)
    registered = [call.args[0] for call in application.add_handler.call_args_list]
    edit_handlers = [
        handler for handler in registered
        if handler.callback is handlers.handle_edited_message
    ]
    assert len(edit_handlers) == 1
    assert edit_handlers[0].block is False
