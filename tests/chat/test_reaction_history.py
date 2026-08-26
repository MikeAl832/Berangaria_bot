import asyncio
import copy

import pytest

from berangaria.chat import handlers, llm_client
from berangaria.chat.history_rendering import render_history_for_api
from berangaria.core import state


def _record(history, *, added=None, removed=None):
    return handlers._record_incoming_reaction_change(
        history,
        target_mid=77,
        actor_name="Миша",
        actor_id=42,
        added=added or [],
        removed=removed or [],
    )


def test_reaction_stays_next_to_assistant_before_first_provider_send():
    history = [{
        "role": "assistant",
        "content": "новый ответ",
        "mid": 77,
        "provider_sent": False,
    }]

    assert _record(history, added=["🔥"])
    assert len(history) == 1
    assert history[0]["incoming_reactions"] == [{
        "from": "Миша",
        "from_id": 42,
        "emoji": "🔥",
    }]

    assert _record(history, removed=["🔥"])
    assert len(history) == 1
    assert "incoming_reactions" not in history[0]


@pytest.mark.parametrize("provider_sent", [True, None])
def test_reaction_on_sent_or_legacy_assistant_appends_tail_event(provider_sent):
    target = {"role": "assistant", "content": "старый ответ", "mid": 77}
    if provider_sent is not None:
        target["provider_sent"] = provider_sent
    history = [target]
    before = copy.deepcopy(target)

    assert _record(history, added=["👍"], removed=["🔥"])

    assert target == before
    assert history[1] == {
        "role": "event",
        "event_kind": "incoming_reaction",
        "actor_name": "Миша",
        "actor_id": 42,
        "target_mid": 77,
        "target_excerpt": "старый ответ",
        "added": ["👍"],
        "removed": ["🔥"],
        "provider_sent": False,
    }


def test_tail_reaction_event_preserves_rendered_provider_prefix():
    history = [
        {"role": "user", "content": "привет", "provider_sent": True},
        {
            "role": "assistant",
            "content": "давно отправленный ответ",
            "mid": 77,
            "provider_sent": True,
        },
    ]
    before = render_history_for_api(history)

    assert _record(history, added=["❤️"])
    after = render_history_for_api(history)

    assert after[: len(before)] == before
    assert after[-1]["role"] == "system"
    assert "Миша" in after[-1]["content"]
    assert "❤️" in after[-1]["content"]
    assert "давно отправленный ответ" in after[-1]["content"]


def test_first_provider_send_boundary_is_persisted(isolated_db):
    key = "private_42"
    history = [
        {"role": "assistant", "content": "legacy", "mid": 1},
        {
            "role": "assistant",
            "content": "new",
            "mid": 2,
            "provider_sent": False,
        },
    ]
    state.histories[key] = history
    assert state.save_history(key)

    asyncio.run(llm_client._mark_history_sent_to_provider(history, key=key))

    assert "provider_sent" not in history[0]
    assert history[1]["provider_sent"] is True
    state.histories.clear()
    assert state.load_all_histories() == 1
    assert "provider_sent" not in state.histories[key][0]
    assert state.histories[key][1]["provider_sent"] is True


def test_failed_boundary_persistence_keeps_rows_mutable(monkeypatch):
    key = "private_42"
    history = [{
        "role": "assistant",
        "content": "new",
        "mid": 2,
        "provider_sent": False,
    }]
    state.histories[key] = history
    monkeypatch.setattr(llm_client, "save_history", lambda _key: False)

    with pytest.raises(RuntimeError, match="границу отправки"):
        asyncio.run(llm_client._mark_history_sent_to_provider(history, key=key))

    assert history[0]["provider_sent"] is False
