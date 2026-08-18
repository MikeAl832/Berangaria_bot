from berangaria.analytics import store


def test_durable_overview_and_chat_scoped_leaderboards(isolated_db):
    now = 1_800_000_000.0
    store.record_event(
        "user_message",
        chat_id=-100,
        chat_type="supergroup",
        actor_id=1,
        actor_name="Аня",
        occurred_at=now,
    )
    store.record_event(
        "user_message",
        chat_id=-100,
        chat_type="supergroup",
        actor_id=1,
        actor_name="Аня",
        occurred_at=now + 1,
    )
    store.record_event(
        "user_message",
        chat_id=-100,
        chat_type="supergroup",
        actor_id=2,
        actor_name="Борис",
        occurred_at=now + 2,
    )
    store.record_event(
        "user_message",
        chat_id=-100,
        chat_type="supergroup",
        actor_id=99,
        actor_name="Другой бот",
        actor_kind="bot",
        occurred_at=now + 3,
    )
    store.record_event(
        "assistant_reply",
        chat_id=-100,
        chat_type="supergroup",
        target_user_id=2,
        target_user_name="Борис",
        occurred_at=now + 4,
    )
    store.record_event(
        "bot_reaction",
        chat_id=-100,
        chat_type="supergroup",
        target_user_id=1,
        target_user_name="Аня",
        occurred_at=now + 5,
    )
    store.record_event(
        "user_message",
        chat_id=-200,
        chat_type="supergroup",
        actor_id=3,
        actor_name="Чужой чат",
        occurred_at=now + 6,
    )
    store.record_llm_usage(
        chat_id=-100,
        chat_type="supergroup",
        user_id=1,
        user_name="Аня",
        provider="test",
        model="test-model",
        prompt_tokens=100,
        cached_tokens=80,
        cache_write_tokens=0,
        completion_tokens=20,
        total_tokens=120,
        cost_usd=0.000123,
        occurred_at=now + 7,
    )

    overview = store.get_overview("all", chat_id=-100)
    assert overview == {
        "messages": 3,
        "active_users": 2,
        "assistant_replies": 1,
        "bot_reactions": 1,
        "incoming_reactions": 0,
        "requests": 1,
        "prompt_tokens": 100,
        "cached_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cost_microusd": 123,
    }

    leaders = store.get_leaderboards("all", chat_id=-100)
    assert [(row["name"], row["value"]) for row in leaders["messages"]] == [
        ("Аня", 2),
        ("Борис", 1),
    ]
    assert leaders["replies"][0]["user_id"] == 2
    assert leaders["bot_reactions"][0]["user_id"] == 1
    assert leaders["cost"][0]["value"] == 123


def test_alerts_are_grouped_by_fingerprint(isolated_db):
    assert store.record_alert(category="LLM", fingerprint="same", message="timeout")
    assert store.record_alert(category="LLM", fingerprint="same", message="timeout")
    assert store.record_alert(category="DB", fingerprint="other", message="locked")

    rows = store.get_recent_alerts("all")

    grouped = {row["category"]: row for row in rows}
    assert grouped["LLM"]["count"] == 2
    assert grouped["DB"]["count"] == 1


def test_recording_is_disabled_until_current_database_schema_is_ready(tmp_path, monkeypatch):
    from berangaria.core import state

    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "not-initialized.db"))

    assert not store.record_event(
        "user_message",
        chat_id=1,
        chat_type="private",
        actor_id=1,
        actor_name="Аня",
    )
