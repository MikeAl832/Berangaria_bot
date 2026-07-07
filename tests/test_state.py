import pytest

import state


def test_media_cache_roundtrip():
    state.media_description_cache.clear()
    assert state.get_cached_media_description("unknown") is None

    state.cache_media_description("abc", "описание картинки")
    assert state.get_cached_media_description("abc") == "описание картинки"


def test_media_cache_ignores_empty():
    state.media_description_cache.clear()
    state.cache_media_description("", "desc")          # пустой ключ
    state.cache_media_description("id", "")            # пустое описание
    assert state.get_cached_media_description("") is None
    assert state.get_cached_media_description("id") is None


def test_media_cache_lru_eviction(monkeypatch):
    state.media_description_cache.clear()
    monkeypatch.setattr(state, "MEDIA_CACHE_MAX", 3)

    for i in range(3):
        state.cache_media_description(f"id{i}", f"d{i}")

    # Освежаем id0 — теперь самый старый id1
    assert state.get_cached_media_description("id0") == "d0"

    state.cache_media_description("id3", "d3")  # переполнение -> вытесняется id1

    assert state.get_cached_media_description("id1") is None
    assert state.get_cached_media_description("id0") == "d0"
    assert state.get_cached_media_description("id3") == "d3"
    assert len(state.media_description_cache) == 3


def test_history_persistence_roundtrip(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(state, "DB_PATH", str(db_file))

    state.histories.clear()
    state.init_db()

    state.histories["private_42"] = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "и тебе"},
    ]
    state.save_history("private_42")

    # Эмулируем рестарт: память чистая, грузим с диска
    state.histories.clear()
    loaded = state.load_all_histories()

    assert loaded == 1
    assert state.histories["private_42"][0]["content"] == "привет"
    assert state.histories["private_42"][1]["content"] == "и тебе"


def test_history_delete(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(state, "DB_PATH", str(db_file))

    state.histories.clear()
    state.init_db()
    state.histories["group_1"] = [{"role": "user", "content": "x"}]
    state.save_history("group_1")

    state.delete_history("group_1")
    state.histories.clear()
    assert state.load_all_histories() == 0


def test_runtime_settings_use_config_default_when_empty(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(state, "DB_PATH", str(db_file))
    monkeypatch.setattr(state, "random_reply_chance", 10)

    state.init_db()
    settings = state.load_runtime_settings(default_random_reply_chance=17)

    assert settings.random_reply_chance == 17
    assert state.random_reply_chance == 17


def test_random_reply_chance_persists_between_restarts(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(state, "DB_PATH", str(db_file))
    monkeypatch.setattr(state, "random_reply_chance", 10)

    state.init_db()
    assert state.set_random_reply_chance(42) == 42

    # Эмулируем рестарт: in-memory значение потеряно, БД осталась.
    state.random_reply_chance = 10
    settings = state.load_runtime_settings(default_random_reply_chance=10)

    assert settings.random_reply_chance == 42
    assert state.random_reply_chance == 42


@pytest.mark.parametrize("value", [-1, 101, True, "10"])
def test_random_reply_chance_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        state.validate_random_reply_chance(value)
