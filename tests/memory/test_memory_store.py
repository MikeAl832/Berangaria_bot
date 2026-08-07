from berangaria.memory import store as memory_store
from berangaria.prompts import MEM0_CUSTOM_INSTRUCTIONS


def test_memory_initialization_retries(monkeypatch):
    sentinel = object()
    calls = {"count": 0}

    def from_config(config):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("qdrant starting")
        return sentinel

    import mem0

    monkeypatch.setattr(mem0.Memory, "from_config", from_config)
    monkeypatch.setattr(memory_store.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(memory_store, "memory", None)

    result = memory_store.initialize_memory(attempts=3, delay_seconds=0)

    assert result is sentinel
    assert memory_store.memory is sentinel
    assert calls["count"] == 3


def test_memory_quality_settings_are_strict_and_explicit():
    # Prompt-surface policy, not shipped numeric knobs (memory_min_score lives in
    # config.yaml and must stay free to tune without breaking unit tests).
    assert "только один уже одобренный факт" in MEM0_CUSTOM_INSTRUCTIONS
    assert "не перефразируй" in MEM0_CUSTOM_INSTRUCTIONS
    assert "точный переданный текст" in MEM0_CUSTOM_INSTRUCTIONS
