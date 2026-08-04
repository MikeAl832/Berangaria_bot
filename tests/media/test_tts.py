"""Unit tests for Fish TTS helpers (no live network)."""

import pytest

from berangaria.media import tts as tts_mod


def test_sanitize_strips_control_brackets_and_caps(monkeypatch):
    monkeypatch.setattr(tts_mod, "TTS_MAX_CHARS", 20)
    out = tts_mod.sanitize_speech_text("  [angry] Hello  [whisper] world!!!  ")
    assert "[" not in out
    assert out.startswith("Hello")
    assert len(out) <= 20


def test_normalize_emotion_whitelist():
    assert tts_mod.normalize_emotion("Sarcastic") == "sarcastic"
    assert tts_mod.normalize_emotion("none") is None
    assert tts_mod.normalize_emotion("hysterical") is None
    assert tts_mod.normalize_emotion(42) is None


def test_resolve_emotion_default_and_none(monkeypatch):
    monkeypatch.setattr(tts_mod, "TTS_DEFAULT_EMOTION", "calm")
    assert tts_mod.resolve_emotion_key(...) == "calm"
    assert tts_mod.resolve_emotion_key(None) == "calm"
    assert tts_mod.resolve_emotion_key("none") is None
    assert tts_mod.resolve_emotion_key("bored") == "bored"


def test_build_tts_payload_applies_tag(monkeypatch):
    monkeypatch.setattr(tts_mod, "TTS_DEFAULT_EMOTION", "calm")
    assert tts_mod.build_tts_payload_text("Привет", "sarcastic").startswith("[sarcastic]")
    assert tts_mod.build_tts_payload_text("Привет", "none") == "Привет"
    assert tts_mod.build_tts_payload_text("Привет", ...).startswith("[calm]")


def test_synthesize_raises_on_empty(monkeypatch):
    monkeypatch.setattr(tts_mod, "is_tts_ready", lambda: True)
    monkeypatch.setattr(tts_mod, "FISH_API_KEY", "k")
    monkeypatch.setattr(tts_mod, "FISH_VOICE_ID", "v")
    with pytest.raises(tts_mod.TTSError, match="Пустой"):
        tts_mod.synthesize_speech("   ")
