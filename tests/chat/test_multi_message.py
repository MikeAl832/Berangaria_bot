"""Unit tests for multi-message delay and delivery helpers."""
import asyncio

import pytest

from berangaria.chat import llm_client


def test_multi_message_delay_respects_total_cap(monkeypatch):
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_DELAY_MIN", 0.5)
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_DELAY_MAX", 2.0)
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_DELAY_TOTAL_CAP", 1.0)
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_CHARS_PER_SEC", 10.0)
    monkeypatch.setattr(llm_client.random, "uniform", lambda a, b: 1.0)

    # Already slept up to the cap → no further pause.
    assert llm_client._multi_message_delay_seconds("xxxxxxxx", slept_total=1.0) == 0.0
    # Remaining budget clamps a large base delay.
    delay = llm_client._multi_message_delay_seconds("x" * 100, slept_total=0.6)
    assert 0 < delay <= 0.4


def test_multi_message_delay_scales_with_length(monkeypatch):
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_DELAY_MIN", 0.1)
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_DELAY_MAX", 5.0)
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_DELAY_TOTAL_CAP", 10.0)
    monkeypatch.setattr(llm_client, "MULTI_MESSAGE_CHARS_PER_SEC", 20.0)
    monkeypatch.setattr(llm_client.random, "uniform", lambda a, b: 1.0)

    short = llm_client._multi_message_delay_seconds("hi")
    long = llm_client._multi_message_delay_seconds("x" * 80)
    assert short == pytest.approx(0.1)  # clamped to min
    assert long == pytest.approx(4.0)
