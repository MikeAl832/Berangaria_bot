"""Smoke tests for the external bot sampler helpers."""

import scripts.bot_sampler as s


def test_stamp_age_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "LOOP_STAMP", tmp_path / "nope")
    assert s._stamp_age(tmp_path / "nope") is None


def test_log_writes(tmp_path, monkeypatch):
    log = tmp_path / "sampler.log"
    monkeypatch.setattr(s, "LOG_PATH", log)
    s._log("hello")
    assert "hello" in log.read_text(encoding="utf-8")
