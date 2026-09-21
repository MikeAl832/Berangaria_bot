"""Docker HEALTHCHECK helper for frozen event loops."""

import os
import signal
import time

import scripts.loop_healthcheck as hc


def test_healthcheck_ok_when_stamp_fresh(tmp_path, monkeypatch):
    stamp = tmp_path / "loop_heartbeat"
    stamp.write_text("1\n", encoding="utf-8")
    monkeypatch.setattr(hc, "PATH", stamp)
    monkeypatch.setattr(hc, "STALE_SECONDS", 60.0)
    assert hc.main() == 0


def test_healthcheck_kills_berangaria_workers_not_pid1(tmp_path, monkeypatch):
    stamp = tmp_path / "loop_heartbeat"
    stamp.write_text("1\n", encoding="utf-8")
    old = time.time() - 1000
    os.utime(stamp, (old, old))
    killed = []
    monkeypatch.setattr(hc, "PATH", stamp)
    monkeypatch.setattr(hc, "STALE_SECONDS", 60.0)
    monkeypatch.setattr(hc, "_berangaria_pids", lambda: [42, 99])
    monkeypatch.setattr(hc.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    assert hc.main() == 1
    assert killed == [(42, signal.SIGKILL), (99, signal.SIGKILL)]


def test_healthcheck_fails_when_no_worker(tmp_path, monkeypatch):
    stamp = tmp_path / "loop_heartbeat"
    stamp.write_text("1\n", encoding="utf-8")
    old = time.time() - 1000
    os.utime(stamp, (old, old))
    monkeypatch.setattr(hc, "PATH", stamp)
    monkeypatch.setattr(hc, "STALE_SECONDS", 60.0)
    monkeypatch.setattr(hc, "_berangaria_pids", lambda: [])
    assert hc.main() == 1
