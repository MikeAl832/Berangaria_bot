#!/usr/bin/env python3
"""Docker HEALTHCHECK: kill the bot process if the loop heartbeat stamp is stale.

Runs in a separate process from the bot. Must not rely on killing PID 1:
container init ignores SIGKILL/SIGTERM to PID 1, so a frozen `python -m
berangaria` as PID 1 never dies. With Compose `init: true`, tini is PID 1 and
we SIGKILL the berangaria child; tini then exits and `restart: always` recovers.
"""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

STALE_SECONDS = float(os.environ.get("BOT_LOOP_HEALTHCHECK_STALE_SECONDS", str(25 * 60)))
PATH = Path(os.environ.get("BOT_LOOP_HEARTBEAT_PATH", "/data/loop_heartbeat"))


def _berangaria_pids() -> list[int]:
    """PIDs whose cmdline looks like the bot (never this healthcheck)."""
    me = os.getpid()
    found: list[int] = []
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit():
            continue
        pid = int(ent.name)
        if pid == me or pid == 1:
            # Skip PID 1: kernel/container init will not honor SIGKILL there.
            continue
        try:
            raw = (ent / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        if "loop_healthcheck" in cmd or "bot_sampler" in cmd:
            continue
        if "docker-init" in cmd:
            continue
        if "python" in cmd and "berangaria" in cmd:
            found.append(pid)
    return found


def main() -> int:
    if not PATH.is_file():
        print(f"missing {PATH}", file=sys.stderr)
        return 1
    age = time.time() - PATH.stat().st_mtime
    if age <= STALE_SECONDS:
        return 0
    print(
        f"stale loop heartbeat age={age:.0f}s > {STALE_SECONDS:.0f}s — "
        "killing berangaria worker(s)",
        file=sys.stderr,
    )
    pids = _berangaria_pids()
    if not pids:
        print("no berangaria worker PID found (is init: true set?)", file=sys.stderr)
        return 1
    killed = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError as exc:
            print(f"kill {pid} failed: {exc}", file=sys.stderr)
    if not killed:
        return 1
    print(f"killed pids={killed}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
