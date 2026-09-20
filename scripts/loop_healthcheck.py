#!/usr/bin/env python3
"""Docker HEALTHCHECK: kill PID 1 if the asyncio loop heartbeat stamp is stale.

Runs in a separate process from the bot, so a GIL-frozen main thread still
cannot block this check. Stale stamp → SIGKILL PID 1 → restart: always.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

STALE_SECONDS = float(os.environ.get("BOT_LOOP_HEALTHCHECK_STALE_SECONDS", str(25 * 60)))
PATH = Path(os.environ.get("BOT_LOOP_HEARTBEAT_PATH", "/data/loop_heartbeat"))


def main() -> int:
    if not PATH.is_file():
        # Start period should cover first heartbeat; fail soft until stamp exists
        # only if process is young — healthcheck start_period handles that.
        print(f"missing {PATH}", file=sys.stderr)
        return 1
    age = time.time() - PATH.stat().st_mtime
    if age <= STALE_SECONDS:
        return 0
    print(f"stale loop heartbeat age={age:.0f}s > {STALE_SECONDS:.0f}s — killing PID 1", file=sys.stderr)
    try:
        os.kill(1, 9)
    except OSError as exc:
        print(f"kill failed: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
