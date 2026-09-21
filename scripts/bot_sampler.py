#!/usr/bin/env python3
"""External sampler for a frozen Berangaria bot process.

Runs in a sibling container that shares the bot PID namespace (`pid: service:bot`)
and the `/data` volume. Writes `/data/sampler.log` even when the bot event loop
and logging are wedged — that is the point.

Every SAMPLE_INTERVAL_SECONDS:
  - ages of loop_heartbeat / poll_heartbeat stamps
  - worker PID, state, wchan, RSS
  - on warn/critical (stale stamps) or every FULL_DUMP_EVERY samples:
      /proc stacks, open syscall, and SIGUSR1 so the bot's faulthandler
      dumps Python threads to /data/faulthandler.log
"""
from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(os.environ.get("BOT_SAMPLER_DATA", "/data"))
LOG_PATH = Path(os.environ.get("BOT_SAMPLER_LOG", str(DATA / "sampler.log")))
LOOP_STAMP = Path(os.environ.get("BOT_LOOP_HEARTBEAT_PATH", str(DATA / "loop_heartbeat")))
POLL_STAMP = Path(os.environ.get("BOT_POLL_HEARTBEAT_PATH", str(DATA / "poll_heartbeat")))
SAMPLE_INTERVAL_SECONDS = float(os.environ.get("BOT_SAMPLER_INTERVAL_SECONDS", "30"))
WARN_AFTER_SECONDS = float(os.environ.get("BOT_SAMPLER_WARN_SECONDS", str(15 * 60)))
FULL_DUMP_EVERY = int(os.environ.get("BOT_SAMPLER_FULL_DUMP_EVERY", "20"))  # ~10 min at 30s
LOG_MAX_BYTES = int(os.environ.get("BOT_SAMPLER_LOG_MAX_BYTES", str(5 * 1024 * 1024)))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.is_file() and path.stat().st_size >= LOG_MAX_BYTES:
            bak = path.with_suffix(path.suffix + ".1")
            if bak.exists():
                bak.unlink()
            path.rename(bak)
    except OSError:
        pass


def _log(line: str) -> None:
    _rotate_if_needed(LOG_PATH)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {line}\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        print(f"sampler log write failed: {exc}", file=sys.stderr)


def _stamp_age(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _berangaria_workers() -> list[int]:
    me = os.getpid()
    found: list[int] = []
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit():
            continue
        pid = int(ent.name)
        if pid == me:
            continue
        try:
            cmd = (ent / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", "replace"
            )
        except OSError:
            continue
        if "bot_sampler" in cmd or "loop_healthcheck" in cmd:
            continue
        if "berangaria" in cmd:
            found.append(pid)
    return found


def _read_text(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[:limit]
    except OSError as exc:
        return f"<err {exc}>"


def _worker_summary(pid: int) -> str:
    base = Path(f"/proc/{pid}")
    status = {}
    for line in _read_text(base / "status", 8000).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            status[k.strip()] = v.strip()
    wchan = _read_text(base / "wchan", 200).strip() or "?"
    cmdline = _read_text(base / "cmdline", 500).replace("\x00", " ").strip()
    return (
        f"pid={pid} state={status.get('State', '?')} "
        f"threads={status.get('Threads', '?')} "
        f"vmrss={status.get('VmRSS', '?')} wchan={wchan} cmd={cmdline[:120]}"
    )


def _dump_proc(pid: int) -> None:
    base = Path(f"/proc/{pid}")
    _log(f"--- proc dump pid={pid} BEGIN ---")
    _log(f"status:\n{_read_text(base / 'status', 3000)}")
    _log(f"wchan={_read_text(base / 'wchan', 200).strip()}")
    _log(f"syscall={_read_text(base / 'syscall', 500).strip()}")
    stack = _read_text(base / "stack", 8000)
    _log(f"stack:\n{stack}")
    task_dir = base / "task"
    try:
        tasks = sorted(task_dir.iterdir(), key=lambda p: int(p.name))[:32]
    except OSError as exc:
        _log(f"task list err: {exc}")
        tasks = []
    for task in tasks:
        tstack = _read_text(task / "stack", 2000)
        twchan = _read_text(task / "wchan", 200).strip()
        if tstack.strip() or twchan:
            _log(f"task {task.name} wchan={twchan}\n{tstack}")
    _log(f"--- proc dump pid={pid} END ---")


def _request_python_traceback(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGUSR1)
        _log(f"sent SIGUSR1 to pid={pid} (faulthandler -> /data/faulthandler.log)")
    except OSError as exc:
        _log(f"SIGUSR1 pid={pid} failed: {exc}")


def sample(*, force_full: bool = False) -> None:
    loop_age = _stamp_age(LOOP_STAMP)
    poll_age = _stamp_age(POLL_STAMP)
    workers = _berangaria_workers()
    loop_s = "missing" if loop_age is None else f"{loop_age:.0f}s"
    poll_s = "missing" if poll_age is None else f"{poll_age:.0f}s"
    level = "ok"
    if loop_age is None or poll_age is None:
        level = "warn"
    elif loop_age >= WARN_AFTER_SECONDS or poll_age >= WARN_AFTER_SECONDS:
        level = "critical"
    _log(
        f"[{level}] loop_heartbeat_age={loop_s} poll_heartbeat_age={poll_s} "
        f"workers={workers}"
    )
    for pid in workers:
        _log(_worker_summary(pid))
    if force_full or level in ("warn", "critical"):
        if not workers:
            _log("no berangaria worker in shared PID namespace")
            return
        for pid in workers:
            _dump_proc(pid)
            if level == "critical" or force_full:
                _request_python_traceback(pid)


def main() -> None:
    _log(
        f"sampler start interval={SAMPLE_INTERVAL_SECONDS}s "
        f"warn_after={WARN_AFTER_SECONDS}s log={LOG_PATH}"
    )
    n = 0
    while True:
        try:
            force = n > 0 and n % FULL_DUMP_EVERY == 0
            sample(force_full=force)
        except Exception as exc:
            _log(f"sample error: {exc!r}")
        n += 1
        time.sleep(SAMPLE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
