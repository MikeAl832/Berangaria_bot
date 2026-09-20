"""Process-local fingerprints for overnight getUpdates / Conflict diagnosis."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How often the heartbeat reminds us the poller is still alive.
HEARTBEAT_INTERVAL_SECONDS = 600.0
# Real stall: getUpdates has not *returned* (even empty) for this long.
STALE_POLL_SECONDS = 30 * 60
# After this long without a successful getUpdates return, kill PID 1 so Docker
# restart:always recovers. Must be > STALE_POLL_SECONDS. Quiet nights with a
# working long-poll still call getUpdates regularly — this is not "no messages".
STALE_POLL_EXIT_SECONDS = 45 * 60
# Daemon thread: if the asyncio loop stops beating this long, kill PID 1.
# Must be > HEARTBEAT_INTERVAL_SECONDS.
LOOP_WATCHDOG_SECONDS = 20 * 60
LOOP_WATCHDOG_POLL_SECONDS = 30.0
# Sidecar stamp for Docker HEALTHCHECK (separate process can kill frozen PID 1).
LOOP_HEARTBEAT_PATH = Path(os.environ.get("BOT_LOOP_HEARTBEAT_PATH", "/data/loop_heartbeat"))
POLL_HEARTBEAT_PATH = Path(os.environ.get("BOT_POLL_HEARTBEAT_PATH", "/data/poll_heartbeat"))

_last_loop_beat: float = 0.0
_watchdog_started: bool = False
_stall_dump_emitted: bool = False


@dataclass(frozen=True)
class PollingSnapshot:
    hostname: str
    pid: int
    bot_id: int | None
    started_at: float | None
    last_update_at: float | None
    last_poll_at: float | None
    last_error_at: float | None
    last_error_type: str | None
    updates_seen: int
    polls_seen: int

    @property
    def uptime_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        return max(0.0, time.time() - self.started_at)

    @property
    def seconds_since_update(self) -> float | None:
        if self.last_update_at is None:
            return None
        return max(0.0, time.time() - self.last_update_at)

    @property
    def seconds_since_poll(self) -> float | None:
        if self.last_poll_at is None:
            return None
        return max(0.0, time.time() - self.last_poll_at)


_started_at: float | None = None
_bot_id: int | None = None
_last_update_at: float | None = None
_last_poll_at: float | None = None
_last_error_at: float | None = None
_last_error_type: str | None = None
_updates_seen: int = 0
_polls_seen: int = 0


def reset_for_tests() -> None:
    """Clear process-local counters between unit tests."""
    global _started_at, _bot_id, _last_update_at, _last_poll_at
    global _last_error_at, _last_error_type, _updates_seen, _polls_seen
    global _last_loop_beat, _watchdog_started, _stall_dump_emitted
    _started_at = None
    _bot_id = None
    _last_update_at = None
    _last_poll_at = None
    _last_error_at = None
    _last_error_type = None
    _updates_seen = 0
    _polls_seen = 0
    _last_loop_beat = 0.0
    _watchdog_started = False
    _stall_dump_emitted = False


def _write_stamp(path: Path) -> None:
    """Best-effort mtime stamp for an external HEALTHCHECK (no logging)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.3f}\n", encoding="utf-8")
    except OSError:
        pass


def touch_loop_beat() -> None:
    """Record that the asyncio loop is still scheduling work."""
    global _last_loop_beat
    _last_loop_beat = time.monotonic()
    _write_stamp(LOOP_HEARTBEAT_PATH)


def loop_watchdog_should_exit(*, now: float | None = None) -> bool:
    """True when the event loop has missed more than one heartbeat."""
    if _last_loop_beat <= 0.0:
        return False
    clock = time.monotonic() if now is None else now
    return (clock - _last_loop_beat) > LOOP_WATCHDOG_SECONDS


def _loop_watchdog_thread() -> None:
    while True:
        time.sleep(LOOP_WATCHDOG_POLL_SECONDS)
        if loop_watchdog_should_exit():
            # Do not log: the loop (and logging) may be the thing that is stuck.
            os._exit(1)


def start_loop_watchdog() -> None:
    """Daemon thread independent of asyncio. Safe to call once from main()."""
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    touch_loop_beat()
    thread = threading.Thread(
        target=_loop_watchdog_thread,
        name="berangaria-loop-watchdog",
        daemon=True,
    )
    thread.start()


def mark_started(*, bot_id: int | None = None) -> PollingSnapshot:
    """Stamp process identity when Application is ready to poll."""
    global _started_at, _bot_id, _stall_dump_emitted
    _started_at = time.time()
    _stall_dump_emitted = False
    if bot_id is not None:
        _bot_id = int(bot_id)
    touch_loop_beat()
    snap = snapshot()
    logger.info(
        "📡 Polling identity: host=%s pid=%s bot_id=%s",
        snap.hostname,
        snap.pid,
        snap.bot_id if snap.bot_id is not None else "?",
    )
    return snap


def mark_update_received() -> None:
    """Record that Telegram delivered an update to this process."""
    global _last_update_at, _updates_seen
    _last_update_at = time.time()
    _updates_seen += 1


def mark_poll_ok(*, update_count: int = 0) -> None:
    """Record that getUpdates returned (including an empty tuple)."""
    global _last_poll_at, _polls_seen, _stall_dump_emitted
    _last_poll_at = time.time()
    _polls_seen += 1
    _stall_dump_emitted = False
    _write_stamp(POLL_HEARTBEAT_PATH)
    if update_count:
        # Empty long-polls are the common case; non-empty also advances updates
        # via TypeHandler, but keep poll stamp independent.
        pass


def mark_error(error: BaseException | None) -> None:
    """Record the latest polling/handler transport error type."""
    global _last_error_at, _last_error_type
    _last_error_at = time.time()
    _last_error_type = type(error).__name__ if error is not None else "unknown"


def snapshot() -> PollingSnapshot:
    return PollingSnapshot(
        hostname=socket.gethostname(),
        pid=os.getpid(),
        bot_id=_bot_id,
        started_at=_started_at,
        last_update_at=_last_update_at,
        last_poll_at=_last_poll_at,
        last_error_at=_last_error_at,
        last_error_type=_last_error_type,
        updates_seen=_updates_seen,
        polls_seen=_polls_seen,
    )


def format_context(error: BaseException | None = None) -> str:
    """Compact one-line context for logs (no secrets).

    ``host=`` / ``pid=`` / ``bot_id=`` identify *this* process — the one that
    observed the event — not the holder of a competing getUpdates long-poll.
    Volatile fields (uptime, since_update, updates) belong in log lines and
    alert ``detail``, never in the owner-alert fingerprint.
    """
    snap = snapshot()
    uptime = (
        f"{snap.uptime_seconds:.0f}s"
        if snap.uptime_seconds is not None
        else "n/a"
    )
    since_update = (
        f"{snap.seconds_since_update:.0f}s"
        if snap.seconds_since_update is not None
        else "never"
    )
    since_poll = (
        f"{snap.seconds_since_poll:.0f}s"
        if snap.seconds_since_poll is not None
        else "never"
    )
    err_bit = ""
    if error is not None:
        err_bit = f" err={type(error).__name__}"
    elif snap.last_error_type:
        age = (
            f"{max(0.0, time.time() - snap.last_error_at):.0f}s"
            if snap.last_error_at is not None
            else "?"
        )
        err_bit = f" last_err={snap.last_error_type}@{age}"
    return (
        f"host={snap.hostname} pid={snap.pid} bot_id="
        f"{snap.bot_id if snap.bot_id is not None else '?'} "
        f"uptime={uptime} since_update={since_update} "
        f"since_poll={since_poll} updates={snap.updates_seen} "
        f"polls={snap.polls_seen}{err_bit}"
    )


def install_get_updates_probe(bot: Any) -> None:
    """Wrap ``bot.get_updates`` so empty long-polls still count as poll OK."""
    if getattr(bot, "_berangaria_poll_probe", False):
        return
    original = bot.get_updates

    async def get_updates_probed(*args: Any, **kwargs: Any):
        try:
            result = await original(*args, **kwargs)
        except Exception as exc:
            mark_error(exc)
            raise
        try:
            count = len(result) if result is not None else 0
        except TypeError:
            count = 0
        mark_poll_ok(update_count=count)
        return result

    # ExtBot freezes public attrs after init; bypass TelegramObject.__setattr__.
    object.__setattr__(bot, "get_updates", get_updates_probed)
    object.__setattr__(bot, "_berangaria_poll_probe", True)


def dump_stall_diagnostics() -> str:
    """Build a hard debug dump of asyncio tasks and thread stacks."""
    lines: list[str] = ["📡 Polling stall dump BEGIN", format_context()]
    try:
        loop = asyncio.get_running_loop()
        tasks = asyncio.all_tasks(loop)
        lines.append(f"asyncio_tasks={len(tasks)}")
        for task in sorted(tasks, key=lambda t: t.get_name()):
            name = task.get_name()
            coro = task.get_coro()
            coro_name = getattr(coro, "__qualname__", type(coro).__name__)
            state = "done" if task.done() else "pending"
            lines.append(f"  task name={name} state={state} coro={coro_name}")
            stack = task.get_stack(limit=8)
            if stack:
                lines.append("    " + "".join(traceback.format_list(stack)).strip().replace("\n", "\n    "))
    except RuntimeError as exc:
        lines.append(f"asyncio_tasks unavailable: {exc}")

    lines.append("thread_stacks:")
    frames = sys._current_frames()
    for thread in threading.enumerate():
        frame = frames.get(thread.ident) if thread.ident is not None else None
        lines.append(f"  thread name={thread.name} ident={thread.ident} daemon={thread.daemon}")
        if frame is None:
            lines.append("    <no frame>")
            continue
        lines.append(
            "    "
            + "".join(traceback.format_stack(frame, limit=12)).strip().replace("\n", "\n    ")
        )
    lines.append("📡 Polling stall dump END")
    return "\n".join(lines)


def poll_stall_should_exit(*, now: float | None = None) -> bool:
    """True when getUpdates has not returned for STALE_POLL_EXIT_SECONDS."""
    snap = snapshot()
    if snap.last_poll_at is None:
        # Never successfully polled after start — give the first long-poll time,
        # but still exit if we never get a return past the exit budget from start.
        if snap.started_at is None:
            return False
        clock = time.time() if now is None else now
        return (clock - snap.started_at) > STALE_POLL_EXIT_SECONDS
    since = snap.seconds_since_poll
    if since is None:
        return False
    return since > STALE_POLL_EXIT_SECONDS


async def polling_heartbeat_loop(sleep=None, *, exit_fn=None) -> None:
    """Periodic INFO so a quiet night still leaves a trail in bot.log.

    Stall is based on ``since_poll`` (getUpdates returned), not ``since_update``
    (a message arrived). Quiet chats are normal; a stuck long-poll is not.
    After ``STALE_POLL_EXIT_SECONDS`` without a poll return, exit so Docker
    recreates the container. Do not DM the owner on stall.
    """
    sleeper = sleep if sleep is not None else asyncio.sleep
    killer = exit_fn if exit_fn is not None else os._exit
    global _stall_dump_emitted
    while True:
        await sleeper(HEARTBEAT_INTERVAL_SECONDS)
        try:
            touch_loop_beat()
            ctx = format_context()
            logger.info("📡 Polling heartbeat: %s", ctx)
            since_poll = snapshot().seconds_since_poll
            if since_poll is None:
                # No successful poll yet — still allow exit via poll_stall_should_exit.
                if poll_stall_should_exit():
                    logger.critical(
                        "📡 Polling never completed getUpdates — exiting for Docker restart | %s",
                        ctx,
                    )
                    killer(1)
                continue
            if since_poll < STALE_POLL_SECONDS:
                continue
            logger.warning(
                "📡 Polling stalled: нет успешного getUpdates уже %.0fс "
                "(порог %.0fс) | %s",
                since_poll,
                STALE_POLL_SECONDS,
                ctx,
            )
            if not _stall_dump_emitted:
                _stall_dump_emitted = True
                try:
                    logger.warning("%s", dump_stall_diagnostics())
                except Exception:
                    logger.exception("Polling stall dump failed")
            if poll_stall_should_exit():
                logger.critical(
                    "📡 Polling stalled beyond %.0fс — exiting for Docker restart | %s",
                    STALE_POLL_EXIT_SECONDS,
                    ctx,
                )
                killer(1)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Polling heartbeat failed")
