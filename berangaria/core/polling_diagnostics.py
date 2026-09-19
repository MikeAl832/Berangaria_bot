"""Process-local fingerprints for overnight getUpdates / Conflict diagnosis."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# How often the heartbeat reminds us the poller is still alive.
HEARTBEAT_INTERVAL_SECONDS = 600.0
# Owner/log signal when getUpdates has been silent this long while the process is up.
STALE_UPDATE_SECONDS = 30 * 60
# Daemon thread: if the asyncio loop stops beating this long, kill PID 1 so
# Docker restart:always recovers. Must be > HEARTBEAT_INTERVAL_SECONDS.
LOOP_WATCHDOG_SECONDS = 20 * 60
LOOP_WATCHDOG_POLL_SECONDS = 30.0

_last_loop_beat: float = 0.0
_watchdog_started: bool = False


@dataclass(frozen=True)
class PollingSnapshot:
    hostname: str
    pid: int
    bot_id: int | None
    started_at: float | None
    last_update_at: float | None
    last_error_at: float | None
    last_error_type: str | None
    updates_seen: int

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


_started_at: float | None = None
_bot_id: int | None = None
_last_update_at: float | None = None
_last_error_at: float | None = None
_last_error_type: str | None = None
_updates_seen: int = 0


def reset_for_tests() -> None:
    """Clear process-local counters between unit tests."""
    global _started_at, _bot_id, _last_update_at, _last_error_at, _last_error_type, _updates_seen
    global _last_loop_beat, _watchdog_started
    _started_at = None
    _bot_id = None
    _last_update_at = None
    _last_error_at = None
    _last_error_type = None
    _updates_seen = 0
    _last_loop_beat = 0.0
    _watchdog_started = False


def touch_loop_beat() -> None:
    """Record that the asyncio loop is still scheduling work."""
    global _last_loop_beat
    _last_loop_beat = time.monotonic()


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
    global _started_at, _bot_id
    _started_at = time.time()
    if bot_id is not None:
        _bot_id = int(bot_id)
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
        last_error_at=_last_error_at,
        last_error_type=_last_error_type,
        updates_seen=_updates_seen,
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
        f"updates={snap.updates_seen}{err_bit}"
    )


async def polling_heartbeat_loop(sleep=None) -> None:
    """Periodic INFO so a quiet night still leaves a trail in bot.log.

    After ``STALE_UPDATE_SECONDS`` without an update this process has already
    seen, emit a WARNING in ``bot.log`` only. Quiet chats are normal at night —
    do not DM the owner. Silence before the first update is not this signal —
    that is still ``since_update=never``.
    """
    sleeper = sleep if sleep is not None else asyncio.sleep
    while True:
        await sleeper(HEARTBEAT_INTERVAL_SECONDS)
        try:
            touch_loop_beat()
            ctx = format_context()
            logger.info("📡 Polling heartbeat: %s", ctx)
            since = snapshot().seconds_since_update
            if since is None or since < STALE_UPDATE_SECONDS:
                continue
            logger.warning(
                "📡 Polling stalled: нет getUpdates уже %.0fс (порог %.0fс) | %s",
                since,
                STALE_UPDATE_SECONDS,
                ctx,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Polling heartbeat failed")
