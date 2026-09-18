"""Process-local fingerprints for overnight getUpdates / Conflict diagnosis."""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# How often the heartbeat reminds us the poller is still alive.
HEARTBEAT_INTERVAL_SECONDS = 600.0


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
    _started_at = None
    _bot_id = None
    _last_update_at = None
    _last_error_at = None
    _last_error_type = None
    _updates_seen = 0


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
    """Compact one-line context safe for logs and owner alerts (no secrets)."""
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
    """Periodic INFO so a quiet night still leaves a trail in bot.log."""
    sleeper = sleep
    if sleeper is None:
        import asyncio

        sleeper = asyncio.sleep
    while True:
        await sleeper(HEARTBEAT_INTERVAL_SECONDS)
        logger.info("📡 Polling heartbeat: %s", format_context())
