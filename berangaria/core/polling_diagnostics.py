"""Process-local polling identity, heartbeat stamp, and Conflict context."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from berangaria.core.paths import project_path

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 600.0


def _default_heartbeat_path() -> Path:
    explicit = os.environ.get("BOT_LOOP_HEARTBEAT_PATH")
    if explicit:
        return Path(explicit)
    db_path = project_path(os.environ.get("BOT_DB_PATH") or "bot_state.db")
    return Path(db_path).parent / "loop_heartbeat"


LOOP_HEARTBEAT_PATH = _default_heartbeat_path()


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
    global _started_at, _bot_id, _last_update_at
    global _last_error_at, _last_error_type, _updates_seen
    _started_at = None
    _bot_id = None
    _last_update_at = None
    _last_error_at = None
    _last_error_type = None
    _updates_seen = 0


def touch_loop_beat() -> None:
    """Best-effort mtime stamp for optional container healthcheck."""
    try:
        LOOP_HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOOP_HEARTBEAT_PATH.write_text(f"{time.time():.3f}\n", encoding="utf-8")
    except OSError:
        pass


def mark_started(*, bot_id: int | None = None) -> PollingSnapshot:
    """Stamp process identity when Application is ready to poll."""
    global _started_at, _bot_id
    _started_at = time.time()
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
    """Compact one-line context for logs and Conflict alert details."""
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
    """Periodic INFO heartbeat and loop stamp."""
    sleeper = sleep if sleep is not None else asyncio.sleep
    while True:
        await sleeper(HEARTBEAT_INTERVAL_SECONDS)
        try:
            touch_loop_beat()
            logger.info("📡 Polling heartbeat: %s", format_context())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Polling heartbeat failed")
