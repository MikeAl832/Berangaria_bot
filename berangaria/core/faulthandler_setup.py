"""Dump all Python threads on SIGUSR1 for the external bot sampler."""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
from pathlib import Path

logger = logging.getLogger(__name__)

_FAULT_PATH = Path(os.environ.get("BOT_FAULTHANDLER_PATH", "/data/faulthandler.log"))
_fault_file = None


def install_faulthandler_sigusr1() -> None:
    """Register SIGUSR1 → all-threads dump into /data/faulthandler.log."""
    global _fault_file
    try:
        _FAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _fault_file = _FAULT_PATH.open("a", encoding="utf-8")
        faulthandler.register(
            signal.SIGUSR1,
            file=_fault_file,
            all_threads=True,
            chain=False,
        )
        logger.info("🩺 Faulthandler SIGUSR1 → %s", _FAULT_PATH)
    except OSError as exc:
        logger.warning("Faulthandler not installed: %s", exc)
