"""Explicit terminal outcomes shared by chat turn callers."""

from enum import Enum


class TurnOutcome(Enum):
    """Whether a turn delivered a reply/action, stayed silent, or failed."""

    DELIVERED = "delivered"
    SILENT = "silent"
    FAILED = "failed"
