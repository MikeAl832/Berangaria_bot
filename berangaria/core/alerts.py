"""Throttled critical notifications for the bot owner."""

from __future__ import annotations

import hashlib
import logging
import time

from berangaria.analytics import store as analytics_store
from berangaria.config import ADMIN_ALERT_CHAT_ID, OWNER_USER_ID

logger = logging.getLogger(__name__)

ALERT_COOLDOWN_SECONDS = 60.0
_alert_state: dict[str, dict[str, float | int]] = {}


def get_alert_chat_id() -> int | None:
    """Use an explicit destination when set, otherwise the owner's private ID."""
    if ADMIN_ALERT_CHAT_ID is not None:
        return ADMIN_ALERT_CHAT_ID
    return OWNER_USER_ID


def _fingerprint(category: str, message: str, error: BaseException | None) -> str:
    error_type = type(error).__name__ if error is not None else ""
    material = f"{category}\n{error_type}\n{message}".encode("utf-8", errors="replace")
    return hashlib.sha256(material).hexdigest()[:24]


async def notify_owner(
    bot,
    *,
    category: str,
    message: str,
    error: BaseException | None = None,
) -> bool:
    """Persist and send a deduplicated critical alert without raising outward."""
    clean_message = " ".join(str(message).split())[:1000] or "Неизвестная ошибка"
    if error is not None:
        error_text = " ".join(str(error).split())
        if error_text and error_text not in clean_message:
            clean_message = f"{clean_message}: {error_text}"[:1000]
    fingerprint = _fingerprint(category, clean_message, error)
    analytics_store.record_alert(
        category=category,
        fingerprint=fingerprint,
        message=clean_message,
    )

    destination = get_alert_chat_id()
    if destination is None or bot is None:
        return False

    now = time.time()
    alert = _alert_state.setdefault(
        fingerprint,
        {"last_attempt": 0.0, "suppressed": 0},
    )
    last_attempt = float(alert["last_attempt"])
    if now - last_attempt < ALERT_COOLDOWN_SECONDS:
        alert["suppressed"] = int(alert["suppressed"]) + 1
        return False

    suppressed = int(alert["suppressed"])
    suffix = f"\nПовторов подавлено: {suppressed}" if suppressed else ""
    text = f"⚠️ {category}\n{clean_message}{suffix}"
    alert["last_attempt"] = now
    alert["suppressed"] = 0
    try:
        await bot.send_message(chat_id=destination, text=text)
        return True
    except Exception as exc:
        logger.error("Не удалось отправить критический алерт в chat_id=%s: %s", destination, exc)
        return False


def reset_alert_throttle() -> None:
    """Test helper; production never needs to reset deduplication state."""
    _alert_state.clear()
