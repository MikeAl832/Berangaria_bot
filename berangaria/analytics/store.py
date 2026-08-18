"""Append-only analytics stored alongside bot state in SQLite.

Analytics must never become part of the chat transaction: every write is
best-effort, contains no message text, and failures are logged without changing
the user-visible bot behaviour.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from berangaria.config import TIMEZONE_NAME
from berangaria.core import state

logger = logging.getLogger(__name__)

_schema_db_path: str | None = None
_PERIOD_SECONDS = {
    "24h": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
    "all": None,
}


def init_schema() -> None:
    """Create analytics tables for the currently configured state database."""
    global _schema_db_path
    state._db_execute(
        "CREATE TABLE IF NOT EXISTS analytics_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "occurred_at REAL NOT NULL, "
        "event_type TEXT NOT NULL, "
        "chat_id INTEGER NOT NULL, "
        "chat_type TEXT NOT NULL, "
        "actor_id INTEGER, "
        "actor_name TEXT, "
        "actor_kind TEXT NOT NULL DEFAULT 'user', "
        "target_user_id INTEGER, "
        "target_user_name TEXT, "
        "message_id INTEGER, "
        "value INTEGER NOT NULL DEFAULT 1, "
        "local_hour INTEGER NOT NULL, "
        "local_weekday INTEGER NOT NULL, "
        "details TEXT NOT NULL DEFAULT '{}')"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_events_time "
        "ON analytics_events(occurred_at)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_events_type_time "
        "ON analytics_events(event_type, occurred_at)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_events_chat_time "
        "ON analytics_events(chat_id, occurred_at)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_events_actor_time "
        "ON analytics_events(actor_id, occurred_at)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_events_target_time "
        "ON analytics_events(target_user_id, occurred_at)"
    )
    state._db_execute(
        "CREATE TABLE IF NOT EXISTS analytics_llm_usage ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "occurred_at REAL NOT NULL, "
        "chat_id INTEGER NOT NULL, "
        "chat_type TEXT NOT NULL, "
        "user_id INTEGER NOT NULL, "
        "user_name TEXT NOT NULL, "
        "provider TEXT NOT NULL, "
        "model TEXT NOT NULL, "
        "prompt_tokens INTEGER NOT NULL, "
        "cached_tokens INTEGER NOT NULL, "
        "cache_write_tokens INTEGER NOT NULL, "
        "completion_tokens INTEGER NOT NULL, "
        "total_tokens INTEGER NOT NULL, "
        "cost_microusd INTEGER NOT NULL)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_usage_time "
        "ON analytics_llm_usage(occurred_at)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_usage_chat_time "
        "ON analytics_llm_usage(chat_id, occurred_at)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_usage_user_time "
        "ON analytics_llm_usage(user_id, occurred_at)"
    )
    state._db_execute(
        "CREATE TABLE IF NOT EXISTS analytics_alerts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "occurred_at REAL NOT NULL, "
        "category TEXT NOT NULL, "
        "fingerprint TEXT NOT NULL, "
        "message TEXT NOT NULL)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_alerts_time "
        "ON analytics_alerts(occurred_at)"
    )
    state._db_execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_alerts_fingerprint_time "
        "ON analytics_alerts(fingerprint, occurred_at)"
    )
    _schema_db_path = state.DB_PATH


def _ready() -> bool:
    return _schema_db_path == state.DB_PATH


def _local_parts(timestamp: float) -> tuple[int, int]:
    local = datetime.fromtimestamp(timestamp, ZoneInfo(TIMEZONE_NAME))
    return local.hour, local.weekday()


def record_event(
    event_type: str,
    *,
    chat_id: int,
    chat_type: str,
    actor_id: int | None = None,
    actor_name: str | None = None,
    actor_kind: str = "user",
    target_user_id: int | None = None,
    target_user_name: str | None = None,
    message_id: int | None = None,
    value: int = 1,
    occurred_at: float | None = None,
    details: dict[str, Any] | None = None,
) -> bool:
    """Append a content-free event. Returns False if analytics is unavailable."""
    if not _ready():
        return False
    try:
        timestamp = float(occurred_at if occurred_at is not None else time.time())
        hour, weekday = _local_parts(timestamp)
        state._db_execute(
            "INSERT INTO analytics_events "
            "(occurred_at, event_type, chat_id, chat_type, actor_id, actor_name, "
            "actor_kind, target_user_id, target_user_name, message_id, value, "
            "local_hour, local_weekday, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                event_type,
                int(chat_id),
                str(chat_type),
                actor_id,
                actor_name,
                actor_kind,
                target_user_id,
                target_user_name,
                message_id,
                max(1, int(value)),
                hour,
                weekday,
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        return True
    except Exception as exc:
        logger.warning("Аналитика: не удалось записать событие %s: %s", event_type, exc)
        return False


def record_llm_usage(
    *,
    chat_id: int,
    chat_type: str,
    user_id: int,
    user_name: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost_usd: float,
    occurred_at: float | None = None,
) -> bool:
    if not _ready():
        return False
    try:
        state._db_execute(
            "INSERT INTO analytics_llm_usage "
            "(occurred_at, chat_id, chat_type, user_id, user_name, provider, model, "
            "prompt_tokens, cached_tokens, cache_write_tokens, completion_tokens, "
            "total_tokens, cost_microusd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                float(occurred_at if occurred_at is not None else time.time()),
                int(chat_id),
                str(chat_type),
                int(user_id),
                user_name,
                provider,
                model,
                int(prompt_tokens or 0),
                int(cached_tokens or 0),
                int(cache_write_tokens or 0),
                int(completion_tokens or 0),
                int(total_tokens or 0),
                max(0, round(float(cost_usd) * 1_000_000)),
            ),
        )
        return True
    except Exception as exc:
        logger.warning("Аналитика: не удалось записать LLM usage: %s", exc)
        return False


def record_alert(*, category: str, fingerprint: str, message: str) -> bool:
    if not _ready():
        return False
    try:
        state._db_execute(
            "INSERT INTO analytics_alerts(occurred_at, category, fingerprint, message) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), category, fingerprint, message[:1000]),
        )
        return True
    except Exception as exc:
        logger.warning("Аналитика: не удалось записать критический алерт: %s", exc)
        return False


def _cutoff(period: str, *, now: float | None = None) -> float | None:
    seconds = _PERIOD_SECONDS.get(period, _PERIOD_SECONDS["7d"])
    if seconds is None:
        return None
    return float(now if now is not None else time.time()) - seconds


def _scope_params(period: str, chat_id: int | None) -> tuple[Any, ...]:
    """Parameters for static optional-cutoff and optional-chat predicates."""
    cutoff = _cutoff(period)
    normalized_chat_id = int(chat_id) if chat_id is not None else None
    return cutoff, cutoff, normalized_chat_id, normalized_chat_id


def get_overview(period: str = "7d", *, chat_id: int | None = None) -> dict[str, int]:
    if not _ready():
        return {}
    scope_params = _scope_params(period, chat_id)
    event_rows = state._db_execute(
        "SELECT "
        "COALESCE(SUM(CASE WHEN event_type='user_message' "
        "AND actor_kind IN ('user','owner') THEN value ELSE 0 END), 0), "
        "COUNT(DISTINCT CASE WHEN event_type='user_message' AND actor_kind IN ('user','owner') "
        "THEN actor_id END), "
        "COALESCE(SUM(CASE WHEN event_type='assistant_reply' THEN value ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN event_type='bot_reaction' THEN value ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN event_type='incoming_reaction_added' THEN value ELSE 0 END), 0) "
        "FROM analytics_events "
        "WHERE (? IS NULL OR occurred_at >= ?) AND (? IS NULL OR chat_id = ?)",
        scope_params,
        fetch=True,
    )
    usage_rows = state._db_execute(
        "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
        "COALESCE(SUM(cached_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
        "COALESCE(SUM(total_tokens), 0), COALESCE(SUM(cost_microusd), 0) "
        "FROM analytics_llm_usage "
        "WHERE (? IS NULL OR occurred_at >= ?) AND (? IS NULL OR chat_id = ?)",
        scope_params,
        fetch=True,
    )
    events = (event_rows or [(0, 0, 0, 0, 0)])[0]
    usage = (usage_rows or [(0, 0, 0, 0, 0, 0)])[0]
    return {
        "messages": int(events[0]),
        "active_users": int(events[1]),
        "assistant_replies": int(events[2]),
        "bot_reactions": int(events[3]),
        "incoming_reactions": int(events[4]),
        "requests": int(usage[0]),
        "prompt_tokens": int(usage[1]),
        "cached_tokens": int(usage[2]),
        "completion_tokens": int(usage[3]),
        "total_tokens": int(usage[4]),
        "cost_microusd": int(usage[5]),
    }


def _top_event_users(
    event_type: str,
    *,
    period: str,
    chat_id: int | None,
    target: bool,
    limit: int,
) -> list[dict[str, Any]]:
    params = _scope_params(period, chat_id) + (event_type, max(1, int(limit)))
    if target:
        query = (
            "SELECT target_user_id, "
            "MAX(COALESCE(target_user_name, 'ID ' || target_user_id)), SUM(value) "
            "FROM analytics_events "
            "WHERE (? IS NULL OR occurred_at >= ?) AND (? IS NULL OR chat_id = ?) "
            "AND event_type = ? AND target_user_id IS NOT NULL "
            "GROUP BY target_user_id "
            "ORDER BY SUM(value) DESC, target_user_id ASC LIMIT ?"
        )
    else:
        query = (
            "SELECT actor_id, MAX(COALESCE(actor_name, 'ID ' || actor_id)), SUM(value) "
            "FROM analytics_events "
            "WHERE (? IS NULL OR occurred_at >= ?) AND (? IS NULL OR chat_id = ?) "
            "AND event_type = ? AND actor_id IS NOT NULL "
            "AND actor_kind IN ('user','owner') "
            "GROUP BY actor_id ORDER BY SUM(value) DESC, actor_id ASC LIMIT ?"
        )
    rows = state._db_execute(
        query,
        params,
        fetch=True,
    )
    return [
        {"user_id": int(row[0]), "name": str(row[1]), "value": int(row[2])}
        for row in (rows or [])
    ]


def get_leaderboards(
    period: str = "7d",
    *,
    chat_id: int | None = None,
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    if not _ready():
        return {"messages": [], "replies": [], "bot_reactions": [], "cost": []}
    usage_params = _scope_params(period, chat_id) + (max(1, int(limit)),)
    usage_rows = state._db_execute(
        "SELECT user_id, MAX(user_name), SUM(cost_microusd) "
        "FROM analytics_llm_usage "
        "WHERE (? IS NULL OR occurred_at >= ?) AND (? IS NULL OR chat_id = ?) "
        "GROUP BY user_id ORDER BY SUM(cost_microusd) DESC, user_id ASC LIMIT ?",
        usage_params,
        fetch=True,
    )
    return {
        "messages": _top_event_users(
            "user_message", period=period, chat_id=chat_id, target=False, limit=limit
        ),
        "replies": _top_event_users(
            "assistant_reply", period=period, chat_id=chat_id, target=True, limit=limit
        ),
        "bot_reactions": _top_event_users(
            "bot_reaction", period=period, chat_id=chat_id, target=True, limit=limit
        ),
        "cost": [
            {"user_id": int(row[0]), "name": str(row[1]), "value": int(row[2])}
            for row in (usage_rows or [])
        ],
    }


def get_activity(period: str = "7d") -> dict[str, list[tuple[int, int]]]:
    if not _ready():
        return {"hours": [], "weekdays": []}
    cutoff = _cutoff(period)
    params = (cutoff, cutoff)
    hours = state._db_execute(
        "SELECT local_hour, SUM(value) FROM analytics_events "
        "WHERE (? IS NULL OR occurred_at >= ?) AND event_type='user_message' "
        "AND actor_kind IN ('user','owner') "
        "GROUP BY local_hour ORDER BY SUM(value) DESC, local_hour ASC",
        params,
        fetch=True,
    )
    weekdays = state._db_execute(
        "SELECT local_weekday, SUM(value) FROM analytics_events "
        "WHERE (? IS NULL OR occurred_at >= ?) AND event_type='user_message' "
        "AND actor_kind IN ('user','owner') "
        "GROUP BY local_weekday ORDER BY SUM(value) DESC, local_weekday ASC",
        params,
        fetch=True,
    )
    return {
        "hours": [(int(row[0]), int(row[1])) for row in (hours or [])],
        "weekdays": [(int(row[0]), int(row[1])) for row in (weekdays or [])],
    }


def get_recent_alerts(period: str = "7d", *, limit: int = 8) -> list[dict[str, Any]]:
    if not _ready():
        return []
    cutoff = _cutoff(period)
    params = (cutoff, cutoff, max(1, int(limit)))
    rows = state._db_execute(
        "SELECT MAX(occurred_at), category, MAX(message), COUNT(*) "
        "FROM analytics_alerts WHERE (? IS NULL OR occurred_at >= ?) "
        "GROUP BY fingerprint "
        "ORDER BY MAX(occurred_at) DESC LIMIT ?",
        params,
        fetch=True,
    )
    return [
        {
            "occurred_at": float(row[0]),
            "category": str(row[1]),
            "message": str(row[2]),
            "count": int(row[3]),
        }
        for row in (rows or [])
    ]


def get_tracking_started_at(*, chat_id: int | None = None) -> float | None:
    if not _ready():
        return None
    normalized_chat_id = int(chat_id) if chat_id is not None else None
    params = (normalized_chat_id, normalized_chat_id)
    event_rows = state._db_execute(
        "SELECT MIN(occurred_at) FROM analytics_events "
        "WHERE (? IS NULL OR chat_id = ?)",
        params,
        fetch=True,
    )
    usage_rows = state._db_execute(
        "SELECT MIN(occurred_at) FROM analytics_llm_usage "
        "WHERE (? IS NULL OR chat_id = ?)",
        params,
        fetch=True,
    )
    values = [
        row[0]
        for row in ((event_rows or []) + (usage_rows or []))
        if row and row[0] is not None
    ]
    return min(float(value) for value in values) if values else None
