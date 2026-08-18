"""Telegram-native rendering for public chat stats and the owner dashboard."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from berangaria.analytics import store as analytics_store
from berangaria.config import TIMEZONE_NAME

PERIOD_LABELS = {
    "24h": "24 часа",
    "7d": "7 дней",
    "30d": "30 дней",
    "all": "всё время",
}
PAGE_LABELS = {
    "overview": "Обзор",
    "leaders": "Лидеры",
    "activity": "Активность",
    "errors": "Ошибки",
}
_WEEKDAYS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


def _name(value: str) -> str:
    return " ".join((value or "Без имени").split())[:40]


def _integer(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _cost(microusd: int) -> str:
    usd = int(microusd) / 1_000_000
    return f"${usd:.6f}" if usd < 0.01 else f"${usd:.4f}"


def _rank(rows: list[dict], *, cost: bool = False, empty: str = "пока пусто") -> str:
    if not rows:
        return f"  {empty}"
    lines = []
    for index, row in enumerate(rows, 1):
        value = _cost(row["value"]) if cost else _integer(row["value"])
        lines.append(f"  {index}. {_name(row['name'])} — {value}")
    return "\n".join(lines)


def _tracking_label(timestamp: float | None) -> str:
    if timestamp is None:
        return "Статистика начнёт собираться после обновления бота."
    local = datetime.fromtimestamp(timestamp, ZoneInfo(TIMEZONE_NAME))
    return f"Собирается с {local:%d.%m.%Y %H:%M} ({TIMEZONE_NAME})."


def render_top(chat_id: int) -> str:
    leaders = analytics_store.get_leaderboards("all", chat_id=chat_id, limit=10)
    started = analytics_store.get_tracking_started_at(chat_id=chat_id)
    return (
        "🏆 Топ текущего чата\n\n"
        "Сообщения:\n"
        f"{_rank(leaders['messages'])}\n\n"
        "Ответы Бер пользователям:\n"
        f"{_rank(leaders['replies'])}\n\n"
        "Реакции Бер на сообщения пользователей:\n"
        f"{_rank(leaders['bot_reactions'])}\n\n"
        f"{_tracking_label(started)}"
    )


def dashboard_keyboard(page: str, period: str) -> InlineKeyboardMarkup:
    period_buttons = [
        InlineKeyboardButton(
            ("• " if value == period else "") + label,
            callback_data=f"dashboard:{page}:{value}",
        )
        for value, label in (("24h", "24ч"), ("7d", "7д"), ("30d", "30д"), ("all", "Всё"))
    ]
    page_buttons = [
        InlineKeyboardButton(
            ("• " if value == page else "") + label,
            callback_data=f"dashboard:{value}:{period}",
        )
        for value, label in PAGE_LABELS.items()
    ]
    return InlineKeyboardMarkup([period_buttons, page_buttons[:2], page_buttons[2:]])


def parse_dashboard_callback(data: str) -> tuple[str, str]:
    parts = (data or "").split(":")
    if len(parts) != 3 or parts[0] != "dashboard":
        return "overview", "7d"
    page = parts[1] if parts[1] in PAGE_LABELS else "overview"
    period = parts[2] if parts[2] in PERIOD_LABELS else "7d"
    return page, period


def _render_overview(period: str) -> str:
    overview = analytics_store.get_overview(period)
    if not overview:
        return "📊 Панель владельца\n\nСтатистика пока недоступна."
    prompt = overview["prompt_tokens"]
    cached = overview["cached_tokens"]
    cache_share = round(cached * 100 / prompt) if prompt else 0
    return (
        f"📊 Панель владельца · {PERIOD_LABELS[period]}\n\n"
        f"Расходы чат-модели: {_cost(overview['cost_microusd'])}\n"
        f"Запросов к модели: {_integer(overview['requests'])}\n"
        f"Токены: {_integer(overview['total_tokens'])}\n"
        f"Кэш prompt: {_integer(cached)} ({cache_share}%)\n\n"
        f"Сообщений пользователей: {_integer(overview['messages'])}\n"
        f"Активных пользователей: {_integer(overview['active_users'])}\n"
        f"Ответов Бер: {_integer(overview['assistant_replies'])}\n"
        f"Реакций Бер: {_integer(overview['bot_reactions'])}\n"
        f"Реакций на Бер: {_integer(overview['incoming_reactions'])}"
    )


def _render_leaders(period: str) -> str:
    leaders = analytics_store.get_leaderboards(period, limit=5)
    return (
        f"🏆 Лидеры · {PERIOD_LABELS[period]}\n\n"
        "По сообщениям:\n"
        f"{_rank(leaders['messages'])}\n\n"
        "По расходам чат-модели:\n"
        f"{_rank(leaders['cost'], cost=True)}\n\n"
        "По ответам Бер:\n"
        f"{_rank(leaders['replies'])}\n\n"
        "По реакциям Бер:\n"
        f"{_rank(leaders['bot_reactions'])}"
    )


def _render_activity(period: str) -> str:
    activity = analytics_store.get_activity(period)
    hours = activity["hours"][:5]
    weekdays = activity["weekdays"]
    hour_lines = (
        "\n".join(f"  {hour:02d}:00–{hour:02d}:59 — {_integer(count)}" for hour, count in hours)
        or "  пока пусто"
    )
    weekday_lines = (
        "\n".join(f"  {_WEEKDAYS[day]} — {_integer(count)}" for day, count in weekdays)
        or "  пока пусто"
    )
    return (
        f"🕒 Активность · {PERIOD_LABELS[period]}\n\n"
        f"Самые активные часы ({TIMEZONE_NAME}):\n{hour_lines}\n\n"
        f"Дни недели:\n{weekday_lines}"
    )


def _render_errors(period: str) -> str:
    alerts = analytics_store.get_recent_alerts(period, limit=8)
    if not alerts:
        body = "Критических ошибок не зафиксировано."
    else:
        lines = []
        timezone = ZoneInfo(TIMEZONE_NAME)
        for alert in alerts:
            when = datetime.fromtimestamp(alert["occurred_at"], timezone)
            repeated = f" ×{alert['count']}" if alert["count"] > 1 else ""
            message = _name(alert["message"][:120])
            lines.append(f"• {when:%d.%m %H:%M} [{alert['category']}]{repeated}\n  {message}")
        body = "\n".join(lines)
    return f"⚠️ Ошибки · {PERIOD_LABELS[period]}\n\n{body}"


def render_dashboard(page: str = "overview", period: str = "7d") -> tuple[str, InlineKeyboardMarkup]:
    page = page if page in PAGE_LABELS else "overview"
    period = period if period in PERIOD_LABELS else "7d"
    renderers = {
        "overview": _render_overview,
        "leaders": _render_leaders,
        "activity": _render_activity,
        "errors": _render_errors,
    }
    text = renderers[page](period)
    started = analytics_store.get_tracking_started_at()
    if page == "overview":
        text += f"\n\n{_tracking_label(started)}"
    return text, dashboard_keyboard(page, period)

