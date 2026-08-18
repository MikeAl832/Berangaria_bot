"""Structured logging for LLM payloads, responses, tokens, and cost."""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def log_request(messages: list[dict], *, enabled: bool) -> None:
    """Write the complete provider payload when full audit logging is enabled."""
    if not enabled:
        return
    logger.debug("[cyan]%s[/]", "=" * 80)
    logger.debug("[bright_green]📤 ЗАПРОС К МОДЕЛИ:[/]")
    logger.debug("[cyan]%s[/]", "=" * 80)
    for index, message in enumerate(messages, 1):
        role = message["role"]
        content = str(message.get("content", ""))
        role_color = {
            "system": "magenta",
            "user": "cyan",
            "assistant": "green",
        }.get(role, "white")
        logger.debug(
            "\n[yellow][%s][/] Role: [%s]%s[/]",
            index,
            role_color,
            role.upper(),
        )
        logger.debug("Length: [dim]%s символов[/]", len(content))
        logger.debug("[%s]Content:[/]", role_color)
        suffix = "..." if len(content) > 2000 else ""
        logger.debug("[dim]%s%s[/]", content[:2000], suffix)
        logger.debug("[dim]%s[/]", "-" * 80)
    logger.debug("[cyan]%s[/]", "=" * 80)


def log_response(reply: str, finish_reason: str, *, enabled: bool) -> None:
    """Write the full final model response when audit logging is enabled."""
    if not enabled:
        return
    logger.debug("[blue]%s[/]", "=" * 80)
    logger.debug("[bright_green]📥 ОТВЕТ ОТ МОДЕЛИ:[/]")
    logger.debug("[blue]%s[/]", "=" * 80)
    finish_color = {
        "stop": "green",
        "length": "yellow",
        "tool_calls": "cyan",
    }.get(finish_reason, "white")
    logger.debug("Finish reason: [%s]%s[/]", finish_color, finish_reason)
    logger.debug("Content length: [dim]%s символов[/]", len(reply))
    logger.debug("[green]Content:[/]")
    logger.debug("[bright_green]%s[/]", reply)
    logger.debug("[blue]%s[/]", "=" * 80)


def record_usage(
    usage: dict,
    *,
    key: str,
    chat_tokens: dict,
    estimate_request_cost: Callable[..., float],
) -> float:
    """Update live token totals, log cost, and return that request cost."""
    if not usage:
        return 0.0
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    details = usage.get("prompt_tokens_details", {})
    cached_tokens = details.get("cached_tokens", 0)
    cache_write_tokens = details.get("cache_write_tokens", 0)
    chat_tokens[key] = total_tokens
    logger.info(
        "📊 Токены: запрос=[cyan]%s[/] (кэш=[cyan]%s[/]), "
        "ответ=[cyan]%s[/], всего=[bright_green]%s[/]",
        prompt_tokens,
        cached_tokens,
        completion_tokens,
        total_tokens,
    )
    total_cost = estimate_request_cost(
        usage,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    logger.info("💰 Стоимость запроса: [bright_green]$%.6f[/]", total_cost)
    return total_cost
