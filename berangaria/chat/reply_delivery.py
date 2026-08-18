"""Telegram delivery of final model replies and multi-message bursts."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from telegram.error import BadRequest

from berangaria.chat.reply_formatting import (
    markdown_to_html,
    split_for_telegram,
    strip_markdown,
)

logger = logging.getLogger(__name__)


class ReplyDeliveryError(RuntimeError):
    """Telegram did not confirm final delivery, so the turn cannot commit."""


@dataclass(frozen=True)
class DeliveryRuntime:
    """Per-turn objects and patchable formatting boundaries."""

    update: Any
    context: Any
    clean_reply: Callable[[str], str]
    is_parse_error: Callable[[BaseException], bool]
    multi_message_delay_seconds: Callable[..., float]


async def delete_turn_status(turn: Any) -> None:
    """Best-effort removal of the transient tool/stream status message."""
    if turn.status_message is None:
        return
    try:
        await turn.status_message.delete()
    except Exception:
        pass
    finally:
        turn.status_message = None


async def deliver(
    text: str,
    target_mid: int | None,
    status_message: Any,
    runtime: DeliveryRuntime,
) -> int | None:
    """Deliver one final reply, returning the first Telegram message ID."""
    update = runtime.update
    context = runtime.context
    reply_html = markdown_to_html(text)
    reply_plain = strip_markdown(text)
    chat_id = update.effective_chat.id
    thread_id = getattr(update.message, "message_thread_id", None)

    if status_message is not None:
        if target_mid == update.message.message_id and len(reply_html) <= 4096:
            try:
                await status_message.edit_text(reply_html, parse_mode="HTML")
                return status_message.message_id
            except BadRequest as error:
                logger.warning(
                    "⚠️ [yellow]Правка статуса с HTML не прошла:[/] %s",
                    error,
                )
            except Exception as error:
                logger.error(
                    "❌ [red]Правка статуса оборвалась неоднозначно:[/] %s",
                    error,
                )
                raise ReplyDeliveryError(
                    "Telegram не подтвердил правку статусного сообщения"
                ) from error
        try:
            await status_message.delete()
        except Exception:
            pass

    async def send_raw(body: str, html: bool) -> int:
        kwargs = {"chat_id": chat_id, "text": body}
        if thread_id is not None:
            kwargs["message_thread_id"] = thread_id
        if target_mid is not None:
            kwargs["reply_to_message_id"] = target_mid
            kwargs["allow_sending_without_reply"] = True
        if html:
            kwargs["parse_mode"] = "HTML"
        sent = await context.bot.send_message(**kwargs)
        return sent.message_id

    if len(reply_html) <= 4096:
        try:
            return await send_raw(reply_html, True)
        except BadRequest as error:
            if not runtime.is_parse_error(error):
                raise
            logger.warning(
                "⚠️ [yellow]HTML не распарсился, отправляю как текст:[/] %s",
                error,
            )
            return await send_raw(reply_plain, False)

    first_mid = None
    for chunk in split_for_telegram(reply_plain):
        kwargs = {"chat_id": chat_id, "text": chunk}
        if thread_id is not None:
            kwargs["message_thread_id"] = thread_id
        if first_mid is None and target_mid is not None:
            kwargs["reply_to_message_id"] = target_mid
            kwargs["allow_sending_without_reply"] = True
        try:
            sent = await context.bot.send_message(**kwargs)
        except Exception as error:
            if first_mid is None:
                raise
            logger.error(
                "❌ [red]Длинный ответ оборвался после первого чанка:[/] %s",
                error,
            )
            return first_mid
        if first_mid is None:
            first_mid = sent.message_id
    return first_mid


async def deliver_multi(
    messages: list[str],
    target_mid: int | None,
    status_message: Any,
    runtime: DeliveryRuntime,
) -> tuple[int | None, list[str]]:
    """Deliver a short bubble sequence, preserving confirmed partial delivery."""
    update = runtime.update
    context = runtime.context
    cleaned: list[str] = []
    for raw in messages or []:
        try:
            piece = runtime.clean_reply(raw)
        except Exception:
            piece = (raw or "").strip()
        if piece:
            cleaned.append(piece)

    if not cleaned:
        if status_message is not None:
            try:
                await status_message.delete()
            except Exception:
                pass
        return None, []
    if len(cleaned) == 1:
        message_id = await deliver(
            cleaned[0], target_mid, status_message, runtime
        )
        return message_id, cleaned

    if status_message is not None:
        try:
            await status_message.delete()
        except Exception:
            pass

    chat_id = update.effective_chat.id
    thread_id = getattr(update.message, "message_thread_id", None)
    first_mid = None
    delivered: list[str] = []
    slept_total = 0.0

    for index, text in enumerate(cleaned):
        if index > 0:
            delay = runtime.multi_message_delay_seconds(
                text, slept_total=slept_total
            )
            if delay > 0:
                try:
                    await update.message.chat.send_action(action="typing")
                except Exception:
                    pass
                await asyncio.sleep(delay)
                slept_total += delay

        reply_html = markdown_to_html(text)
        reply_plain = strip_markdown(text)
        use_html = len(reply_html) <= 4096
        body = reply_html if use_html else reply_plain

        async def send_message(body: str, html: bool):
            kwargs = {"chat_id": chat_id, "text": body}
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            if first_mid is None and target_mid is not None:
                kwargs["reply_to_message_id"] = target_mid
                kwargs["allow_sending_without_reply"] = True
            if html:
                kwargs["parse_mode"] = "HTML"
            return await context.bot.send_message(**kwargs)

        try:
            try:
                sent = await send_message(body, use_html)
            except BadRequest as error:
                if use_html and runtime.is_parse_error(error):
                    logger.warning(
                        "⚠️ [yellow]HTML multi-bubble не распарсился, plain:[/] %s",
                        error,
                    )
                    sent = await send_message(reply_plain, False)
                else:
                    raise
        except Exception as error:
            if first_mid is None:
                raise
            logger.error(
                "❌ [red]Серия сообщений оборвалась после %s:[/] %s",
                len(delivered),
                error,
            )
            return first_mid, delivered

        if first_mid is None:
            first_mid = sent.message_id
        delivered.append(text)

    return first_mid, delivered
