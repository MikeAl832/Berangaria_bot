import asyncio
from types import SimpleNamespace

from telegram.error import BadRequest

from berangaria.chat.reply_delivery import DeliveryRuntime, deliver


class _Bot:
    def __init__(self, *, reject_quote: bool = False):
        self.calls = []
        self.reject_quote = reject_quote

    async def send_message(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_quote and len(self.calls) == 1:
            raise BadRequest("quote not found")
        return SimpleNamespace(message_id=700 + len(self.calls))


def _runtime(bot: _Bot) -> DeliveryRuntime:
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=100),
        message=SimpleNamespace(message_id=9, message_thread_id=None),
    )
    return DeliveryRuntime(
        update=update,
        context=SimpleNamespace(bot=bot),
        clean_reply=lambda text: text,
        is_parse_error=lambda error: "parse" in str(error).lower(),
        multi_message_delay_seconds=lambda *args, **kwargs: 0,
    )


def test_deliver_uses_reply_parameters_for_exact_quote():
    bot = _Bot()

    message_id = asyncio.run(
        deliver(
            "**Ответ**",
            42,
            None,
            _runtime(bot),
            quote="точные слова",
            quote_position=5,
        )
    )

    assert message_id == 701
    call = bot.calls[0]
    assert "reply_to_message_id" not in call
    assert call["parse_mode"] == "HTML"
    assert call["text"] == "<b>Ответ</b>"
    assert call["reply_parameters"].message_id == 42
    assert call["reply_parameters"].quote == "точные слова"
    assert call["reply_parameters"].quote_position == 5


def test_deliver_retries_as_whole_message_reply_when_quote_is_rejected():
    bot = _Bot(reject_quote=True)

    message_id = asyncio.run(
        deliver(
            "Ответ",
            42,
            None,
            _runtime(bot),
            quote="устаревшая цитата",
            quote_position=0,
        )
    )

    assert message_id == 702
    assert "reply_parameters" in bot.calls[0]
    assert "reply_parameters" not in bot.calls[1]
    assert bot.calls[1]["reply_to_message_id"] == 42
    assert bot.calls[1]["allow_sending_without_reply"] is True
