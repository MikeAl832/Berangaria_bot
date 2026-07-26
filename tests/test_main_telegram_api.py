import pytest
from telegram.ext import MessageReactionHandler

import main


def _handlers(app):
    return [handler for group in app.handlers.values() for handler in group]


def test_passive_handlers_do_not_hold_the_update_slot(monkeypatch):
    """Пассивные хендлеры не должны блокировать единственный слот обновлений.

    Приложение работает с дефолтным SimpleUpdateProcessor(1). Оба этих хендлера
    берут turn-lock чата, поэтому блокирующая регистрация морозит весь бот на
    время LLM-хода в любом одном чате.
    """
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_BASE_URL", "")
    app = main.build_telegram_application()
    main.register_handlers(app)

    passive = {main.handle_message_reaction, main.handle_chat_event}
    found = {}
    for handler in _handlers(app):
        callback = getattr(handler, "callback", None)
        if callback in passive:
            found[callback] = handler

    assert found.keys() == passive, "оба пассивных хендлера должны быть зарегистрированы"
    for callback, handler in found.items():
        assert handler.block is False, f"{callback.__name__} зарегистрирован блокирующим"


def test_message_intake_stays_serialized(monkeypatch):
    """Обратная сторона: приём сообщений остаётся блокирующим.

    Debounce-буфер собирает combined_text в порядке прихода, поэтому
    распараллеливать сам приём (в т.ч. через concurrent_updates) нельзя.
    """
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_BASE_URL", "")
    app = main.build_telegram_application()
    main.register_handlers(app)

    intake = {main.handle_message, main.handle_media, main.handle_video,
              main.handle_sticker, main.handle_voice}
    seen = set()
    for handler in _handlers(app):
        callback = getattr(handler, "callback", None)
        if callback in intake:
            seen.add(callback)
            assert handler.block is not False, (
                f"{callback.__name__} не должен обрабатываться конкурентно"
            )
    assert seen == intake


@pytest.mark.parametrize("command", ["start", "clear", "stats", "random", "summarize"])
def test_documented_commands_are_registered(command, monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_BASE_URL", "")
    app = main.build_telegram_application()
    main.register_handlers(app)

    registered = set()
    for handler in _handlers(app):
        registered |= set(getattr(handler, "commands", None) or ())
    assert command in registered


def test_reaction_handler_is_registered(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_BASE_URL", "")
    app = main.build_telegram_application()
    main.register_handlers(app)

    assert any(isinstance(h, MessageReactionHandler) for h in _handlers(app))


def test_build_application_uses_local_bot_api(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_BASE_URL", "http://127.0.0.1:8081")
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_BASE_FILE_URL", "http://127.0.0.1:8081/file")
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_LOCAL_MODE", True)

    app = main.build_telegram_application()

    assert app.bot.base_url == "http://127.0.0.1:8081/bottest-token"
    assert app.bot.base_file_url == "http://127.0.0.1:8081/file/bottest-token"
    assert app.bot.local_mode is True


def test_build_application_uses_cloud_defaults_without_override(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_BOT_API_BASE_URL", "")

    app = main.build_telegram_application()

    assert app.bot.base_url == "https://api.telegram.org/bottest-token"
    assert app.bot.base_file_url == "https://api.telegram.org/file/bottest-token"
    assert app.bot.local_mode is False
