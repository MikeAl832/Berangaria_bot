import asyncio

import pytest
from telegram.ext import CallbackQueryHandler, MessageReactionHandler

from berangaria import app as main


def _handlers(app):
    return [handler for group in app.handlers.values() for handler in group]


def test_passive_handlers_do_not_hold_the_update_slot(monkeypatch):
    """Пассивные хендлеры не должны блокировать единственный слот обновлений.

    Приложение работает с дефолтным SimpleUpdateProcessor(1). Оба этих хендлера
    берут turn-lock чата, поэтому блокирующая регистрация морозит весь бот на
    время LLM-хода в любом одном чате.
    """
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


@pytest.mark.parametrize(
    "command",
    ["start", "clear", "stats", "top", "dashboard", "random", "summarize"],
)
def test_documented_commands_are_registered(command, monkeypatch):
    app = main.build_telegram_application()
    main.register_handlers(app)

    registered = set()
    for handler in _handlers(app):
        registered |= set(getattr(handler, "commands", None) or ())
    assert command in registered


def test_reaction_handler_is_registered(monkeypatch):
    app = main.build_telegram_application()
    main.register_handlers(app)

    assert any(isinstance(h, MessageReactionHandler) for h in _handlers(app))


def test_dashboard_callback_handler_is_registered(monkeypatch):
    app = main.build_telegram_application()
    main.register_handlers(app)

    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback is main.dashboard_callback
        for handler in _handlers(app)
    )


def test_build_application_uses_cloud_defaults_without_override(monkeypatch):

    app = main.build_telegram_application()

    assert app.bot.base_url == "https://api.telegram.org/bottest-token"
    assert app.bot.base_file_url == "https://api.telegram.org/file/bottest-token"
    assert app.bot.local_mode is False


def test_readiness_is_registered_as_post_init(monkeypatch, caplog):
    started = {}

    async def _fake_start(application):
        started["app"] = application
        return None

    # post_init now boots the user bridge; do not start Telethon from this test.
    monkeypatch.setattr(main, "start_user_bridge", _fake_start)
    app = main.build_telegram_application()

    assert app.post_init is main._telegram_post_init

    with caplog.at_level("INFO"):
        asyncio.run(app.post_init(app))

    assert "Бот запущен" in caplog.text
    assert started["app"] is app


def test_post_init_bridge_boot_is_fail_open(monkeypatch):

    async def _boom(_application):
        raise RuntimeError("bridge boot exploded")

    monkeypatch.setattr(main, "start_user_bridge", _boom)
    app = main.build_telegram_application()
    asyncio.run(app.post_init(app))


def test_build_application_raises_media_http_timeouts(monkeypatch):
    """Outgoing media uploads need headroom beyond PTB defaults."""

    app = main.build_telegram_application()
    request = app.bot.request
    timeout = request._client.timeout

    assert request.read_timeout == 60.0
    assert timeout.read == 60.0
    assert timeout.write == 60.0
    assert timeout.connect == 10.0
    assert timeout.pool == 5.0
    assert request._media_write_timeout == 120.0


def test_get_updates_http_client_uses_the_same_timeout_budget(monkeypatch):
    """getUpdates is a separate HTTPX client; #15 only raised the outgoing one."""

    app = main.build_telegram_application()
    get_updates_request = app.bot._request[0]
    timeout = get_updates_request._client.timeout

    assert get_updates_request.read_timeout == 60.0
    assert timeout.read == 60.0
    assert timeout.write == 60.0
    assert timeout.connect == 10.0
    assert timeout.pool == 5.0
    assert get_updates_request is not app.bot.request


def test_media_client_shutdown_is_registered():
    app = main.build_telegram_application()
    assert app.post_shutdown is main.stop_media_downloader
