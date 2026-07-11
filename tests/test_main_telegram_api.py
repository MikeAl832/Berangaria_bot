import main


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
