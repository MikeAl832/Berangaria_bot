from datetime import timedelta

from utils import escape_user_text, get_video_duration


def test_escape_user_text_neutralizes_service_tags():
    # Служебные теги в тексте пользователя не должны выглядеть как системные
    out = escape_user_text("[Message: привет]")
    assert "[Message:" not in out
    assert "привет" in out


def test_escape_user_text_empty():
    assert escape_user_text("") == ""
    assert escape_user_text(None) == ""


def test_escape_user_text_plain_passthrough():
    # Обычный текст без служебных тегов не калечится
    assert escape_user_text("просто текст") == "просто текст"


def test_get_video_duration_int():
    class Obj:
        duration = 42
    assert get_video_duration(Obj()) == 42.0


def test_get_video_duration_timedelta():
    class Obj:
        duration = timedelta(seconds=15)
    assert get_video_duration(Obj()) == 15.0


def test_get_video_duration_none():
    class Obj:
        duration = None
    assert get_video_duration(Obj()) == 0.0


def test_get_video_duration_missing_attr():
    class Obj:
        pass
    assert get_video_duration(Obj()) == 0.0
