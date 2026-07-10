from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from utils import (
    escape_user_text,
    get_video_duration,
    now_local,
    next_summary_run,
    is_url_only_text,
    is_low_signal_user_text,
    strip_tiktok_urls,
)
from config import SUMMARY_HOURS, BOT_TZ


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


def test_is_url_only_text():
    assert is_url_only_text("https://vt.tiktok.com/ZSCKeAjpT/")
    assert is_url_only_text("https://a.com/x https://b.com/y")
    assert not is_url_only_text("смотри https://example.com/x это")
    assert not is_url_only_text("просто текст")


def test_strip_tiktok_urls():
    assert strip_tiktok_urls("https://vt.tiktok.com/ZSCKeAjpT/") == ""
    assert strip_tiktok_urls("смотри https://www.tiktok.com/@x/video/1 смешно") == "смотри смешно"
    assert strip_tiktok_urls("vm.tiktok.com/abc") == ""
    assert strip_tiktok_urls("https://example.com/x") == "https://example.com/x"


def test_is_low_signal_user_text():
    assert is_low_signal_user_text("ок")
    assert is_low_signal_user_text("https://example.com/page")
    assert is_low_signal_user_text("  ")
    assert not is_low_signal_user_text("сегодня купил новую видеокарту")


def test_next_summary_run_picks_afternoon_slot():
    # 10:00 МСК → ближайший 14:00 того же дня
    now = datetime(2026, 7, 8, 10, 0, 0, tzinfo=BOT_TZ)
    nxt = next_summary_run(now)
    assert nxt.hour == 14
    assert nxt.day == 8


def test_next_summary_run_rolls_to_next_morning():
    # 16:00 МСК → следующий 05:00
    now = datetime(2026, 7, 8, 16, 0, 0, tzinfo=BOT_TZ)
    nxt = next_summary_run(now)
    assert nxt.hour == 5
    assert nxt.day == 9
    assert set(SUMMARY_HOURS) >= {5, 14}


def test_now_local_is_bot_tz():
    n = now_local()
    assert n.tzinfo is not None
    # Смещение должно совпадать с BOT_TZ (МСК = UTC+3)
    assert n.utcoffset() == datetime.now(BOT_TZ).utcoffset()
