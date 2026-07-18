from handlers import _build_memory_text


def test_memory_text_keeps_only_user_text_when_media_is_present():
    text = _build_memory_text(
        "Я использую Fedora",
        [("video", "на видео виден компьютер с Ubuntu")],
    )

    assert text == "Я использую Fedora"


def test_media_only_message_has_no_long_term_memory_source():
    text = _build_memory_text(
        "",
        [("image", "на изображении человек с видеокартой RTX 5090")],
    )

    assert text == ""


def test_forwarded_text_has_no_long_term_memory_source():
    text = _build_memory_text(
        "Я живу в Москве",
        [],
        is_forwarded=True,
    )

    assert text == ""
