from llm_client import _chunk_lines_by_chars
from handlers import _build_memory_text


def test_chunks_respect_budget():
    lines = [f"Аня: {'a' * 50}" for _ in range(10)]
    chunks = _chunk_lines_by_chars(lines, 200)
    for c in chunks:
        # либо влезли в бюджет, либо это одиночная строка (не разбиваемая дальше)
        assert sum(len(l) for l in c) <= 200 or len(c) == 1


def test_oversized_single_line_is_split():
    big = ["x" * 500]
    chunks = _chunk_lines_by_chars(big, 200)
    assert len(chunks) == 3
    assert all(len(c[0]) <= 200 for c in chunks)


def test_empty_input():
    assert _chunk_lines_by_chars([], 200) == []


def test_all_lines_preserved():
    lines = [f"line{i}" for i in range(7)]
    chunks = _chunk_lines_by_chars(lines, 20)
    flat = [l for c in chunks for l in c]
    assert flat == lines  # ничего не потеряли и не переставили


def test_memory_text_includes_bounded_media_description():
    text = _build_memory_text("", [("image", "очень важное описание " * 100)])

    assert text.startswith("Изображение:")
    assert len(text) <= 600


def test_memory_text_keeps_user_text_with_media():
    text = _build_memory_text("смотри", [("video", "кот прыгает по столу")])

    assert "смотри" in text
    assert "Видео:" in text
