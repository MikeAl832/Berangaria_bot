"""Catalogue load + embed-text formula for the v3 sticker pack."""
import json
from pathlib import Path

from berangaria.stickers.store import load_sticker_records, sticker_text


def test_load_sticker_records_json_array(tmp_path):
    path = tmp_path / "pack.json"
    path.write_text(
        json.dumps(
            [
                {"filename": "a", "emotion": "радость", "description": "улыбка"},
                {"filename": "b", "emotion": "грусть"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = load_sticker_records(path)
    assert len(rows) == 2
    assert rows[0]["filename"] == "a"


def test_load_sticker_records_jsonl(tmp_path):
    path = tmp_path / "pack.jsonl"
    path.write_text(
        '{"filename": "x", "emotion": "шок"}\n'
        '{"filename": "y", "emotion": "ирония"}\n',
        encoding="utf-8",
    )
    rows = load_sticker_records(path)
    assert [r["filename"] for r in rows] == ["x", "y"]


def test_sticker_text_prefers_occasion_fields_over_visual_only():
    text = sticker_text(
        {
            "emotion": "отказ",
            "secondary_emotions": ["ирония", "сомнение"],
            "action": "машет рукой",
            "use_cases": ["вежливый отказ", "не хочу"],
            "situation": "Когда твердо отказываешься от предложения.",
            "keywords": ["отказ", "ладонь"],
            "character": "Omar Sy",
            "franchise": "meme",  # noise — must not appear
            "description": "Мужчина поднял ладонь.",
            "text_on_sticker": "1+1",
        }
    )
    assert "отказ" in text
    assert "ирония" in text
    assert "вежливый отказ" in text
    assert "не хочу" in text
    assert "твердо отказываешься" in text
    assert "Omar Sy" in text
    assert "meme" not in text
    assert "текст: 1+1" in text
    # Occasion-ish content should appear before the pure visual line.
    assert text.index("вежливый отказ") < text.index("Мужчина поднял ладонь")


def test_repo_catalogue_loads_if_present():
    path = Path("data/stickers_clean.json")
    if not path.is_file():
        return
    rows = load_sticker_records(path)
    assert len(rows) >= 100
    assert all(r.get("filename") for r in rows)
    sample = sticker_text(rows[0])
    assert rows[0].get("emotion", "") in sample or "emotion" not in rows[0]
