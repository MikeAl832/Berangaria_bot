#!/usr/bin/env python
"""
Smoke-test Fish Audio TTS via berangaria.media.tts (same path the bot uses).

Requires FISH_API_KEY + FISH_VOICE_ID in .env (never commit keys).

Examples:
    python scripts/fish_tts_smoke.py
    python scripts/fish_tts_smoke.py --suite
    python scripts/fish_tts_smoke.py --text "Ну да." --emotion sarcastic
    python scripts/fish_tts_smoke.py --format mp3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUITE_TEXTS: list[tuple[str, str, str | None]] = [
    ("01_swine", "Ну да. Конечно. И свиньи полетели.", "sarcastic"),
    ("02_wrong", "Коротко: ты не прав, и это даже не интересно.", "disdainful"),
    ("03_record", "Ого. Целых три слова смысла на абзац. Рекорд дня.", "bored"),
    ("04_code", "Я код. И мне от этого только спокойнее.", "calm"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fish Audio TTS smoke test (bot client)")
    ap.add_argument("--text", default="Привет. Это дымовой тест голосового Berangaria.")
    ap.add_argument(
        "--emotion",
        default=None,
        help="optional emotion (calm/sarcastic/.../none); omit for config default",
    )
    ap.add_argument("--format", dest="fmt", default=None, help="override format (opus/mp3/…)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--suite", action="store_true", help="four deadpan sample lines")
    args = ap.parse_args()

    # Import after path setup so config loads .env from project root.
    from berangaria.media.tts import (
        TTSError,
        is_tts_ready,
        synthesize_speech,
        voice_filename,
    )
    from berangaria.config import (
        FISH_VOICE_ID,
        TTS_FORMAT,
        TTS_MODEL,
        TTS_ENABLED,
    )

    if not is_tts_ready():
        print(
            "TTS not ready. Set FISH_API_KEY and FISH_VOICE_ID in .env, "
            f"and tts_enabled in config.yaml (TTS_ENABLED={TTS_ENABLED}).",
            file=sys.stderr,
        )
        return 2

    out_dir = ROOT / "scripts" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    use_fmt = (args.fmt or TTS_FORMAT or "opus").strip().lower()

    jobs: list[tuple[str, str, object, Path]]
    if args.suite:
        jobs = [
            (label, text, emo, out_dir / f"frieren_{label}.{use_fmt if use_fmt != 'opus' else 'ogg'}")
            for label, text, emo in SUITE_TEXTS
        ]
    else:
        out = args.out or (out_dir / f"fish_tts_smoke.{use_fmt if use_fmt != 'opus' else 'ogg'}")
        emotion = args.emotion if args.emotion is not None else ...
        jobs = [("single", args.text, emotion, out)]

    print(f"model    : {TTS_MODEL}")
    print(f"voice    : {FISH_VOICE_ID}")
    print(f"format   : {use_fmt}")
    print(f"jobs     : {len(jobs)}")
    print("---")

    failed = 0
    for label, text, emotion, out in jobs:
        print(f"[{label}] {text[:80]!r} emotion={emotion!r}")
        t0 = time.perf_counter()
        try:
            audio = synthesize_speech(text, emotion=emotion, fmt=use_fmt)
        except TTSError as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            failed += 1
            continue
        elapsed = time.perf_counter() - t0
        out.write_bytes(audio)
        print(f"  ok {elapsed:.2f}s  {len(audio):,} bytes → {out}  ({voice_filename(use_fmt)})")

    if failed:
        print(f"failed {failed}/{len(jobs)}", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
