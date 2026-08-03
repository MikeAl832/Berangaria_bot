#!/usr/bin/env python
"""
Sync stickers into Qdrant.

Reads a catalogue (default data/stickers_clean.json — JSON array; .jsonl still works),
compares to the collection, and upserts ONLY missing stickers (deterministic point_id
from Telegram file_id). Workflow: edit the catalogue, run this command — only new
file_ids are embedded. A second run with no new rows is a no-op. Rows removed from
the file are NOT deleted from Qdrant unless you pass --recreate.

Embeddings use the same Gemini model as mem0. Batching (1 HTTP call per --batch-size)
saves quota; use --limit to chunk a large first load.

Examples (inside the bot container):
    docker compose exec bot python scripts/build_sticker_index.py
    docker compose exec bot python scripts/build_sticker_index.py --recreate   # full rewrite
    docker compose exec bot python scripts/build_sticker_index.py --dry-run

From the host (Qdrant on 127.0.0.1:6333):
    QDRANT_HOST=localhost python scripts/build_sticker_index.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# The script lives in scripts/; the package sits at the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_records(path):
    from berangaria.stickers.store import load_sticker_records

    return load_sticker_records(path)


def main():
    ap = argparse.ArgumentParser(description="Sync stickers to Qdrant (missing only, unless --recreate)")
    ap.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "stickers_clean.json"),
        help="sticker catalogue (.json array or .jsonl)",
    )
    ap.add_argument("--limit", type=int, default=None, help="залить не больше N новых за прогон (беречь квоту)")
    ap.add_argument("--batch-size", type=int, default=50, help="стикеров на один HTTP-запрос эмбеддинга")
    ap.add_argument("--sleep", type=float, default=1.5, help="пауза между запросами, сек (троттлинг под лимиты)")
    ap.add_argument("--recreate", action="store_true", help="пересоздать коллекцию с нуля (сотрёт индекс)")
    ap.add_argument("--dry-run", action="store_true", help="ничего не слать: показать, сколько нового и оценку")
    ap.add_argument("--qdrant-host", default=None, help="переопределить хост Qdrant (напр. localhost)")
    args = ap.parse_args()

    if args.qdrant_host:
        os.environ["QDRANT_HOST"] = args.qdrant_host

    # Импортируем после возможного переопределения QDRANT_HOST
    from berangaria.stickers.store import (
        sticker_text, get_client, ensure_collection, upsert_stickers,
        collection_count, missing_records, STICKER_COLLECTION,
    )

    if not os.path.exists(args.input):
        print(f"❌ Файл не найден: {args.input}")
        sys.exit(1)

    records = load_records(args.input)
    client = get_client()

    if args.recreate and not args.dry_run:
        ensure_collection(client, recreate=True)

    in_collection = collection_count(client)
    new = missing_records(records, client)
    if args.limit:
        new = new[:args.limit]

    print(f"📄 Файл: {args.input} — {len(records)} записей")
    print(f"📦 В коллекции '{STICKER_COLLECTION}' сейчас: {in_collection} точек")
    print(f"🆕 Недостающих к заливке: {len(new)}" + (f" (ограничено --limit {args.limit})" if args.limit else ""))

    if not new:
        print("✅ Всё уже синхронизировано — новых стикеров нет.")
        return

    est_tokens = sum(len(sticker_text(r)) for r in new) // 4
    n_requests = (len(new) + args.batch_size - 1) // args.batch_size
    print(f"🔢 Оценка: ~{n_requests} HTTP-запросов (батч={args.batch_size}), ~{est_tokens} токенов")
    print("   Лимиты Gemini: 100 RPM / 30k TPM / 1000 RPD")
    print(f"📝 Пример: {sticker_text(new[0])[:120]!r}")

    if args.dry_run:
        print("\n🧪 dry-run: реальные запросы не отправлялись.")
        return

    def progress(done, total):
        print(f"  … {done}/{total}", flush=True)

    n = upsert_stickers(
        new, client=client,
        batch_size=args.batch_size, sleep_between=args.sleep,
        on_progress=progress,
    )
    print(f"\n✅ Залито новых точек: {n}")
    print(f"📦 В коллекции '{STICKER_COLLECTION}' теперь: {collection_count(client)} точек")


if __name__ == "__main__":
    main()
