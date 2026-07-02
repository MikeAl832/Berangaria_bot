#!/usr/bin/env python
"""
Синхронизация стикеров с Qdrant.

Читает jsonl (по умолчанию stickers_clean.jsonl), сверяет с коллекцией и заливает
ТОЛЬКО те стикеры, которых там ещё нет (сравнение по детерминированному point_id из
file_id). Рабочий процесс: ты дописываешь новые строки в конец stickers_clean.jsonl,
запускаешь эту команду — добавляются только новые. Повторный запуск без новых строк
не делает ничего. Уже удалённые из файла строки из коллекции НЕ удаляются (только добавление).

Эмбеддинги считаются той же Gemini-моделью, что и mem0. Дневной лимит запросов бережём
батчингом (1 HTTP-запрос на --batch-size стикеров); если новых много и не хочешь тратить
квоту разом — ограничь порцию через --limit, остальное дольёшь следующим запуском.

Примеры (на сервере, внутри контейнера бота):
    docker compose exec bot python build_sticker_index.py                # залить все недостающие
    docker compose exec bot python build_sticker_index.py --limit 300    # только 300 новых за раз
    docker compose exec bot python build_sticker_index.py --dry-run      # показать, сколько нового

С хоста (Qdrant проброшен на 127.0.0.1:6333):
    QDRANT_HOST=localhost python build_sticker_index.py --dry-run
"""

import argparse
import json
import os
import sys


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser(description="Синхронизация стикеров с Qdrant (добавляет только новые)")
    ap.add_argument("--input", default="stickers_clean.jsonl", help="jsonl со стикерами")
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
    from sticker_store import (
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
    print(f"   Лимиты Gemini: 100 RPM / 30k TPM / 1000 RPD")
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
