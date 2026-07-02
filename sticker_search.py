#!/usr/bin/env python
"""
Диагностика поиска стикеров: вбиваешь запрос — видишь, что находит Qdrant.

Показывает топ-N кандидатов со СКОРАМИ (без отсечки по порогу), чтобы понять,
что тюнить: порог (sticker_min_score), ранжирование или сами данные.

Примеры (внутри контейнера бота):
    docker compose exec bot python sticker_search.py "недоумение, кто-то сморозил глупость"
    docker compose exec bot python sticker_search.py --top 15 "ржу в голос"

С хоста:
    QDRANT_HOST=localhost python sticker_search.py "одобряю, огонь"
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Диагностика векторного поиска стикеров")
    ap.add_argument("query", nargs="+", help="текст запроса")
    ap.add_argument("--top", type=int, default=10, help="сколько кандидатов показать")
    ap.add_argument("--qdrant-host", default=None, help="переопределить хост Qdrant (напр. localhost)")
    args = ap.parse_args()

    if args.qdrant_host:
        os.environ["QDRANT_HOST"] = args.qdrant_host

    from sticker_store import search_stickers, collection_count, STICKER_MIN_SCORE

    query = " ".join(args.query)
    print(f"🔎 Запрос: {query!r}")
    print(f"📦 В коллекции: {collection_count()} стикеров | текущий порог отсечки: {STICKER_MIN_SCORE}\n")

    # min_score=0 — показываем ВСЁ, включая то, что отсеклось бы порогом
    results = search_stickers(query, top_k=args.top, min_score=0.0)
    if not results:
        print("Ничего не найдено (коллекция пуста или ошибка поиска).")
        sys.exit(1)

    for i, r in enumerate(results, 1):
        mark = "✅" if r["score"] >= STICKER_MIN_SCORE else "  "
        emo = r.get("emotion") or "—"
        desc = (r.get("description") or "").replace("\n", " ")[:90]
        print(f"{mark} {i:2}. score={r['score']:.3f}  [{emo}]  {desc}")

    print(f"\n(✅ = прошёл бы текущий порог {STICKER_MIN_SCORE})")


if __name__ == "__main__":
    main()
