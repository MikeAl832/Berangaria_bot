"""
Векторное хранилище стикеров в Qdrant.

Общий модуль для:
  - build_sticker_index.py (разовая индексация jsonl -> Qdrant)
  - llm_client.py (рантайм-поиск стикера по описанию через tool send_sticker)

Эмбеддинги считаем той же моделью, что и mem0 (Gemini `EMBEDDING_MODEL`),
но напрямую по REST (как vision_provider.py) — ради контроля над батчингом,
outputDimensionality и taskType. Индекс и запрос идут в ОДНОМ пространстве:
  - индексация: taskType=RETRIEVAL_DOCUMENT
  - поиск:      taskType=RETRIEVAL_QUERY
Вектора L2-нормализуем и храним с косинусной метрикой.
"""

import logging
import math
import os
import time
import uuid

import httpx

from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL,
    QDRANT_HOST, QDRANT_PORT,
    STICKER_COLLECTION, STICKER_DIMS,
    STICKER_MIN_SCORE, STICKER_TOP_K,
)

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Стабильный namespace, чтобы point_id детерминированно выводился из file_id
# (переиндексация идемпотентна — дублей не появляется).
_STICKER_NS = uuid.UUID("5f3d8c1a-1c2b-4e7a-9b6d-0a1b2c3d4e5f")


# ============================ ЭМБЕДДИНГИ ============================

def _l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class _RateLimited(Exception):
    """Внутренний сигнал: упёрлись в лимит запросов (429) — уже переждали, но исчерпали ретраи."""


def _parse_retry_delay(resp) -> float:
    """Достаёт рекомендованную паузу из тела ошибки Gemini (RetryInfo.retryDelay='36s')."""
    try:
        for d in resp.json().get("error", {}).get("details", []):
            rd = d.get("retryDelay")
            if rd and rd.endswith("s"):
                return float(rd[:-1])
    except Exception:
        pass
    return 0.0


def _post_embed(url, headers, body, timeout=60.0, max_retries=8):
    """
    POST к Gemini с ретраями на 429 (rate limit) и 503 (overloaded).
    На лимите ЖДЁТ (по retryDelay из ответа, иначе экспоненциально до 65с) и повторяет,
    а не падает — чтобы один прогон дозаливал всё без ручных рестартов.
    Ошибки вроде 400 (напр. батч не поддерживается) пробрасывает сразу — для фолбэка.
    """
    delay = 10.0
    last = None
    for attempt in range(max_retries):
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers, json=body)
        if r.status_code == 200:
            return r.json()
        last = r
        if r.status_code in (429, 503):
            wait = _parse_retry_delay(r) or delay
            wait = min(wait, 65.0)
            logger.warning(
                f"⏳ [yellow]Gemini {r.status_code} (лимит), жду {wait:.0f}s и повторяю "
                f"(попытка {attempt + 1}/{max_retries})[/]"
            )
            time.sleep(wait)
            delay = min(delay * 2, 65.0)
            continue
        # прочие коды — не ретраим, пусть решает вызывающий (фолбэк на поштучный и т.п.)
        r.raise_for_status()
    # Исчерпали ретраи по лимиту
    raise _RateLimited(f"rate limit не отпустил после {max_retries} попыток (last={last.status_code if last else '?'})")


def _embed_batch_request(texts, task_type, timeout=60.0):
    """Один вызов :batchEmbedContents (с ретраями на лимит). Возвращает list[list[float]] или бросает."""
    url = f"{GEMINI_API_BASE}/models/{EMBEDDING_MODEL}:batchEmbedContents"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    body = {
        "requests": [
            {
                "model": f"models/{EMBEDDING_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": STICKER_DIMS,
            }
            for t in texts
        ]
    }
    data = _post_embed(url, headers, body, timeout=timeout)
    return [_l2_normalize(e["values"]) for e in data["embeddings"]]


def _embed_single_request(text, task_type, timeout=60.0):
    """Фолбэк: один текст через :embedContent (с ретраями на лимит)."""
    url = f"{GEMINI_API_BASE}/models/{EMBEDDING_MODEL}:embedContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    body = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": STICKER_DIMS,
    }
    data = _post_embed(url, headers, body, timeout=timeout)
    return _l2_normalize(data["embedding"]["values"])


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=50, sleep_between=0.0):
    """
    Считает эмбеддинги для списка текстов. Пытается батчить (1 HTTP-запрос на
    batch_size текстов — экономит дневной лимит запросов). Если батч-эндпоинт
    отказывает, откатывается на поштучные запросы.

    sleep_between — пауза (сек) между HTTP-запросами (троттлинг под RPM/TPM).
    Возвращает list[list[float]] той же длины и порядка, что texts.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY не задан — эмбеддинги стикеров недоступны")

    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        try:
            vecs = _embed_batch_request(chunk, task_type)
        except _RateLimited:
            raise  # лимит, а не «батч не поддерживается» — поштучно не спасёт, чистим наверх
        except Exception as e:
            logger.warning(f"⚠️ batchEmbedContents не сработал ({e}); откат на поштучный режим")
            vecs = []
            for t in chunk:
                vecs.append(_embed_single_request(t, task_type))
                if sleep_between:
                    time.sleep(sleep_between)
        out.extend(vecs)
        if sleep_between:
            time.sleep(sleep_between)
    return out


# ============================ QDRANT ============================

def get_client():
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# Мусорные значения franchise: у 680/782 стикеров это «meme»/«other» — только шумят вектор.
_NOISE_FRANCHISE = {None, "", "meme", "other", "unknown"}


def sticker_text(rec: dict) -> str:
    """
    Текст для эмбеддинга стикера. Запросы модель формулирует по ЭМОЦИИ/ПОВОДУ
    ("недоумение, кто-то сморозил глупость"), поэтому emotion+keywords ставим ВПЕРЁД,
    а длинное визуальное описание — в хвост, чтобы оно не забивало сигнал.
    Шум (franchise=meme/other, character=null) выкидываем.
    """
    kw = rec.get("keywords") or []
    emotion = rec.get("emotion")
    desc = rec.get("description")
    franchise = rec.get("franchise")
    if franchise in _NOISE_FRANCHISE:
        franchise = None
    character = rec.get("character")  # почти всегда null; берём, только если есть

    parts = [
        emotion,
        ", ".join(kw) if kw else None,
        character,
        franchise,
        desc,
    ]
    return " | ".join(str(p) for p in parts if p)


def point_id(file_id: str) -> str:
    return str(uuid.uuid5(_STICKER_NS, file_id))


def ensure_collection(client=None, recreate=False):
    """Создаёт коллекцию стикеров, если её нет. recreate=True пересоздаёт с нуля."""
    from qdrant_client.models import Distance, VectorParams
    client = client or get_client()
    exists = client.collection_exists(STICKER_COLLECTION)
    if exists and recreate:
        client.delete_collection(STICKER_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=STICKER_COLLECTION,
            vectors_config=VectorParams(size=STICKER_DIMS, distance=Distance.COSINE),
        )
        logger.info(f"🆕 Коллекция '{STICKER_COLLECTION}' создана (dims={STICKER_DIMS}, cosine)")
    return client


def upsert_stickers(records, client=None, batch_size=50, sleep_between=0.0, on_progress=None):
    """
    Индексирует записи стикеров (list[dict]) в Qdrant.
    Каждая запись должна содержать 'filename' (Telegram file_id) и описание.
    Возвращает число залитых точек.
    """
    from qdrant_client.models import PointStruct
    client = ensure_collection(client)

    total = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        texts = [sticker_text(r) for r in chunk]
        vecs = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT",
                           batch_size=batch_size, sleep_between=sleep_between)
        points = []
        for r, v in zip(chunk, vecs):
            fid = r.get("filename")
            if not fid:
                continue
            points.append(PointStruct(
                id=point_id(fid),
                vector=v,
                payload={
                    "file_id": fid,
                    "description": r.get("description"),
                    "keywords": r.get("keywords") or [],
                    "emotion": r.get("emotion"),
                    "franchise": r.get("franchise"),
                },
            ))
        if points:
            client.upsert(collection_name=STICKER_COLLECTION, points=points)
            total += len(points)
        if on_progress:
            on_progress(min(i + batch_size, len(records)), len(records))
    return total


def collection_count(client=None) -> int:
    client = client or get_client()
    if not client.collection_exists(STICKER_COLLECTION):
        return 0
    return client.count(STICKER_COLLECTION, exact=True).count


def existing_ids(ids, client=None, batch=256) -> set:
    """Возвращает множество point_id (из переданных), которые УЖЕ есть в коллекции."""
    client = client or get_client()
    if not client.collection_exists(STICKER_COLLECTION):
        return set()
    found = set()
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        pts = client.retrieve(
            collection_name=STICKER_COLLECTION,
            ids=chunk, with_payload=False, with_vectors=False,
        )
        for p in pts:
            found.add(str(p.id))
    return found


def missing_records(records, client=None):
    """Из списка записей стикеров оставляет только те, которых ещё нет в коллекции
    (сравнение по детерминированному point_id из file_id). Записи без file_id отбрасываются."""
    client = client or get_client()
    ids = [point_id(r["filename"]) for r in records if r.get("filename")]
    have = existing_ids(ids, client)
    out = []
    for r in records:
        fid = r.get("filename")
        if fid and point_id(fid) not in have:
            out.append(r)
    return out


def load_jsonl(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                import json
                recs.append(json.loads(line))
    return recs


# ---- Маркер версии формата индекса (для авто-миграции при старте) ----
# Лежит рядом с bot_state.db (том bot_data → переживает деплой).

def _version_marker_path():
    db = os.environ.get("BOT_DB_PATH", "bot_state.db")
    return os.path.join(os.path.dirname(os.path.abspath(db)), "sticker_index.version")


def get_applied_version() -> int:
    try:
        with open(_version_marker_path(), encoding="utf-8") as f:
            return int((f.read() or "0").strip() or 0)
    except Exception:
        return 0


def set_applied_version(v: int) -> None:
    try:
        with open(_version_marker_path(), "w", encoding="utf-8") as f:
            f.write(str(int(v)))
    except Exception as e:
        logger.warning(f"⚠️ Не удалось записать маркер версии индекса стикеров: {e}")


def sync_from_file(path, limit=None, batch_size=50, sleep_between=1.5,
                   recreate=False, force_all=False, on_progress=None):
    """
    Синхронизирует коллекцию с jsonl.
      - обычный режим: заливает только недостающие стикеры (по file_id);
      - force_all=True: переэмбеддивает ВСЕ записи и перезаписывает точки (по тем же id),
        БЕЗ стирания коллекции — для миграции формата эмбеддинга без «пустого окна».
    Блокирующая (эмбеддинг + Qdrant) — вызывать из потока (asyncio.to_thread).
    Возвращает dict со статистикой.
    """
    records = load_jsonl(path)
    client = get_client()
    if recreate:
        ensure_collection(client, recreate=True)
    already = collection_count(client)
    if force_all:
        ensure_collection(client)  # создать, если нет; существующие точки перезапишем upsert'ом
        new = [r for r in records if r.get("filename")]
    else:
        new = missing_records(records, client)
    if limit:
        new = new[:limit]
    if not new:
        return {"in_file": len(records), "already": already, "added": 0, "total": already}
    added = upsert_stickers(
        new, client=client,
        batch_size=batch_size, sleep_between=sleep_between, on_progress=on_progress,
    )
    return {"in_file": len(records), "already": already, "added": added,
            "total": collection_count(client)}


# ============================ ПОИСК (рантайм) ============================

def search_stickers(query: str, top_k: int = None, min_score: float = None):
    """
    Ищет подходящие стикеры по текстовому описанию.
    Возвращает list[dict]: {file_id, description, emotion, score}, отсортированный
    по убыванию score, только выше порога min_score. Пустой список — ничего не найдено
    или коллекция недоступна (вызывающий код тогда просто не шлёт стикер).
    """
    top_k = top_k or STICKER_TOP_K
    min_score = STICKER_MIN_SCORE if min_score is None else min_score
    try:
        client = get_client()
        if not client.collection_exists(STICKER_COLLECTION):
            logger.warning(f"⚠️ Коллекция '{STICKER_COLLECTION}' не найдена — стикеры не проиндексированы")
            return []
        vec = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
        res = client.query_points(
            collection_name=STICKER_COLLECTION,
            query=vec,
            limit=top_k,
            with_payload=True,
        )
        out = []
        for p in res.points:
            if p.score is None or p.score < min_score:
                continue
            pl = p.payload or {}
            out.append({
                "file_id": pl.get("file_id"),
                "description": pl.get("description"),
                "emotion": pl.get("emotion"),
                "score": p.score,
            })
        return out
    except Exception as e:
        logger.error(f"❌ Ошибка поиска стикера: {e}")
        return []
