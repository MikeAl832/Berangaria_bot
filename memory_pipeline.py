"""Строгий fail-closed пайплайн долговременной памяти."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

import memory_store
import state
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    MEM0_LLM_MODEL,
    MEMORY_MAX_ATTEMPTS,
    MEMORY_MEM0_MIN_CHARS,
    MEMORY_QUERY_MIN_CHARS,
    MEMORY_QUEUE_BATCH_SIZE,
)
from utils import is_low_signal_user_text, is_url_only_text, strip_tiktok_urls

logger = logging.getLogger(__name__)
_worker_task: asyncio.Task[None] | None = None
_processing_lock = asyncio.Lock()


_SCOPE_RE = re.compile(r"^(?:private|group)_-?\d+$")
_FACT_KEY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,79}$")
_SENSITIVE_RE = re.compile(
    r"\b(?:парол\w*|токен\w*|api[ _-]?key|secret\w*|паспорт\w*|"
    r"адрес\w*|банковск\w*|кредитн\w*|cvv|диагноз\w*|медицин\w*|"
    r"лечение\w*|болезн\w*)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"(?:\b(?:кажется|наверно|наверное|возможно|хочу|хотел\w*|планир\w*|"
    r"собира\w*ся|думаю|если|попробую|сейчас|завтра|вчера|сегодня|"
    r"может быть|не уверен\w*)\b|\?|\"|«|»)",
    re.IGNORECASE,
)


class MemoryTransientError(RuntimeError):
    """Временная ошибка, при которой исходник надо повторить позже."""


class MemoryCandidateRejected(ValueError):
    """Кандидат нарушает политику памяти и не должен повторяться."""


@dataclass(frozen=True)
class MemoryCandidate:
    fact: str
    source_quote: str
    fact_key: str


@dataclass(frozen=True)
class VerifiedMemoryFact:
    fact: str
    source_quote: str
    fact_key: str
    reason: str


@dataclass(frozen=True)
class MemoryProcessReport:
    processed: int = 0
    approved: int = 0
    discarded: int = 0
    retried: int = 0
    dead_lettered: int = 0


class MemoryExtractor(Protocol):
    async def extract(self, source: state.MemorySourceRecord) -> list[MemoryCandidate]: ...


class MemoryVerifier(Protocol):
    async def verify(
        self, source: state.MemorySourceRecord, candidate: MemoryCandidate
    ) -> VerifiedMemoryFact | None: ...


class ApprovedFactStore(Protocol):
    async def save(self, source: state.MemorySourceRecord, fact: VerifiedMemoryFact) -> str: ...

    async def replace(
        self,
        memory_id: str,
        source: state.MemorySourceRecord,
        fact: VerifiedMemoryFact,
    ) -> str: ...


def _normalise_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _parse_json_response(data: object) -> dict:
    if not isinstance(data, dict):
        raise MemoryTransientError("ответ DeepSeek не является объектом")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MemoryTransientError("в ответе DeepSeek нет message.content") from exc
    if not isinstance(content, str):
        raise MemoryTransientError("message.content DeepSeek не является строкой")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MemoryTransientError("DeepSeek вернул невалидный JSON") from exc
    if not isinstance(parsed, dict):
        raise MemoryTransientError("JSON DeepSeek должен быть объектом")
    return parsed


async def _deepseek_json(*, system: str, user: str, max_tokens: int) -> dict:
    payload = {
        "model": MEM0_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(DEEPSEEK_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise MemoryTransientError(f"DeepSeek недоступен: {exc}") from exc
    return _parse_json_response(data)


class DeepSeekMemoryExtractor:
    _PROMPT = """Ты извлекаешь кандидатов для строгой долговременной памяти Berangaria.
Источник — одно обычное текстовое сообщение пользователя. Верни JSON-объект ровно
формата {\"candidates\":[{\"fact\":\"...\",\"source_quote\":\"...\",\"fact_key\":\"...\"}]}.

Правила:
- Сохраняй только ясные устойчивые самоутверждения автора о себе.
- Не сохраняй планы, временные состояния, вопросы, желания, догадки, цитаты, иронию,
  сведения о других людях, групповые «мы», секреты, адреса, документы, финансовые и
  медицинские данные, а также медиа или мета про бота.
- `source_quote` должен быть точным фрагментом исходного сообщения.
- `fact_key` — стабильный ASCII-ключ свойства, например `hardware.gpu` или
  `preferences.music`; одинаковое свойство должно получать одинаковый ключ.
- Если подходящих фактов нет, верни пустой список. Ничего не додумывай.
"""

    async def extract(self, source: state.MemorySourceRecord) -> list[MemoryCandidate]:
        payload = json.dumps(
            {
                "author_name": source.author_name,
                "scope": source.scope,
                "source_message_id": source.message_id,
                "text": source.text,
            },
            ensure_ascii=False,
        )
        data = await _deepseek_json(system=self._PROMPT, user=payload, max_tokens=600)
        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, list):
            raise MemoryTransientError("extractor не вернул candidates[]")
        candidates: list[MemoryCandidate] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                raise MemoryTransientError("extractor вернул некорректный кандидат")
            fact = item.get("fact")
            quote = item.get("source_quote")
            fact_key = item.get("fact_key")
            if not all(isinstance(value, str) for value in (fact, quote, fact_key)):
                raise MemoryTransientError("кандидат extractor'а имеет неверные поля")
            candidates.append(MemoryCandidate(fact.strip(), quote.strip(), fact_key.strip()))
        return candidates


class DeepSeekMemoryVerifier:
    _PROMPT = """Ты — независимый verifier строгой долговременной памяти.
Проверь кандидата только по одному исходному сообщению и верни JSON:
{\"decision\":\"KEEP\",\"fact\":\"...\",\"source_quote\":\"...\",\"fact_key\":\"...\",\"reason\":\"...\"}
или {\"decision\":\"DISCARD\",\"reason\":\"...\"}.

KEEP разрешён только когда одновременно верно всё:
- автор сообщения прямо утверждает факт о себе;
- факт устойчивый и полезный, а не временный статус или план;
- точная цитата прямо и полностью подтверждает формулировку;
- нет догадок, модальных слов, иронии, вопросов, цитат, медиа и чувствительных данных;
- факт относится к этому автору, а не к другому человеку или группе.
При малейшем сомнении верни DISCARD. Не используй знания вне сообщения.
"""

    async def verify(
        self, source: state.MemorySourceRecord, candidate: MemoryCandidate
    ) -> VerifiedMemoryFact | None:
        payload = json.dumps(
            {
                "source": {
                    "author_name": source.author_name,
                    "author_id": source.author_id,
                    "message_id": source.message_id,
                    "text": source.text,
                },
                "candidate": {
                    "fact": candidate.fact,
                    "source_quote": candidate.source_quote,
                    "fact_key": candidate.fact_key,
                },
            },
            ensure_ascii=False,
        )
        data = await _deepseek_json(system=self._PROMPT, user=payload, max_tokens=300)
        decision = str(data.get("decision") or "").upper()
        if decision == "DISCARD":
            return None
        if decision != "KEEP":
            raise MemoryTransientError("verifier не вернул KEEP/DISCARD")
        fields = (data.get("fact"), data.get("source_quote"), data.get("fact_key"))
        if not all(isinstance(value, str) for value in fields):
            raise MemoryTransientError("KEEP verifier не содержит обязательных полей")
        return VerifiedMemoryFact(
            fact=fields[0].strip(),
            source_quote=fields[1].strip(),
            fact_key=fields[2].strip(),
            reason=str(data.get("reason") or "без причины").strip()[:200],
        )


class Mem0ApprovedFactStore:
    """Передаёт в Mem0 только уже одобренную формулировку."""

    @staticmethod
    def _metadata(
        source: state.MemorySourceRecord, fact: VerifiedMemoryFact
    ) -> dict[str, str]:
        return {
            "source_id": str(source.id),
            "source_message_id": str(source.message_id),
            "source_created_at": str(source.created_at),
            "source_quote": fact.source_quote,
            "subject_id": source.author_id,
            "fact_key": fact.fact_key,
        }

    async def save(self, source: state.MemorySourceRecord, fact: VerifiedMemoryFact) -> str:
        if memory_store.memory is None:
            raise MemoryTransientError("Mem0 недоступен")
        try:
            result = await asyncio.to_thread(
                memory_store.memory.add,
                [{"role": "user", "content": fact.fact}],
                user_id=source.scope,
                metadata=self._metadata(source, fact),
                infer=False,
            )
        except Exception as exc:
            raise MemoryTransientError(f"Mem0 недоступен: {exc}") from exc
        results = result.get("results") if isinstance(result, dict) else None
        if not isinstance(results, list) or len(results) != 1:
            await self._delete_results(results)
            raise MemoryTransientError("Mem0 вернул не ровно один факт")
        item = results[0] if isinstance(results[0], dict) else {}
        memory_id = item.get("id")
        stored_fact = _normalise_text(item.get("memory") or "")
        event = str(item.get("event") or "").upper()
        if (
            not memory_id
            or event not in {"ADD", "UPDATE"}
            or stored_fact != _normalise_text(fact.fact)
        ):
            await self._delete_results(results)
            raise MemoryTransientError("Mem0 изменил или не подтвердил одобренный факт")
        return str(memory_id)

    async def replace(
        self,
        memory_id: str,
        source: state.MemorySourceRecord,
        fact: VerifiedMemoryFact,
    ) -> str:
        """Заменяет прежнюю версию в том же vector ID без окна дубликатов."""
        if memory_store.memory is None:
            raise MemoryTransientError("Mem0 недоступен")
        try:
            await asyncio.to_thread(
                memory_store.memory.update,
                memory_id,
                fact.fact,
                metadata=self._metadata(source, fact),
            )
            stored = await asyncio.to_thread(memory_store.memory.get, memory_id)
        except Exception as exc:
            raise MemoryTransientError(f"Mem0 update недоступен: {exc}") from exc
        if not isinstance(stored, dict) or _normalise_text(
            stored.get("memory") or ""
        ) != _normalise_text(fact.fact):
            raise MemoryTransientError("Mem0 не подтвердил замену одобренного факта")
        return memory_id

    async def _delete_results(self, results: object) -> None:
        if not isinstance(results, list):
            return
        for item in results:
            if isinstance(item, dict) and item.get("id"):
                try:
                    await asyncio.to_thread(memory_store.memory.delete, item["id"])
                except Exception as exc:
                    logger.error("Память: не удалось удалить неподтверждённый результат Mem0: %s", exc)

def _validate_verified_fact(
    source: state.MemorySourceRecord, fact: VerifiedMemoryFact
) -> None:
    normalized_source = _normalise_text(source.text)
    normalized_quote = _normalise_text(fact.source_quote)
    normalized_fact = _normalise_text(fact.fact)
    if not normalized_fact or len(normalized_fact) > 500:
        raise MemoryCandidateRejected("пустой или слишком длинный факт")
    if not normalized_quote or normalized_quote not in normalized_source:
        raise MemoryCandidateRejected("доказательная цитата не найдена в источнике")
    if not _FACT_KEY_RE.fullmatch(fact.fact_key):
        raise MemoryCandidateRejected("некорректный fact_key")
    evidence_text = f"{normalized_fact} {normalized_quote}"
    if _SENSITIVE_RE.search(evidence_text):
        raise MemoryCandidateRejected("чувствительная категория")
    if _UNCERTAIN_RE.search(evidence_text):
        raise MemoryCandidateRejected("неясная или временная формулировка")
    if "[image description:" in normalized_source.lower() or "[video description:" in normalized_source.lower():
        raise MemoryCandidateRejected("медиа не является источником памяти")


async def _process_pending_memory(
    extractor: MemoryExtractor | None = None,
    verifier: MemoryVerifier | None = None,
    store: ApprovedFactStore | None = None,
    *,
    limit: int = MEMORY_QUEUE_BATCH_SIZE,
    max_attempts: int = MEMORY_MAX_ATTEMPTS,
) -> MemoryProcessReport:
    """Обрабатывает FIFO-очередь, сохраняя только прошедшие проверки факты."""
    if store is None and memory_store.memory is None:
        return MemoryProcessReport()
    extractor = extractor or DeepSeekMemoryExtractor()
    verifier = verifier or DeepSeekMemoryVerifier()
    store = store or Mem0ApprovedFactStore()

    processed = approved = discarded = retried = dead_lettered = 0
    bounded_limit = max(1, min(int(limit), 100))
    for _ in range(bounded_limit):
        claimed = state.claim_memory_sources(1)
        if not claimed:
            break
        source = claimed[0]
        processed += 1
        try:
            candidates = await extractor.extract(source)
            if not isinstance(candidates, list):
                raise MemoryTransientError("extractor вернул не список")
            seen_keys: set[str] = set()
            for candidate in candidates:
                if not isinstance(candidate, MemoryCandidate):
                    raise MemoryTransientError("extractor вернул не MemoryCandidate")
                if candidate.fact_key in seen_keys:
                    discarded += 1
                    continue
                seen_keys.add(candidate.fact_key)
                try:
                    verified = await verifier.verify(source, candidate)
                    if verified is None:
                        discarded += 1
                        continue
                    _validate_verified_fact(source, verified)
                    previous = state.get_memory_fact(
                        source.scope, source.author_id, verified.fact_key
                    )
                    if previous:
                        mem0_id = await store.replace(previous.mem0_id, source, verified)
                    else:
                        mem0_id = await store.save(source, verified)
                    state.upsert_memory_fact(
                        scope=source.scope,
                        subject_id=source.author_id,
                        fact_key=verified.fact_key,
                        fact=_normalise_text(verified.fact),
                        source_id=source.id,
                        source_quote=_normalise_text(verified.source_quote),
                        source_message_id=source.message_id,
                        source_created_at=source.created_at,
                        mem0_id=mem0_id,
                    )
                    approved += 1
                    logger.info(
                        "Память: факт одобрен (source_id=%s, scope=%s, key=%s)",
                        source.id,
                        source.scope,
                        verified.fact_key,
                    )
                except MemoryCandidateRejected as exc:
                    discarded += 1
                    logger.info(
                        "Память: кандидат отклонён (source_id=%s, reason=%s)",
                        source.id,
                        str(exc),
                    )
            state.complete_memory_source(source.id)
        except Exception as exc:
            if isinstance(exc, MemoryCandidateRejected):
                state.complete_memory_source(source.id)
                discarded += 1
                continue
            terminal = state.fail_memory_source(source.id, str(exc), max_attempts=max_attempts)
            if terminal:
                dead_lettered += 1
                logger.error("Память: источник отправлен в dead-letter (source_id=%s)", source.id)
            else:
                retried += 1
                logger.warning("Память: источник возвращён в очередь (source_id=%s)", source.id)
                # Более новое сообщение не должно обогнать старое и затем быть
                # перезаписано его запоздавшим retry.
                break

    return MemoryProcessReport(
        processed=processed,
        approved=approved,
        discarded=discarded,
        retried=retried,
        dead_lettered=dead_lettered,
    )


async def process_pending_memory(
    extractor: MemoryExtractor | None = None,
    verifier: MemoryVerifier | None = None,
    store: ApprovedFactStore | None = None,
    *,
    limit: int = MEMORY_QUEUE_BATCH_SIZE,
    max_attempts: int = MEMORY_MAX_ATTEMPTS,
) -> MemoryProcessReport:
    """Сериализует обработку очереди, сохраняя FIFO-порядок конфликтов."""
    async with _processing_lock:
        return await _process_pending_memory(
            extractor,
            verifier,
            store,
            limit=limit,
            max_attempts=max_attempts,
        )


def enqueue_memory_source(
    *,
    scope: str,
    author_id: str,
    author_name: str,
    message_id: int,
    text: str,
    created_at: float,
) -> int | None:
    """Ставит одно содержательное текстовое сообщение в SQLite-очередь.

    Отсутствие provenance, URL-only и low-signal текст отбрасываются до любой
    внешней модели. Медиа сюда нельзя передать отдельным параметром намеренно.
    """
    clean_text = strip_tiktok_urls((text or "").strip())
    if not _SCOPE_RE.fullmatch(scope or ""):
        return None
    if not str(author_id).strip() or not str(author_name).strip():
        return None
    if not isinstance(message_id, int):
        return None
    if len(clean_text) < MEMORY_MEM0_MIN_CHARS:
        return None
    if is_url_only_text(clean_text) or is_low_signal_user_text(
        clean_text, min_alnum=MEMORY_QUERY_MIN_CHARS
    ):
        return None

    source_id = state.insert_memory_source(
        scope=scope,
        author_id=str(author_id),
        author_name=author_name.strip(),
        message_id=message_id,
        created_at=float(created_at),
        text=clean_text,
    )
    schedule_memory_processing()
    return source_id


def schedule_memory_processing() -> None:
    """Запускает один фоновый worker, если текущий код работает в event loop."""
    global _worker_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _worker_task is not None and not _worker_task.done():
        return
    _worker_task = loop.create_task(_run_memory_worker())


async def _run_memory_worker() -> None:
    limit = MEMORY_QUEUE_BATCH_SIZE
    while True:
        report = await process_pending_memory(limit=limit)
        if report.processed < limit or report.retried or report.dead_lettered:
            return


async def wait_for_memory_worker() -> None:
    """Дожидается текущего worker перед graceful shutdown."""
    global _worker_task
    task = _worker_task
    if task is not None and not task.done():
        await asyncio.gather(task, return_exceptions=True)
    _worker_task = None
