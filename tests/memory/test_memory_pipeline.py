from berangaria.core import state
import asyncio
import logging
import json

from berangaria.memory import pipeline as memory_pipeline
import pytest

from berangaria.memory.pipeline import (
    DeepSeekMemoryExtractor,
    DeepSeekMemoryVerifier,
    Mem0ApprovedFactStore,
    MemoryCandidate,
    MemoryCandidateRejected,
    MemoryDiscard,
    MemoryTransientError,
    VerifiedMemoryFact,
    process_pending_memory,
    _validate_verified_fact,
)
from berangaria.memory import store as memory_store


def test_text_message_is_queued_with_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    source_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=901,
        text="Я использую видеокарту RTX 5070 Ti",
        created_at=1_725_000_000.0,
    )

    rows = state.list_memory_sources(status="pending")
    assert source_id == rows[0].id
    assert rows[0].scope == "private_42"
    assert rows[0].author_id == "42"
    assert rows[0].message_id == 901
    assert rows[0].text == "Я использую видеокарту RTX 5070 Ti"
    assert rows[0].attempts == 0


def test_duplicate_message_edit_does_not_replace_original_source(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    original_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=901,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_000.0,
    )
    edited_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=901,
        text="Я постоянно использую Arch Linux",
        created_at=1_725_000_001.0,
    )

    rows = state.list_memory_sources(status="pending")
    assert edited_id == original_id
    assert len(rows) == 1
    assert rows[0].text == "Я постоянно использую Fedora Linux"
    assert rows[0].created_at == 1_725_000_000.0


def test_short_stable_self_statement_is_queued(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    source_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=908,
        text="Люблю джаз",
        created_at=1_725_000_007.0,
    )

    assert source_id is not None
    assert state.list_memory_sources(status="pending")[0].text == "Люблю джаз"


def test_source_waits_for_turn_completion_before_it_can_be_claimed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source, release_memory_sources

    source_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=910,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_009.0,
        ready=False,
    )

    assert state.claim_memory_sources(1) == []

    release_memory_sources([source_id])

    assert state.claim_memory_sources(1)[0].id == source_id


def test_waiting_source_blocks_newer_ready_source(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source, release_memory_sources

    older_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=911,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_010.0,
        ready=False,
    )
    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=912,
        text="Я постоянно живу в Москве",
        created_at=1_725_000_011.0,
    )

    assert state.claim_memory_sources(1) == []

    release_memory_sources([older_id])

    assert state.claim_memory_sources(1)[0].id == older_id


def test_waiting_source_does_not_block_other_scopes(monkeypatch, tmp_path):
    """Гейт FIFO действует внутри области памяти, а не глобально.

    Конфликт перезаписи возможен только внутри области (memory_facts уникален по
    scope+subject_id+fact_key). Глобальный гейт означал бы, что один
    недоставленный ход в любом чате останавливает память во всех остальных.
    """
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source, release_memory_sources

    # Застрявший источник в личке — попал раньше всех, released не будет.
    stuck_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=920,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_020.0,
        ready=False,
    )
    # Здоровые источники в другой области, все новее застрявшего.
    group_ids = [
        enqueue_memory_source(
            scope="group_-100",
            author_id="42",
            author_name="Миша",
            message_id=930 + offset,
            text=f"Я работаю программистом уже {offset + 3} года",
            created_at=1_725_000_030.0 + offset,
            ready=False,
        )
        for offset in range(3)
    ]
    release_memory_sources(group_ids)

    claimed = state.claim_memory_sources(10)

    assert [source.id for source in claimed] == group_ids
    assert all(source.scope == "group_-100" for source in claimed)
    # Застрявший так и остался ждать — его область по-прежнему заблокирована.
    assert state.list_memory_sources("waiting")[0].id == stuck_id


def test_abandoning_stuck_source_unblocks_its_own_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import (
        abandon_memory_sources,
        enqueue_memory_source,
        release_memory_sources,
    )

    stuck_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=940,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_040.0,
        ready=False,
    )
    newer_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=941,
        text="Я больше десяти лет живу в Москве",
        created_at=1_725_000_041.0,
        ready=False,
    )
    release_memory_sources([newer_id])

    # Пока застрявший висит — своя область заблокирована.
    assert state.claim_memory_sources(5) == []

    abandon_memory_sources([stuck_id])

    assert [source.id for source in state.claim_memory_sources(5)] == [newer_id]
    abandoned = {source.id: source for source in state.list_memory_sources("abandoned")}
    # Сырой текст недоставленного хода стирается, как при восстановлении на старте.
    assert abandoned[stuck_id].text == ""


def test_abandon_does_not_touch_already_processed_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source, release_memory_sources

    source_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=950,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_050.0,
        ready=False,
    )
    release_memory_sources([source_id])
    state.claim_memory_sources(1)
    state.complete_memory_source(source_id)

    assert state.abandon_memory_sources([source_id]) == 0
    assert state.list_memory_sources("completed")[0].id == source_id


def test_reaper_buries_only_stale_waiting_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    stale_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=960,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_060.0,
        ready=False,
    )
    fresh_id = enqueue_memory_source(
        scope="group_-100",
        author_id="42",
        author_name="Миша",
        message_id=961,
        text="Я больше десяти лет живу в Москве",
        created_at=1_725_000_061.0,
        ready=False,
    )
    # Состариваем только первый — ход, начавшийся час назад, живым быть не может.
    state._db_execute(
        "UPDATE memory_sources SET updated_at=updated_at-3600 WHERE id=?",
        (stale_id,),
    )

    assert state.reap_stale_waiting_sources(1800) == 1

    waiting = [source.id for source in state.list_memory_sources("waiting")]
    abandoned = {source.id: source for source in state.list_memory_sources("abandoned")}
    assert waiting == [fresh_id]
    assert abandoned[stale_id].text == ""
    # Повторный проход ничего не находит — жнец идемпотентен.
    assert state.reap_stale_waiting_sources(1800) == 0


def test_reaper_disabled_by_non_positive_window(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=970,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_070.0,
        ready=False,
    )

    assert state.reap_stale_waiting_sources(0) == 0
    assert len(state.list_memory_sources("waiting")) == 1


def test_interrupted_waiting_source_is_abandoned_on_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    source_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=913,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_012.0,
        ready=False,
    )

    state.init_db()

    source = state.list_memory_sources()[0]
    assert source.id == source_id
    assert source.status == "abandoned"
    assert source.text == ""
    assert state.claim_memory_sources(1) == []


def test_tiktok_source_is_preserved_but_not_sent_to_extractor_as_a_url(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    original = "смотри https://www.tiktok.com/@x/video/1 смешно"
    source_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=914,
        text=original,
        created_at=1_725_000_013.0,
    )

    source = state.list_memory_sources(status="pending")[0]
    assert source.id == source_id
    assert source.text == original

    captured = {}

    async def fake_deepseek_json(*, system, user, max_tokens):
        captured.update(json.loads(user))
        return {"candidates": []}

    monkeypatch.setattr(memory_pipeline, "_deepseek_json", fake_deepseek_json)
    asyncio.run(DeepSeekMemoryExtractor().extract(source))

    assert "tiktok.com" not in captured["text"]


def test_extractor_requests_deepseek_json_output(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"candidates":[]}'},
                    }
                ]
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json, headers):
            captured.update(json)
            return Response()

    monkeypatch.setattr(memory_pipeline.httpx, "AsyncClient", Client)
    source = state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=915,
        created_at=1_725_000_014.0,
        text="Я постоянно использую Helix",
        status="processing",
        attempts=1,
        last_error=None,
    )

    assert asyncio.run(DeepSeekMemoryExtractor().extract(source)) == []
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["thinking"] == {"type": "disabled"}


def test_extractor_reports_truncated_deepseek_json(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": '{"candidates":['},
                    }
                ]
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json, headers):
            return Response()

    monkeypatch.setattr(memory_pipeline.httpx, "AsyncClient", Client)
    source = state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=915,
        created_at=1_725_000_014.0,
        text="Я постоянно использую Helix",
        status="processing",
        attempts=1,
        last_error=None,
    )

    with pytest.raises(MemoryTransientError, match="обрезал JSON"):
        asyncio.run(DeepSeekMemoryExtractor().extract(source))


def test_extractor_reports_empty_deepseek_json(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}}
                ]
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json, headers):
            return Response()

    monkeypatch.setattr(memory_pipeline.httpx, "AsyncClient", Client)
    source = state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=915,
        created_at=1_725_000_014.0,
        text="Я постоянно использую Helix",
        status="processing",
        attempts=1,
        last_error=None,
    )

    with pytest.raises(MemoryTransientError, match="пустой JSON"):
        asyncio.run(DeepSeekMemoryExtractor().extract(source))


def test_approved_candidate_is_stored_with_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    source_id = enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=901,
        text="Я использую видеокарту RTX 5070 Ti",
        created_at=1_725_000_000.0,
    )

    class Extractor:
        async def extract(self, source):
            return [MemoryCandidate("Миша использует RTX 5070 Ti", "видеокарту RTX 5070 Ti", "hardware.gpu")]

    class Verifier:
        async def verify(self, source, candidate):
            return VerifiedMemoryFact(
                fact=candidate.fact,
                source_quote=candidate.source_quote,
                fact_key=candidate.fact_key,
                reason="прямое утверждение",
            )

    class Store:
        def __init__(self):
            self.saved = []

        async def save(self, source, fact):
            self.saved.append((source, fact))
            return "mem0-1"

        async def delete(self, memory_id):
            raise AssertionError(f"неожиданное удаление {memory_id}")

    store = Store()
    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), store))

    assert report.approved == 1
    assert report.retried == 0
    assert len(store.saved) == 1
    completed_source = state.list_memory_sources(status="completed")[0]
    assert completed_source.id == source_id
    assert completed_source.text == ""
    fact = state.list_memory_facts("private_42")[0]
    assert fact.fact == "Миша использует RTX 5070 Ti"
    assert fact.source_id == source_id


def test_discarded_candidate_never_reaches_memory_store(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="group_7",
        author_id="42",
        author_name="Миша",
        message_id=902,
        text="Я, кажется, завтра начну новый проект",
        created_at=1_725_000_001.0,
    )

    class Extractor:
        async def extract(self, source):
            return [MemoryCandidate("Миша начнёт новый проект", "начну новый проект", "projects.current")]

    class Verifier:
        async def verify(self, source, candidate):
            return MemoryDiscard("это краткосрочный план")

    class Store:
        async def save(self, source, fact):
            raise AssertionError("DISCARD не должен вызывать storage")

        async def delete(self, memory_id):
            raise AssertionError("DISCARD не должен вызывать delete")

    caplog.set_level(logging.INFO, logger="memory_pipeline")
    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), Store()))

    assert report.approved == 0
    assert report.discarded == 1
    assert state.list_memory_sources(status="completed")
    assert state.list_memory_facts("group_7") == []
    assert "это краткосрочный план" in caplog.text


def test_keep_without_exact_source_quote_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=904,
        text="Я использую Fedora",
        created_at=1_725_000_003.0,
    )

    class Extractor:
        async def extract(self, source):
            return [MemoryCandidate("Миша использует Arch Linux", "Arch Linux", "software.os")]

    class Verifier:
        async def verify(self, source, candidate):
            return VerifiedMemoryFact(
                fact=candidate.fact,
                source_quote=candidate.source_quote,
                fact_key=candidate.fact_key,
                reason="ошибочный KEEP",
            )

    class Store:
        async def save(self, source, fact):
            raise AssertionError("неподтверждённый факт не должен попасть в Mem0")

        async def delete(self, memory_id):
            raise AssertionError("не было записи для удаления")

    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), Store()))

    assert report.discarded == 1
    assert state.list_memory_facts("private_42") == []


def test_source_quote_must_be_a_verbatim_substring(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=909,
        text="Я постоянно использую  Fedora Linux",
        created_at=1_725_000_008.0,
    )

    class Extractor:
        async def extract(self, source):
            return [
                MemoryCandidate(
                    "Миша постоянно использует Fedora Linux",
                    "постоянно использую Fedora Linux",
                    "software.os",
                )
            ]

    class Verifier:
        async def verify(self, source, candidate):
            return VerifiedMemoryFact(
                candidate.fact,
                candidate.source_quote,
                candidate.fact_key,
                "ошибочный KEEP",
            )

    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), object()))

    assert report.discarded == 1
    assert state.list_memory_facts("private_42") == []


def test_sensitive_source_quote_is_rejected_even_if_fact_hides_category(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=905,
        text="Я живу: Невский проспект, 12",
        created_at=1_725_000_004.0,
    )

    class Extractor:
        async def extract(self, source):
            return [
                MemoryCandidate(
                    "Миша живёт: Невский проспект, 12",
                    "Невский проспект, 12",
                    "profile.location",
                )
            ]

    class Verifier:
        async def verify(self, source, candidate):
            return VerifiedMemoryFact(
                fact=candidate.fact,
                source_quote=candidate.source_quote,
                fact_key=candidate.fact_key,
                reason="ошибочный KEEP",
            )

    class Store:
        async def save(self, source, fact):
            raise AssertionError("чувствительный факт не должен попасть в Mem0")

        async def delete(self, memory_id):
            raise AssertionError("не было записи для удаления")

    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), Store()))

    assert report.discarded == 1
    assert state.list_memory_facts("private_42") == []


@pytest.mark.parametrize(
    ("text", "fact", "quote"),
    [
        ("У меня диабет", "У Миши диабет", "У меня диабет"),
        (
            "У меня диагноз: хроническая мигрень",
            "У Миши хроническая мигрень",
            "хроническая мигрень",
        ),
        (
            "Номер моей карты 1234 5678 9012 3456",
            "Карта Миши: 1234 5678 9012 3456",
            "Номер моей карты 1234 5678 9012 3456",
        ),
        ("Надеюсь переехать в Казань", "Миша переедет в Казань", "переехать в Казань"),
        ("Мечтаю купить дом", "Миша купит дом", "Мечтаю купить дом"),
    ],
)
def test_deterministic_policy_rejects_sensitive_and_modal_language(
    text, fact, quote
):
    source = state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=917,
        created_at=1_725_000_016.0,
        text=text,
        status="processing",
        attempts=1,
        last_error=None,
    )
    verified = VerifiedMemoryFact(fact, quote, "profile.fact", "ошибочный KEEP")

    with pytest.raises(MemoryCandidateRejected):
        _validate_verified_fact(source, verified)


def _policy_source(text: str) -> state.MemorySourceRecord:
    return state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=918,
        created_at=1_725_000_017.0,
        text=text,
        status="processing",
        attempts=1,
        last_error=None,
    )


@pytest.mark.parametrize(
    "text, fact, quote",
    [
        # `карт\w*` блокировало обычную бытовую речь: картошку, картину,
        # карточную игру, картографию. Это не платёжные данные.
        ("Обожаю картошку с грибами", "Миша любит картошку", "Обожаю картошку"),
        ("Люблю рисовать картины маслом", "Миша рисует картины", "рисовать картины"),
        ("Я увлекаюсь картографией", "Миша увлекается картографией", "увлекаюсь картографией"),
        # `рак` — омоним: рак-отшельник не онкология.
        ("Я развожу раков-отшельников", "Миша разводит раков-отшельников", "развожу раков-отшельников"),
        # Префиксные `\w*` цепляли однокоренные слова без смысла категории.
        ("Я телефонный мастер", "Миша телефонный мастер", "телефонный мастер"),
        ("Я собираю счётчики Гейгера", "Миша собирает счётчики Гейгера", "собираю счётчики Гейгера"),
        ("Я бегаю долгие дистанции", "Миша бегает долгие дистанции", "бегаю долгие дистанции"),
        ("Я изучаю религиоведение", "Миша изучает религиоведение", "изучаю религиоведение"),
    ],
)
def test_policy_does_not_reject_innocent_homonyms(text, fact, quote):
    """Калибровка: категория должна ловить смысл, а не общий корень слова."""
    _validate_verified_fact(_policy_source(text), VerifiedMemoryFact(
        fact, quote, "profile.fact", "ok"
    ))


@pytest.mark.parametrize(
    "text, fact, quote",
    [
        # Платёжная карта по-прежнему обязана блокироваться — во всех формах.
        ("Моя банковская карта заблокирована", "У Миши банковская карта", "банковская карта"),
        ("Карта 4276 1600 1234 5678", "Карта Миши 4276 1600 1234 5678", "Карта 4276 1600 1234 5678"),
        ("Кредитная карта у меня в Сбере", "У Миши кредитная карта", "Кредитная карта"),
        # Медицинская, адресная и контактная категории тоже не должны ослабнуть.
        ("У меня онкология", "У Миши онкология", "У меня онкология"),
        ("Мой адрес — Лесная 5", "Адрес Миши: Лесная 5", "Мой адрес"),
        ("Я живу на улице Лесной", "Миша живёт на улице Лесной", "живу на улице Лесной"),
        ("Мой телефон 89001234567", "Телефон Миши 89001234567", "Мой телефон"),
        ("У меня долг по ипотеке", "У Миши долг по ипотеке", "долг по ипотеке"),
        ("Мой счёт в банке пуст", "У Миши счёт в банке", "Мой счёт в банке"),
    ],
)
def test_policy_still_rejects_real_sensitive_categories(text, fact, quote):
    """Сужение токенов не должно было ослабить ни одну категорию."""
    with pytest.raises(MemoryCandidateRejected):
        _validate_verified_fact(_policy_source(text), VerifiedMemoryFact(
            fact, quote, "profile.fact", "ошибочный KEEP"
        ))


@pytest.mark.parametrize(
    "text, fact, quote",
    [
        # Регресс: кавычки считались модальностью и в одиночку убивали факт.
        # Восемь дней подряд ни один факт не доходил до Mem0 именно из-за этого.
        (
            "Мой любимый напиток — грушевый лимонад «Тархун-77»",
            "Миша любит грушевый лимонад «Тархун-77»",
            "любимый напиток — грушевый лимонад «Тархун-77»",
        ),
        ('Я работаю в "Яндексе"', 'Миша работает в "Яндексе"', 'работаю в "Яндексе"'),
        ("У меня кот по кличке «Пельмень»", "У Миши кот «Пельмень»", "кот по кличке «Пельмень»"),
        # Модальность в соседнем предложении не относится к утверждению.
        (
            "Бер, как дела? Кстати, я живу в Санкт-Петербурге.",
            "Миша живёт в Санкт-Петербурге",
            "я живу в Санкт-Петербурге",
        ),
        (
            "Завтра еду к маме. Я работаю бэкенд-разработчиком на Python.",
            "Миша работает бэкенд-разработчиком на Python",
            "работаю бэкенд-разработчиком на Python",
        ),
    ],
)
def test_policy_allows_quotes_and_unrelated_modality(text, fact, quote):
    """Модальность — свойство утверждения, а не всего сообщения."""
    _validate_verified_fact(_policy_source(text), VerifiedMemoryFact(
        fact, quote, "profile.fact", "ok"
    ))


@pytest.mark.parametrize(
    "text, fact, quote",
    [
        # Сужение до предложения не должно открывать обход через короткую цитату.
        (
            "Сегодня отличная погода. Надеюсь переехать в Казань.",
            "Миша переедет в Казань",
            "переехать в Казань",
        ),
        (
            "Я живу в Питере. Может, куплю дом.",
            "Миша купит дом",
            "куплю дом",
        ),
        # Вопрос остаётся вопросом: знак конца входит в предложение.
        (
            "Я правда живу в Казани?",
            "Миша живёт в Казани",
            "живу в Казани",
        ),
    ],
)
def test_policy_still_rejects_modality_around_the_quote(text, fact, quote):
    """Соседнее предложение больше не вето, своё — по-прежнему вето."""
    with pytest.raises(MemoryCandidateRejected):
        _validate_verified_fact(_policy_source(text), VerifiedMemoryFact(
            fact, quote, "profile.fact", "ошибочный KEEP"
        ))


def test_sensitive_policy_still_spans_the_whole_source():
    """Приватность шире модальности: пароль рядом делает опасным всё сообщение."""
    with pytest.raises(MemoryCandidateRejected) as excinfo:
        _validate_verified_fact(
            _policy_source("Мой пароль qwerty123. Я люблю картошку."),
            VerifiedMemoryFact("Миша любит картошку", "люблю картошку", "profile.fact", "x"),
        )
    assert "парол" in str(excinfo.value).lower()


def test_policy_rejection_names_the_matched_token():
    """Причина отказа должна называть сработавший токен.

    Без этого калибровать порог можно только наугад: в логе видно «чувствительная
    категория», но не то, какое именно слово её вызвало.
    """
    with pytest.raises(MemoryCandidateRejected) as excinfo:
        _validate_verified_fact(
            _policy_source("У меня онкология"),
            VerifiedMemoryFact("У Миши онкология", "У меня онкология", "profile.fact", "x"),
        )
    assert "онколог" in str(excinfo.value).lower()


def test_source_enters_dead_letter_after_five_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=903,
        text="Я использую видеокарту RTX 5070 Ti",
        created_at=1_725_000_002.0,
    )

    class Extractor:
        async def extract(self, source):
            raise MemoryTransientError("DeepSeek временно недоступен")

    for attempt in range(1, 6):
        report = asyncio.run(process_pending_memory(Extractor(), object(), object()))
        if attempt < 5:
            assert report.retried == 1
            assert report.dead_lettered == 0
        else:
            assert report.retried == 0
            assert report.dead_lettered == 1

    row = state.list_memory_sources()[0]
    assert row.status == "dead"
    assert row.attempts == 5
    assert row.text == ""
    assert state.list_memory_facts("private_42") == []


def test_retrying_older_source_blocks_newer_conflicting_source(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    for message_id, city, created_at in (
        (906, "Москве", 1_725_000_005.0),
        (907, "Казани", 1_725_000_006.0),
    ):
        enqueue_memory_source(
            scope="private_42",
            author_id="42",
            author_name="Миша",
            message_id=message_id,
            text=f"Я постоянно живу в {city}",
            created_at=created_at,
        )

    class Extractor:
        async def extract(self, source):
            if source.message_id == 906:
                raise MemoryTransientError("временный сбой старого сообщения")
            raise AssertionError("новое сообщение не должно обогнать старое")

    report = asyncio.run(process_pending_memory(Extractor(), object(), object()))

    rows = state.list_memory_sources()
    assert report.processed == 1
    assert report.retried == 1
    assert [(row.message_id, row.status, row.attempts) for row in rows] == [
        (906, "pending", 1),
        (907, "pending", 0),
    ]


def test_partial_store_failure_rolls_back_all_facts_from_source(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=915,
        text="Я постоянно использую Fedora, а моя видеокарта — RTX 5070 Ti",
        created_at=1_725_000_014.0,
    )

    class Extractor:
        async def extract(self, source):
            return [
                MemoryCandidate(
                    "Миша постоянно использует Fedora",
                    "постоянно использую Fedora",
                    "software.os",
                ),
                MemoryCandidate(
                    "Видеокарта Миши — RTX 5070 Ti",
                    "моя видеокарта — RTX 5070 Ti",
                    "hardware.gpu",
                ),
            ]

    class Verifier:
        async def verify(self, source, candidate):
            return VerifiedMemoryFact(
                candidate.fact,
                candidate.source_quote,
                candidate.fact_key,
                "прямое утверждение",
            )

    class Store:
        def __init__(self):
            self.saved = []
            self.deleted = []

        async def save(self, source, fact):
            if self.saved:
                raise MemoryTransientError("второй Mem0 write не удался")
            self.saved.append("mem0-1")
            return "mem0-1"

        async def delete(self, memory_id):
            self.deleted.append(memory_id)

        async def restore(self, previous):
            raise AssertionError("новый факт не требует restore")

    store = Store()
    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), store))

    assert report.retried == 1
    assert store.deleted == ["mem0-1"]
    assert state.list_memory_facts("private_42") == []


def test_registry_failure_deletes_staged_mem0_fact(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=916,
        text="Я постоянно использую Fedora Linux",
        created_at=1_725_000_015.0,
    )

    class Extractor:
        async def extract(self, source):
            return [
                MemoryCandidate(
                    "Миша постоянно использует Fedora Linux",
                    "постоянно использую Fedora Linux",
                    "software.os",
                )
            ]

    class Verifier:
        async def verify(self, source, candidate):
            return VerifiedMemoryFact(
                candidate.fact,
                candidate.source_quote,
                candidate.fact_key,
                "прямое утверждение",
            )

    class Store:
        def __init__(self):
            self.deleted = []

        async def save(self, source, fact):
            return "mem0-1"

        async def delete(self, memory_id):
            self.deleted.append(memory_id)

        async def restore(self, previous):
            raise AssertionError("новый факт не требует restore")

    def fail_commit(*args, **kwargs):
        raise RuntimeError("SQLite commit failed")

    monkeypatch.setattr(state, "commit_memory_facts", fail_commit)
    store = Store()

    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), store))

    assert report.retried == 1
    assert store.deleted == ["mem0-1"]
    assert state.list_memory_facts("private_42") == []


def test_mem0_indexes_exact_approved_fact_without_inference(monkeypatch):
    class Memory:
        def __init__(self):
            self.calls = []

        def add(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return {
                "results": [{
                    "id": "mem0-1",
                    "memory": "Миша использует RTX 5070 Ti",
                    "event": "ADD",
                }]
            }

    memory = Memory()
    monkeypatch.setattr(memory_store, "memory", memory)
    source = state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=901,
        created_at=1_725_000_000.0,
        text="Я использую видеокарту RTX 5070 Ti",
        status="processing",
        attempts=1,
        last_error=None,
    )
    fact = VerifiedMemoryFact(
        fact="Миша использует RTX 5070 Ti",
        source_quote="видеокарту RTX 5070 Ti",
        fact_key="hardware.gpu",
        reason="прямое утверждение",
    )

    memory_id = asyncio.run(Mem0ApprovedFactStore().save(source, fact))

    assert memory_id == "mem0-1"
    assert memory.calls[0][0] == [{"role": "user", "content": fact.fact}]
    assert memory.calls[0][1]["user_id"] == "private_42"
    assert memory.calls[0][1]["infer"] is False
    assert memory.calls[0][1]["metadata"]["source_quote"] == fact.source_quote


def test_mem0_rejects_whitespace_rewrite_and_deletes_result(monkeypatch):
    class Memory:
        def __init__(self):
            self.deleted = []

        def add(self, messages, **kwargs):
            return {
                "results": [
                    {
                        "id": "mem0-1",
                        "memory": "Миша  использует Fedora Linux",
                        "event": "ADD",
                    }
                ]
            }

        def delete(self, memory_id):
            self.deleted.append(memory_id)

    memory = Memory()
    monkeypatch.setattr(memory_store, "memory", memory)
    source = state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=918,
        created_at=1_725_000_017.0,
        text="Я постоянно использую Fedora Linux",
        status="processing",
        attempts=1,
        last_error=None,
    )
    fact = VerifiedMemoryFact(
        "Миша использует Fedora Linux",
        "использую Fedora Linux",
        "software.os",
        "прямое утверждение",
    )

    with pytest.raises(MemoryTransientError, match="изменил"):
        asyncio.run(Mem0ApprovedFactStore().save(source, fact))

    assert memory.deleted == ["mem0-1"]


def test_mem0_cleanup_failure_is_reported(monkeypatch):
    class Memory:
        def add(self, messages, **kwargs):
            return {
                "results": [
                    {"id": "mem0-1", "memory": "подмена", "event": "ADD"}
                ]
            }

        def delete(self, memory_id):
            raise RuntimeError("Qdrant unavailable")

    monkeypatch.setattr(memory_store, "memory", Memory())
    source = state.MemorySourceRecord(
        id=1,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=919,
        created_at=1_725_000_018.0,
        text="Я постоянно использую Fedora Linux",
        status="processing",
        attempts=1,
        last_error=None,
    )
    fact = VerifiedMemoryFact(
        "Миша использует Fedora Linux",
        "использую Fedora Linux",
        "software.os",
        "прямое утверждение",
    )

    with pytest.raises(MemoryTransientError, match="не удалось удалить"):
        asyncio.run(Mem0ApprovedFactStore().save(source, fact))


def test_mem0_reconcile_deletes_orphans_and_restores_registered_fact(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()
    state.upsert_memory_fact(
        scope="private_42",
        subject_id="42",
        fact_key="profile.city",
        fact="Миша живёт в Москве",
        source_id=1,
        source_quote="живу в Москве",
        source_message_id=900,
        source_created_at=1_725_000_000.0,
        mem0_id="registered",
    )

    class Memory:
        def __init__(self):
            self.deleted = []
            self.updated = []

        def get_all(self, **kwargs):
            return {
                "results": [
                    {"id": "registered", "memory": "новая незавершённая версия"},
                    {"id": "orphan", "memory": "незавершённый новый факт"},
                ]
            }

        def update(self, memory_id, data, metadata=None):
            self.updated.append((memory_id, data, metadata))

        def get(self, memory_id):
            return {"id": memory_id, "memory": "Миша живёт в Москве"}

        def delete(self, memory_id):
            self.deleted.append(memory_id)

    memory = Memory()
    monkeypatch.setattr(memory_store, "memory", memory)
    source = state.MemorySourceRecord(
        id=2,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=920,
        created_at=1_725_000_019.0,
        text="Я постоянно живу в Казани",
        status="processing",
        attempts=2,
        last_error="process crashed",
    )

    asyncio.run(Mem0ApprovedFactStore().reconcile(source))

    assert memory.deleted == ["orphan"]
    assert memory.updated[0][0:2] == ("registered", "Миша живёт в Москве")


def test_verifier_receives_existing_fact_keys_for_canonical_replacement(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()
    state.upsert_memory_fact(
        scope="private_42",
        subject_id="42",
        fact_key="profile.city",
        fact="Миша живёт в Москве",
        source_id=1,
        source_quote="Я живу в Москве",
        source_message_id=900,
        source_created_at=1_725_000_000.0,
        mem0_id="mem0-1",
    )
    source = state.MemorySourceRecord(
        id=2,
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=901,
        created_at=1_725_000_001.0,
        text="Я постоянно живу в Казани",
        status="processing",
        attempts=1,
        last_error=None,
    )
    captured = {}

    async def fake_deepseek_json(*, system, user, max_tokens):
        captured.update(json.loads(user))
        return {
            "decision": "KEEP",
            "fact": "Миша постоянно живёт в Казани",
            "source_quote": "постоянно живу в Казани",
            "fact_key": "profile.city",
            "reason": "замена того же свойства",
        }

    monkeypatch.setattr(memory_pipeline, "_deepseek_json", fake_deepseek_json)

    result = asyncio.run(
        DeepSeekMemoryVerifier().verify(
            source,
            MemoryCandidate(
                "Миша постоянно живёт в Казани",
                "постоянно живу в Казани",
                "location.city",
            ),
        )
    )

    assert isinstance(result, VerifiedMemoryFact)
    assert result.fact_key == "profile.city"
    assert captured["existing_facts"] == [
        {"fact_key": "profile.city", "fact": "Миша живёт в Москве"}
    ]


def test_mem0_replaces_approved_fact_in_place(monkeypatch):
    class Memory:
        def __init__(self):
            self.updates = []

        def update(self, memory_id, data, metadata=None):
            self.updates.append((memory_id, data, metadata))
            return {"message": "Memory updated successfully!"}

        def get(self, memory_id):
            return {
                "id": memory_id,
                "memory": "Миша постоянно живёт в Казани",
            }

    memory = Memory()
    monkeypatch.setattr(memory_store, "memory", memory)
    source = state.MemorySourceRecord(
        id=2,
        scope="group_7",
        author_id="42",
        author_name="Миша",
        message_id=911,
        created_at=1_725_000_011.0,
        text="Я постоянно живу в Казани",
        status="processing",
        attempts=1,
        last_error=None,
    )
    fact = VerifiedMemoryFact(
        fact="Миша постоянно живёт в Казани",
        source_quote="постоянно живу в Казани",
        fact_key="profile.city",
        reason="прямое утверждение",
    )

    memory_id = asyncio.run(Mem0ApprovedFactStore().replace("mem0-1", source, fact))

    assert memory_id == "mem0-1"
    assert memory.updates[0][0:2] == ("mem0-1", fact.fact)
    assert memory.updates[0][2]["source_message_id"] == "911"


def test_new_self_statement_replaces_same_fact_key(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from berangaria.memory.pipeline import enqueue_memory_source

    class Extractor:
        async def extract(self, source):
            city = "Москва" if "Москве" in source.text else "Казань"
            quote = "постоянно живу в Москве" if city == "Москва" else "постоянно живу в Казани"
            return [MemoryCandidate(f"Миша живёт в городе {city}", quote, "profile.city")]

    class Verifier:
        async def verify(self, source, candidate):
            return VerifiedMemoryFact(
                candidate.fact,
                candidate.source_quote,
                candidate.fact_key,
                "прямое утверждение",
            )

    class Store:
        def __init__(self):
            self.next_id = 1
            self.replaced = []

        async def save(self, source, fact):
            memory_id = f"mem0-{self.next_id}"
            self.next_id += 1
            return memory_id

        async def replace(self, memory_id, source, fact):
            self.replaced.append((memory_id, fact.fact))
            return memory_id

    store = Store()
    enqueue_memory_source(
        scope="group_7",
        author_id="42",
        author_name="Миша",
        message_id=910,
        text="Я постоянно живу в Москве",
        created_at=1_725_000_010.0,
    )
    asyncio.run(process_pending_memory(Extractor(), Verifier(), store))
    enqueue_memory_source(
        scope="group_7",
        author_id="42",
        author_name="Миша",
        message_id=911,
        text="Я постоянно живу в Казани",
        created_at=1_725_000_011.0,
    )
    asyncio.run(process_pending_memory(Extractor(), Verifier(), store))

    facts = state.list_memory_facts("group_7")
    assert [fact.fact for fact in facts] == ["Миша живёт в городе Казань"]
    assert facts[0].mem0_id == "mem0-1"
    assert store.replaced == [("mem0-1", "Миша живёт в городе Казань")]
