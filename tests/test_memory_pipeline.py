import state
import asyncio

from memory_pipeline import (
    Mem0ApprovedFactStore,
    MemoryCandidate,
    MemoryTransientError,
    VerifiedMemoryFact,
    process_pending_memory,
)
import memory_store


def test_text_message_is_queued_with_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from memory_pipeline import enqueue_memory_source

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

    from memory_pipeline import enqueue_memory_source

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


def test_approved_candidate_is_stored_with_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from memory_pipeline import enqueue_memory_source

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


def test_discarded_candidate_never_reaches_memory_store(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from memory_pipeline import enqueue_memory_source

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
            return None

    class Store:
        async def save(self, source, fact):
            raise AssertionError("DISCARD не должен вызывать storage")

        async def delete(self, memory_id):
            raise AssertionError("DISCARD не должен вызывать delete")

    report = asyncio.run(process_pending_memory(Extractor(), Verifier(), Store()))

    assert report.approved == 0
    assert report.discarded == 1
    assert state.list_memory_sources(status="completed")
    assert state.list_memory_facts("group_7") == []


def test_keep_without_exact_source_quote_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from memory_pipeline import enqueue_memory_source

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


def test_sensitive_source_quote_is_rejected_even_if_fact_hides_category(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from memory_pipeline import enqueue_memory_source

    enqueue_memory_source(
        scope="private_42",
        author_id="42",
        author_name="Миша",
        message_id=905,
        text="Мой точный адрес: улица Ленина, дом 1",
        created_at=1_725_000_004.0,
    )

    class Extractor:
        async def extract(self, source):
            return [
                MemoryCandidate(
                    "Миша живёт на улице Ленина, дом 1",
                    "адрес: улица Ленина, дом 1",
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


def test_source_enters_dead_letter_after_five_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()

    from memory_pipeline import enqueue_memory_source

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

    from memory_pipeline import enqueue_memory_source

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

    from memory_pipeline import enqueue_memory_source

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
