"""Build and validate long-term-memory context for chat requests."""

import re
from collections.abc import Callable

from berangaria.core import state
from berangaria.core.utils import is_low_signal_user_text

_MEMORY_TERM_RE = re.compile(r"[^\W_]{4,}", flags=re.UNICODE)
_MEMORY_RECALL_RE = re.compile(
    r"\b(?:что|чего)\s+ты\s+(?:обо?\s+мне|про\s+меня)\s+помни\w*|"
    r"\bчто\s+ты\s+знаешь\s+(?:обо?\s+мне|про\s+меня)|"
    r"\b(?:расскажи|напомни)\w*(?:\s+мне)?\s+"
    r"(?:обо?\s+мне|про\s+меня)",
    flags=re.IGNORECASE,
)
_MEMORY_STOP_WORDS = {
    "какой",
    "какая",
    "какие",
    "который",
    "которая",
    "которые",
    "меня",
    "мне",
    "тебя",
    "тебе",
    "твой",
    "твоя",
    "свой",
    "своя",
    "пользователь",
    "использует",
    "сейчас",
    "сегодня",
    "просто",
    "скажи",
    "назови",
    "пожалуйста",
    "about",
    "what",
    "which",
    "user",
}


def is_meaningful_query(text: str, *, min_chars: int) -> bool:
    """Reject short, service-only, and URL-only retrieval queries."""
    return not is_low_signal_user_text(text, min_alnum=min_chars)


def build_search_query(
    history: list,
    user_name: str,
    *,
    min_chars: int,
    recent_messages: int,
    extract_plain_text: Callable[[object], str],
) -> str:
    """Build retrieval text from the most recent meaningful user messages."""
    candidates: list[str] = []
    for entry in reversed(history or []):
        if entry.get("role") != "user":
            continue
        plain = extract_plain_text(entry.get("content", ""))
        if not is_meaningful_query(plain, min_chars=min_chars):
            continue
        candidates.append(plain)
        if len(candidates) >= recent_messages:
            break
    if candidates:
        return "\n".join(reversed(candidates))[:1000]
    return (
        user_name
        if is_meaningful_query(user_name, min_chars=min_chars)
        else ""
    )


def build_relevance_query(
    history: list,
    user_name: str,
    *,
    min_chars: int,
    extract_plain_text: Callable[[object], str],
) -> str:
    """Return only the latest meaningful topic for fail-closed filtering."""
    for entry in reversed(history or []):
        if entry.get("role") != "user":
            continue
        plain = extract_plain_text(entry.get("content", ""))
        if is_meaningful_query(plain, min_chars=min_chars):
            return plain[:1000]
    return (
        user_name
        if is_meaningful_query(user_name, min_chars=min_chars)
        else ""
    )


def _memory_terms(text: str) -> set[str]:
    return {
        token
        for token in _MEMORY_TERM_RE.findall((text or "").casefold())
        if token not in _MEMORY_STOP_WORDS
    }


def is_general_recall(query: str) -> bool:
    return bool(_MEMORY_RECALL_RE.search(query or ""))


def _fact_matches_query(fact: str, query: str) -> bool:
    if not query or is_general_recall(query):
        return True
    fact_terms = _memory_terms(fact)
    query_terms = _memory_terms(query)
    return any(
        fact_term == query_term
        or (
            len(fact_term) >= 5
            and len(query_term) >= 5
            and fact_term[:5] == query_term[:5]
        )
        for fact_term in fact_terms
        for query_term in query_terms
    )


def approved_recall_results(scope: str, *, search_limit: int) -> dict:
    """Return only facts from the approved SQLite registry."""
    facts = state.list_memory_facts(scope)[-search_limit:]
    return {
        "results": [
            {"id": fact.mem0_id, "memory": fact.fact, "score": 1.0}
            for fact in facts
        ]
    }


def format_memory_block(
    mem_results: dict,
    query: str = "",
    *,
    min_score: float,
    max_chars: int,
    search_limit: int,
) -> str:
    """Format relevant memory facts under the configured size budget."""
    results = (mem_results or {}).get("results") or []
    if not results:
        return ""
    results = sorted(
        results, key=lambda item: item.get("score") or 0.0, reverse=True
    )
    lines = []
    total = 0
    for item in results:
        if item.get("score", 0.0) < min_score:
            continue
        fact = (item.get("memory") or "").strip()
        if not fact or not _fact_matches_query(fact, query):
            continue
        line = f"- {fact}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
        if len(lines) >= search_limit:
            break
    return "\n".join(lines)


def count_memory_block_facts(mem_text: str) -> int:
    """Count facts that survived formatting."""
    return sum(1 for line in mem_text.splitlines() if line.startswith("- "))


def filter_approved_results(mem_results: dict, scope: str) -> dict:
    """Fail closed by checking scope, ID, and exact text against SQLite."""
    approved = {
        fact.mem0_id: fact.fact for fact in state.list_memory_facts(scope)
    }
    raw_results = (mem_results or {}).get("results") or []
    results = []
    seen_ids: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        memory_id = str(item.get("id") or "")
        memory_text = item.get("memory")
        if (
            not memory_id
            or memory_id in seen_ids
            or approved.get(memory_id) != memory_text
        ):
            continue
        seen_ids.add(memory_id)
        results.append(item)
    return {"results": results}
