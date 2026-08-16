"""
Характеризационные тесты чистых хелперов llm_client.

Фиксируют ТЕКУЩЕЕ поведение форматирования/очистки/рендера истории,
чтобы последующий рефакторинг send_llm_request нельзя было провести
с молчаливым изменением логики.
"""
import asyncio
import copy

import pytest

from berangaria.chat import llm_client
from berangaria.config import (
    CHAT_PROVIDER_PREFERENCES,
    _normalize_chat_provider,
    apply_chat_routing,
)
from berangaria import config as bot_config
from berangaria.core import state

from berangaria.chat.llm_client import (
    markdown_to_html,
    strip_markdown,
    _clean_reply,
    _render_history_for_api,
    _build_sid_map,
    _renumber_sids,
    _extract_plain_text,
    _format_memory_block,
    _count_memory_block_facts,
    _filter_approved_memory_results,
    _is_meaningful_memory_query,
    _build_memory_search_query,
    _build_memory_relevance_query,
    _approved_memory_recall_results,
)


# ---------- markdown_to_html ----------

def test_markdown_html_escapes_special_chars():
    assert markdown_to_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_markdown_html_bold_italic_code():
    assert markdown_to_html("**bold**") == "<b>bold</b>"
    assert markdown_to_html("*it*") == "<i>it</i>"
    assert markdown_to_html("***bi***") == "<b><i>bi</i></b>"
    assert markdown_to_html("`code`") == "<code>code</code>"
    assert markdown_to_html("~~x~~") == "<s>x</s>"


def test_markdown_html_link():
    assert markdown_to_html("[t](http://x.io)") == '<a href="http://x.io">t</a>'


def test_markdown_html_italic_not_inside_word():
    # *text* внутри слова не превращается в курсив
    assert markdown_to_html("a*b*c") == "a*b*c"


# ---------- strip_markdown ----------

def test_strip_markdown_removes_markup():
    assert strip_markdown("**bold**") == "bold"
    assert strip_markdown("[t](http://x)") == "t (http://x)"
    assert strip_markdown("`code`") == "code"


# ---------- _clean_reply ----------

def test_clean_reply_trailing_single_dot_removed():
    assert _clean_reply("привет.") == "привет"


def test_clean_reply_ellipsis_preserved_as_silence():
    # Голое многоточие — это «молчание», сводится к пустой строке
    assert _clean_reply("...") == ""


def test_clean_reply_strips_think_block():
    assert _clean_reply("<think>рассуждения</think>ответ") == "ответ"


def test_clean_reply_keeps_emoji():
    # Эмодзи больше не вырезаются — промпт отговаривает, но не затыкает.
    assert _clean_reply("текст 😀🔥") == "текст 😀🔥"


def test_clean_reply_silence_word():
    assert _clean_reply("молчу") == ""
    assert _clean_reply("(молчит)") == ""


def test_clean_reply_keeps_normal_text():
    assert _clean_reply("да, согласна") == "да, согласна"


def test_clean_reply_strips_internal_reply_handles():
    reply = "Из текущего чата, [#26] и [#27]. Ты написал про Helix."

    assert _clean_reply(reply) == "Из текущего чата. Ты написал про Helix"


def test_clean_reply_humanizes_internal_memory_tag():
    reply = "Факта нет ни в [Context from memory], ни в сообщениях чата."

    assert _clean_reply(reply) == "Факта нет ни в долгосрочной памяти, ни в сообщениях чата"


# ---------- _render_history_for_api ----------

def test_render_prepends_sid_to_user():
    hist = [{"role": "user", "content": "hi", "sid": 3, "mid": 100}]
    out = _render_history_for_api(hist)
    assert out == [{"role": "user", "content": "[#3] hi"}]
    # служебные ключи sid/mid не утекают в payload
    assert "sid" not in out[0] and "mid" not in out[0]


def test_render_plain_assistant_untouched():
    hist = [{"role": "assistant", "content": "ответ"}]
    assert _render_history_for_api(hist) == [{"role": "assistant", "content": "ответ"}]


def test_render_assistant_reaction_becomes_system_note():
    hist = [{"role": "assistant", "content": "", "reactions": [{"emoji": "🔥", "on": None}]}]
    out = _render_history_for_api(hist)
    # реакция уходит отдельной system-строкой, без пустого assistant
    assert len(out) == 1
    assert out[0]["role"] == "system"
    assert "🔥" in out[0]["content"]


def test_render_reaction_resolves_live_sid_from_mid():
    hist = [
        {"role": "user", "content": "[Message: шутка]", "sid": 3, "mid": 99},
        {
            "role": "assistant",
            "content": "",
            "reactions": [{"emoji": "🤡", "on_mid": 99, "on_sid": 139, "on": "шутка"}],
        },
    ]
    out = _render_history_for_api(hist)
    sys_notes = [m for m in out if m["role"] == "system"]
    assert sys_notes
    # sid в истории уже 3 (после renumber), не протухший 139
    assert "🤡 на [#3]" in sys_notes[0]["content"]
    assert "шутка" in sys_notes[0]["content"]


def test_render_user_without_sid_unchanged():
    hist = [{"role": "user", "content": "hi"}]
    assert _render_history_for_api(hist) == [{"role": "user", "content": "hi"}]


def test_render_assistant_voice_like_sticker_note():
    """Voice is an action note with quoted speech — not typed assistant prose."""
    hist = [{
        "role": "assistant",
        "content": "",
        "voices": [{"text": "Ну да. Конечно.", "emotion": "sarcastic"}],
    }]
    out = _render_history_for_api(hist)
    assert len(out) == 1
    assert out[0]["role"] == "system"
    assert "голосовое" in out[0]["content"]
    assert "Ну да. Конечно." in out[0]["content"]
    assert "sarcastic" in out[0]["content"]
    assert "действия в чате" in out[0]["content"]
    # Must not also appear as a normal assistant message (that looked like typing).
    assert not any(m.get("role") == "assistant" for m in out)


def test_render_legacy_voice_row_with_content_not_duplicated():
    """Older rows may have spoken text in content; still only one voice note."""
    hist = [{
        "role": "assistant",
        "content": "Ну да. Конечно.",
        "voices": [{"text": "Ну да. Конечно.", "emotion": "calm"}],
    }]
    out = _render_history_for_api(hist)
    assistant_msgs = [m for m in out if m["role"] == "assistant"]
    assert assistant_msgs == []
    assert any(
        m["role"] == "system" and "голосовое" in m["content"] and "Ну да" in m["content"]
        for m in out
    )


# ---------- _build_sid_map / _renumber_sids ----------

def test_build_sid_map():
    hist = [
        {"role": "user", "content": "a", "sid": 1, "mid": 10},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c", "sid": 2, "mid": 20},
    ]
    assert _build_sid_map(hist) == {1: 10, 2: 20}


def test_renumber_sids_from_one():
    entries = [
        {"role": "user", "content": "a", "sid": 5},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c", "sid": 9},
    ]
    _renumber_sids(entries)
    assert entries[0]["sid"] == 1
    assert entries[2]["sid"] == 2


# ---------- _extract_plain_text ----------

def test_extract_plain_text_message_tag():
    assert _extract_plain_text("[Message: привет мир]") == "привет мир"


def test_extract_plain_text_list_content():
    content = [{"type": "text", "text": "[Message: hey]"}]
    assert _extract_plain_text(content) == "hey"


def test_extract_plain_text_non_str_returns_empty():
    assert _extract_plain_text(123) == ""


def test_meaningful_memory_query_rejects_trivial_short_text():
    assert not _is_meaningful_memory_query("Ладно")
    assert not _is_meaningful_memory_query("(сообщение без текста)")
    assert not _is_meaningful_memory_query("https://vt.tiktok.com/ZSCKeAjpT/")


def test_build_memory_search_query_uses_recent_meaningful_message():
    hist = [
        {"role": "user", "content": "[Message: обсуждали свежую систему памяти бота]"},
        {"role": "assistant", "content": "ответ"},
        {"role": "user", "content": "[Message: Ладно]"},
    ]
    assert _build_memory_search_query(hist, "Миша") == "обсуждали свежую систему памяти бота"


def test_build_memory_search_query_returns_empty_for_trivial_history():
    hist = [{"role": "user", "content": "[Message: пон]"}]
    assert _build_memory_search_query(hist, "Миша") == ""


def test_memory_relevance_query_uses_current_topic_only():
    hist = [
        {"role": "user", "content": "[Message: мой редактор Helix]"},
        {"role": "user", "content": "[Message: нужен ли сегодня зонт из-за погоды]"},
    ]

    assert _build_memory_relevance_query(hist, "Миша") == "нужен ли сегодня зонт из-за погоды"


# ---------- _format_memory_block ----------

def test_format_memory_empty():
    assert _format_memory_block({}) == ""
    assert _format_memory_block({"results": []}) == ""


def test_format_memory_filters_below_min_score(monkeypatch):
    # Pin the floor so this unit test does not track production memory_min_score.
    floor = 0.3
    monkeypatch.setattr(llm_client, "MEMORY_MIN_SCORE", floor)
    res = {"results": [
        {"memory": "пороговый факт", "score": floor},
        {"memory": "слабый факт", "score": floor - 0.01},
    ]}
    out = _format_memory_block(res)
    assert "пороговый факт" in out
    assert "слабый факт" not in out


def test_memory_results_include_only_registered_ids_from_same_scope(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()
    state.upsert_memory_fact(
        scope="private_42",
        subject_id="42",
        fact_key="software.os",
        fact="Миша использует Fedora",
        source_id=1,
        source_quote="использую Fedora",
        source_message_id=901,
        source_created_at=1_725_000_000.0,
        mem0_id="approved-private",
    )
    state.upsert_memory_fact(
        scope="group_7",
        subject_id="42",
        fact_key="software.os",
        fact="Миша использует Arch",
        source_id=2,
        source_quote="использую Arch",
        source_message_id=902,
        source_created_at=1_725_000_001.0,
        mem0_id="approved-group",
    )
    raw = {
        "results": [
            {"id": "approved-private", "memory": "Миша использует Fedora", "score": 0.9},
            {"id": "approved-private", "memory": "Миша  использует Fedora", "score": 0.95},
            {"id": "approved-private", "memory": "подменённый факт", "score": 0.99},
            {"id": "approved-group", "memory": "Миша использует Arch", "score": 0.9},
            {"id": "legacy-unapproved", "memory": "непроверенный факт", "score": 0.99},
        ]
    }

    filtered = _filter_approved_memory_results(raw, "private_42")

    assert [item["id"] for item in filtered["results"]] == ["approved-private"]


def test_general_recall_reads_only_approved_facts_from_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "memory.db"))
    state.init_db()
    state.upsert_memory_fact(
        scope="group_7",
        subject_id="42",
        fact_key="preferences.text_editor",
        fact="Миша использует Helix",
        source_id=1,
        source_quote="использую Helix",
        source_message_id=901,
        source_created_at=1_725_000_000.0,
        mem0_id="approved-group",
    )
    state.upsert_memory_fact(
        scope="private_42",
        subject_id="42",
        fact_key="software.os",
        fact="Миша использует Fedora",
        source_id=2,
        source_quote="использую Fedora",
        source_message_id=902,
        source_created_at=1_725_000_001.0,
        mem0_id="approved-private",
    )

    results = _approved_memory_recall_results("group_7")

    assert results == {
        "results": [
            {
                "id": "approved-group",
                "memory": "Миша использует Helix",
                "score": 1.0,
            }
        ]
    }


def test_format_memory_formats_as_bullet():
    res = {"results": [{"memory": "факт", "score": 0.9}]}
    assert _format_memory_block(res) == "- факт"


def test_format_memory_rejects_fact_unrelated_to_current_query():
    res = {
        "results": [
            {
                "memory": "Пользователь titlo10 использует Helix как основной редактор",
                "score": 0.9,
            }
        ]
    }

    assert _format_memory_block(res, query="нужен ли зонт из-за погоды") == ""


def test_format_memory_keeps_fact_related_to_current_query():
    res = {
        "results": [
            {
                "memory": "Пользователь titlo10 использует Helix как основной редактор",
                "score": 0.9,
            }
        ]
    }

    assert "Helix" in _format_memory_block(res, query="какой у меня редактор Helix")


def test_format_memory_keeps_live_relevant_score_above_vector_floor(monkeypatch):
    floor = 0.3
    monkeypatch.setattr(llm_client, "MEMORY_MIN_SCORE", floor)
    res = {
        "results": [
            {
                "memory": "Пользователь titlo10 использует Helix как основной редактор",
                # Slightly above the floor — vector match is weak but topical.
                "score": floor + 0.0465,
            }
        ]
    }

    query = "какой у меня основной текстовый редактор? если не знаешь — не угадывай"
    assert "Helix" in _format_memory_block(res, query=query)


def test_format_memory_allows_explicit_general_recall_query():
    res = {
        "results": [
            {
                "memory": "Пользователь titlo10 использует Helix как основной редактор",
                "score": 0.9,
            }
        ]
    }

    assert "Helix" in _format_memory_block(res, query="что ты обо мне помнишь?")


def test_format_memory_does_not_treat_topical_recall_as_general_recall():
    res = {
        "results": [
            {
                "memory": "Пользователь titlo10 использует Helix как основной редактор",
                "score": 0.9,
            }
        ]
    }

    assert _format_memory_block(res, query="что ты помнишь про погоду?") == ""


def test_count_memory_block_facts_counts_single_line():
    assert _count_memory_block_facts("- Пользователь использует Helix") == 1


def test_failed_summary_does_not_mutate_live_history(monkeypatch):
    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise RuntimeError("forced failure")

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", FailingClient)
    history = [
        {"role": "user", "content": f"[Message: m{i}]", "sid": i + 1, "mid": i + 10}
        for i in range(12)
    ]
    before = copy.deepcopy(history)

    result = asyncio.run(llm_client.summarize_history(history))

    assert result is history
    assert history == before


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _summary_history(n=12):
    return [
        {"role": "user", "content": f"[Message: m{i}]", "sid": i + 1, "mid": i + 10}
        for i in range(n)
    ]


def test_successful_summary_enables_reasoning_and_returns_new_list(monkeypatch):
    captured = {}

    class OkClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return _FakeResponse(
                payload={
                    "choices": [
                        {"message": {"content": "Важные факты: RTX 5070 Ti, решили ждать."}}
                    ]
                }
            )

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", OkClient)
    history = _summary_history(12)
    before = copy.deepcopy(history)

    result = asyncio.run(llm_client.summarize_history(history))

    assert result is not history
    assert history == before
    assert result[0]["role"] == "user"
    assert result[0]["content"].startswith("[Previous conversation summary:")
    assert "RTX 5070 Ti" in result[0]["content"]
    assert len(result) == llm_client.SUMMARY_INTERVAL + 1
    assert captured["url"] == llm_client.CHAT_API_URL
    assert captured["headers"]["Authorization"] == "Bearer test-openrouter-key"
    assert captured["headers"]["X-Title"] == "Berangaria"
    assert captured["payload"]["model"] == llm_client.MODEL
    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert "thinking" not in captured["payload"]
    assert "reasoning_effort" not in captured["payload"]
    assert captured["payload"]["max_tokens"] == 8192
    assert "top_k" not in captured["payload"]
    assert captured["timeout"] == 120.0
    if CHAT_PROVIDER_PREFERENCES:
        assert captured["payload"]["provider"] == CHAT_PROVIDER_PREFERENCES
    else:
        assert "provider" not in captured["payload"]


def test_summary_null_content_does_not_crash_and_keeps_history(monkeypatch):
    """content=null раньше ронял re.sub → TypeError → «Ошибка суммаризации»."""

    class EmptyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return _FakeResponse(
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning_content": "долгий CoT без финального ответа",
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", EmptyClient)
    history = _summary_history(12)
    before = copy.deepcopy(history)

    result = asyncio.run(llm_client.summarize_history(history))

    assert result is history
    assert history == before


def test_estimate_request_cost_prefers_provider_usage_cost():
    cost = llm_client._estimate_request_cost(
        {"cost": "0.00123"},
        prompt_tokens=1000,
        completion_tokens=500,
        cached_tokens=0,
        cache_write_tokens=0,
    )
    assert cost == 0.00123


def test_estimate_request_cost_splits_cache_write_from_uncached():
    cost = llm_client._estimate_request_cost(
        {},
        prompt_tokens=1000,
        completion_tokens=100,
        cached_tokens=400,
        cache_write_tokens=200,
    )
    expected = (
        (400 / 1_000_000) * llm_client.PRICE_PROMPT_CACHE_MISS
        + (400 / 1_000_000) * llm_client.PRICE_PROMPT_CACHE_HIT
        + (200 / 1_000_000) * llm_client.PRICE_PROMPT_CACHE_WRITE
        + (100 / 1_000_000) * llm_client.PRICE_COMPLETION
    )
    assert cost == expected


def test_normalize_chat_provider_accepts_auto_aliases():
    assert _normalize_chat_provider(None) == "auto"
    assert _normalize_chat_provider("AUTO") == "auto"
    assert _normalize_chat_provider(" default ") == "auto"
    assert _normalize_chat_provider("OpenAI") == "openai"
    assert _normalize_chat_provider("amazon-bedrock") == "amazon-bedrock"
    assert _normalize_chat_provider("azure/us") == "azure/us"


def test_normalize_chat_provider_rejects_junk():
    with pytest.raises(ValueError):
        _normalize_chat_provider("openai; drop")
    with pytest.raises(ValueError):
        _normalize_chat_provider(["openai"])


def test_apply_chat_routing_omits_provider_in_auto(monkeypatch):
    monkeypatch.setattr(bot_config, "CHAT_PROVIDER_PREFERENCES", None)
    payload = apply_chat_routing({"model": "openai/gpt-5.6-luna"})
    assert "provider" not in payload


def test_apply_chat_routing_pins_openai(monkeypatch):
    prefs = {"order": ["openai"], "allow_fallbacks": True}
    monkeypatch.setattr(bot_config, "CHAT_PROVIDER_PREFERENCES", prefs)
    payload = apply_chat_routing({"model": "openai/gpt-5.6-luna"})
    assert payload["provider"] == prefs
