"""
Характеризационные тесты чистых хелперов llm_client.

Фиксируют ТЕКУЩЕЕ поведение форматирования/очистки/рендера истории,
чтобы последующий рефакторинг send_llm_request нельзя было провести
с молчаливым изменением логики.
"""
import asyncio
import copy
import logging

from berangaria.chat import llm_client
from berangaria.chat import history_rendering, llm_diagnostics, reply_formatting
from berangaria.config import (
    CHAT_API_URL,
    GENERATION_PARAMS,
    MODEL,
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
    _build_system_prompt,
    _build_payload_prefix,
    _current_date_str,
    _provider_trace_for_history,
)


# ---------- markdown_to_html ----------


def test_extracted_helpers_keep_llm_client_compatibility():
    assert llm_client.markdown_to_html is reply_formatting.markdown_to_html
    assert llm_client.strip_markdown is reply_formatting.strip_markdown
    assert llm_client._clean_reply is reply_formatting.clean_reply
    assert llm_client._render_history_for_api is history_rendering.render_history_for_api
    assert llm_client._build_sid_map is history_rendering.build_sid_map
    assert llm_client._renumber_sids is history_rendering.renumber_sids
    assert llm_client._extract_plain_text is history_rendering.extract_plain_text


def test_system_prompt_prefix_excludes_the_calendar_date():
    prefix = _build_system_prompt()
    assert "Today is " not in prefix
    assert "Times of Day" not in prefix
    assert "CURRENT TIME" not in prefix


def test_payload_prefix_puts_date_in_a_second_system_message():
    messages = _build_payload_prefix()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "system"
    assert messages[0]["content"] == _build_system_prompt()
    date_line = messages[1]["content"]
    assert date_line == _current_date_str()
    assert date_line.startswith("Today is ")
    assert "Times of Day" not in date_line
    assert " year." not in date_line


def test_chat_headers_send_openrouter_session_affinity():
    regular = bot_config.chat_api_headers()
    pinned = bot_config.chat_api_headers(session_id="berangaria-abc")

    assert "x-session-id" not in regular
    assert pinned["x-session-id"] == "berangaria-abc"
    assert pinned["HTTP-Referer"] == bot_config.OPENROUTER_HTTP_REFERER
    assert pinned["X-Title"] == bot_config.OPENROUTER_APP_TITLE
    assert regular["HTTP-Referer"] == bot_config.OPENROUTER_HTTP_REFERER
    assert "X-OpenRouter-Metadata" not in pinned
    assert "x-grok-conv-id" not in pinned


def test_apply_chat_gateway_pins_meta_with_required_parameters():
    payload = bot_config.apply_chat_gateway(
        {"model": MODEL, "messages": []},
        session_id="berangaria-abc",
    )
    assert payload["session_id"] == "berangaria-abc"
    assert payload["provider"] == {
        "only": ["meta"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert "service_tier" not in payload


def test_apply_chat_gateway_keeps_provider_pin_without_session_id():
    payload = bot_config.apply_chat_gateway({"model": MODEL, "messages": []}, session_id=None)
    assert "session_id" not in payload
    assert payload["provider"] == {
        "only": ["meta"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert "service_tier" not in payload


def test_markdown_html_escapes_special_chars():
    assert markdown_to_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_markdown_html_bold_italic_code():
    assert markdown_to_html("**bold**") == "<b>bold</b>"
    assert markdown_to_html("*it*") == "<i>it</i>"
    assert markdown_to_html("***bi***") == "<b><i>bi</i></b>"
    assert markdown_to_html("`code`") == "<code>code</code>"
    assert markdown_to_html("~~x~~") == "<s>x</s>"


def test_markdown_html_telegram_quote_spoiler_and_underline():
    text = "> **важное**\n> `точно`\n\n||тайна|| и ++акцент++"

    assert markdown_to_html(text) == (
        "<blockquote><b>важное</b>\n<code>точно</code></blockquote>\n\n"
        "<tg-spoiler>тайна</tg-spoiler> и <u>акцент</u>"
    )


def test_markdown_html_does_not_format_quote_markers_inside_fenced_code():
    text = "```python\n> literal\n**also literal**\n```"

    assert markdown_to_html(text) == (
        '<pre><code class="language-python">&gt; literal\n'
        "**also literal**\n</code></pre>"
    )


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
    assert strip_markdown("> quote\n||secret|| ++under++") == "quote\nsecret under"


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


def test_render_prefers_exact_provider_message_over_telegram_text():
    provider_messages = [{
        "role": "assistant",
        "content": "Ответ с точкой.",
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
    }]
    hist = [{
        "role": "assistant",
        "content": "Ответ с точкой",
        "provider_messages": provider_messages,
    }]

    rendered = _render_history_for_api(hist)

    assert rendered == provider_messages
    assert rendered is not provider_messages
    assert rendered[0] is not provider_messages[0]


def test_provider_trace_keeps_tool_call_and_adds_terminal_result():
    tool_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "reply_to_message",
                "arguments": '{"id":1,"text":"Ответ."}',
            },
        }],
    }
    payload = [
        {"role": "system", "content": "static"},
        {"role": "user", "content": "question"},
        tool_message,
    ]

    trace = _provider_trace_for_history(
        payload,
        2,
        terminal_tool_result="Ответ доставлен в Telegram. Ход завершён.",
    )

    assert trace == [
        tool_message,
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Ответ доставлен в Telegram. Ход завершён.",
        },
    ]
    assert trace[0] is not tool_message


def test_render_plain_assistant_preserves_structured_reasoning():
    reasoning_details = [{
        "type": "reasoning.encrypted",
        "data": "opaque-state",
        "id": "reasoning-1",
        "format": "xai-responses-v1",
        "index": 0,
    }]
    hist = [{
        "role": "assistant",
        "content": "ответ",
        "reasoning_details": reasoning_details,
        "reasoning_content": "flattened duplicate",
    }]

    rendered = _render_history_for_api(hist)

    assert rendered == [{
        "role": "assistant",
        "content": "ответ",
        "reasoning_details": reasoning_details,
    }]
    assert rendered[0]["reasoning_details"] is not reasoning_details


def test_render_plain_assistant_preserves_legacy_reasoning_content():
    hist = [{
        "role": "assistant",
        "content": "ответ",
        "reasoning_content": "opaque legacy state",
    }]
    assert _render_history_for_api(hist) == [{
        "role": "assistant",
        "content": "ответ",
        "reasoning_content": "opaque legacy state",
    }]


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


def test_successful_summary_returns_new_list_and_pins_meta(monkeypatch):
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

    result = asyncio.run(llm_client.summarize_history(history, key="private_1"))

    assert result is not history
    assert history == before
    assert result[0]["role"] == "user"
    assert result[0]["content"].startswith("[Previous conversation summary:")
    assert result[0]["provider_sent"] is False
    assert "RTX 5070 Ti" in result[0]["content"]
    assert len(result) == llm_client.SUMMARY_INTERVAL + 1
    assert captured["url"] == llm_client.CHAT_API_URL
    assert captured["headers"]["Authorization"] == "Bearer test-openrouter-key"
    assert captured["headers"]["X-Title"] == "Berangaria"
    assert captured["headers"]["x-session-id"] == llm_client._chat_session_id(
        "private_1"
    )
    assert "x-grok-conv-id" not in captured["headers"]
    assert captured["payload"]["model"] == llm_client.MODEL
    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert captured["payload"]["temperature"] == 0.3
    assert "service_tier" not in captured["payload"]
    assert "thinking" not in captured["payload"]
    assert "reasoning_effort" not in captured["payload"]
    assert captured["payload"]["max_tokens"] == 8192
    assert "top_k" not in captured["payload"]
    assert "top_p" not in captured["payload"]
    assert "min_p" not in captured["payload"]
    assert captured["timeout"] == 120.0
    assert captured["payload"]["provider"] == {
        "only": ["meta"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert captured["payload"]["session_id"] == llm_client._chat_session_id("private_1")


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


def test_reasoning_tokens_prefers_completion_details():
    assert llm_diagnostics.reasoning_tokens({
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 12},
    }) == 12


def test_reasoning_tokens_uses_total_gap_when_details_missing():
    # xAI counted thinking in total but not in completion_tokens.
    assert llm_diagnostics.reasoning_tokens({
        "prompt_tokens": 15912,
        "completion_tokens": 28,
        "total_tokens": 17941,
    }) == 2001


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


def test_usage_log_identifies_cache_write_and_provider_cost(caplog):
    caplog.set_level(logging.INFO, logger="berangaria.chat.llm_diagnostics")
    usage = {
        "prompt_tokens": 15906,
        "completion_tokens": 106,
        "total_tokens": 16012,
        "prompt_tokens_details": {
            "cached_tokens": 15799,
            "cache_write_tokens": 104,
        },
        "completion_tokens_details": {"reasoning_tokens": 56},
        "cost": 0.002243,
    }
    chat_tokens = {}

    cost = llm_diagnostics.record_usage(
        usage,
        key="private_1",
        chat_tokens=chat_tokens,
        estimate_request_cost=llm_client._estimate_request_cost,
    )

    assert cost == 0.002243
    assert chat_tokens == {"private_1": 16012}
    assert "кэш-чтение=[cyan]15799[/], кэш-запись=[cyan]104[/]" in caplog.text
    assert "источник=usage.cost" in caplog.text


def test_usage_log_labels_fallback_estimate_for_invalid_provider_cost(caplog):
    caplog.set_level(logging.INFO, logger="berangaria.chat.llm_diagnostics")
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "cost": "invalid",
    }

    cost = llm_diagnostics.record_usage(
        usage,
        key="private_1",
        chat_tokens={},
        estimate_request_cost=llm_client._estimate_request_cost,
    )

    assert cost > 0
    assert "источник=локальная оценка" in caplog.text


def test_shipped_chat_model_is_muse_spark_with_low_reasoning():
    assert MODEL == "meta/muse-spark-1.3-contributor"
    assert "openrouter.ai" in CHAT_API_URL
    assert GENERATION_PARAMS.get("temperature") == 1.0
    assert GENERATION_PARAMS.get("reasoning") == {"effort": "low"}
    assert bot_config.CHAT_PROVIDER == "meta"
    assert llm_client.PRICE_PROMPT_CACHE_MISS == 0.10
    assert llm_client.PRICE_PROMPT_CACHE_HIT == 0.002
    assert llm_client.PRICE_PROMPT_CACHE_WRITE == 0.10
    assert llm_client.PRICE_COMPLETION == 0.20
    assert "top_p" not in GENERATION_PARAMS
    assert "top_k" not in GENERATION_PARAMS
    assert "min_p" not in GENERATION_PARAMS


def test_chat_session_id_is_stable_opaque_and_scope_specific():
    first = llm_client._chat_session_id("group_-100")
    assert first == llm_client._chat_session_id("group_-100")
    assert first != llm_client._chat_session_id("private_100")
    assert "group_-100" not in first
    assert len(first) <= 256
