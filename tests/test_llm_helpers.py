"""
Характеризационные тесты чистых хелперов llm_client.

Фиксируют ТЕКУЩЕЕ поведение форматирования/очистки/рендера истории,
чтобы последующий рефакторинг send_llm_request нельзя было провести
с молчаливым изменением логики.
"""
from llm_client import (
    markdown_to_html,
    strip_markdown,
    _clean_reply,
    _render_history_for_api,
    _build_sid_map,
    _renumber_sids,
    _extract_plain_text,
    _format_memory_block,
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


def test_clean_reply_removes_emoji():
    assert _clean_reply("текст 😀🔥") == "текст"


def test_clean_reply_silence_word():
    assert _clean_reply("молчу") == ""
    assert _clean_reply("(молчит)") == ""


def test_clean_reply_keeps_normal_text():
    assert _clean_reply("да, согласна") == "да, согласна"


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


def test_render_user_without_sid_unchanged():
    hist = [{"role": "user", "content": "hi"}]
    assert _render_history_for_api(hist) == [{"role": "user", "content": "hi"}]


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


# ---------- _format_memory_block ----------

def test_format_memory_empty():
    assert _format_memory_block({}) == ""
    assert _format_memory_block({"results": []}) == ""


def test_format_memory_filters_below_min_score():
    # MEMORY_MIN_SCORE == 0.2 в конфиге
    res = {"results": [
        {"memory": "хороший факт", "score": 0.9},
        {"memory": "слабый факт", "score": 0.05},
    ]}
    out = _format_memory_block(res)
    assert "хороший факт" in out
    assert "слабый факт" not in out


def test_format_memory_formats_as_bullet():
    res = {"results": [{"memory": "факт", "score": 0.9}]}
    assert _format_memory_block(res) == "- факт"
