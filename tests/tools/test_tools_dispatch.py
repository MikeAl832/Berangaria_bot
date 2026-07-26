"""
Юнит-тесты вынесенных обработчиков инструментов (berangaria/tools/dispatch.py).

Раньше эта логика жила замыканиями внутри send_llm_request и была непокрыта.
Telegram/сеть подменяются лёгкими фейками, async гоняем через asyncio.run.
"""
import asyncio

import pytest

from berangaria.tools import dispatch as tool_handlers
from berangaria.tools.dispatch import (
    ToolTurn,
    handle_reply,
    handle_react,
    handle_find_stickers,
    handle_send_sticker,
    handle_web_search,
    dispatch_tool_call,
)
from berangaria.tools.web import RATE_LIMIT_PREFIX


# ---------- фейки ----------

class FakeChat:
    def __init__(self):
        self.actions = []

    async def send_action(self, action):
        self.actions.append(action)


class FakeStatusMsg:
    """A status banner that behaves like Telegram: a no-op edit is an error.

    Without this the fake would silently swallow "message is not modified", and
    the tests would see neither the wasted API call nor the false WARNING.
    """

    def __init__(self, mid=999):
        self.message_id = mid
        self.edits = []
        self.rejected_edits = []
        self.deleted = False
        self._text = None

    async def edit_text(self, text, **kw):
        if text == self._text:
            self.rejected_edits.append(text)
            raise RuntimeError("Message is not modified")
        self._text = text
        self.edits.append(text)

    async def delete(self):
        self.deleted = True


class FakeMessage:
    def __init__(self, mid=1, thread_id=None):
        self.message_id = mid
        self.chat = FakeChat()
        self.message_thread_id = thread_id
        self.reply_msg = FakeStatusMsg()

    async def reply_text(self, text, **kw):
        return self.reply_msg


class FakeEffChat:
    id = 555


class FakeUpdate:
    def __init__(self, mid=1):
        self.message = FakeMessage(mid=mid)
        self.effective_chat = FakeEffChat()


class FakeBot:
    def __init__(self):
        self.reactions = []
        self.stickers = []

    async def set_message_reaction(self, chat_id, message_id, reaction):
        self.reactions.append((chat_id, message_id, reaction))

    async def send_sticker(self, **kw):
        self.stickers.append(kw)


class FakeContext:
    def __init__(self):
        self.bot = FakeBot()


TC = {"id": "tc1", "function": {"name": "x", "arguments": "{}"}}


# ---------- handle_reply (терминальный, синхронный) ----------

def test_handle_reply_known_sid():
    turn = ToolTurn()
    handle_reply(turn, FakeUpdate(mid=1), {"id": 2, "text": "хай"}, {2: 42})
    assert turn.pending_reply == (42, "хай", 2)


def test_handle_reply_unknown_sid_falls_back_to_current_message():
    turn = ToolTurn()
    handle_reply(turn, FakeUpdate(mid=7), {"id": 99, "text": "x"}, {2: 42})
    assert turn.pending_reply == (7, "x", 99)


@pytest.mark.parametrize("bad_text", [42, 3.5, None, ["а", "б"], {"t": "x"}, True])
def test_handle_reply_coerces_non_string_text_to_empty(bad_text):
    # Аргументы инструмента — недоверенный JSON от модели. Нестроковый `text`
    # должен дать пустой ответ, а не улететь дальше и уронить _clean_reply.
    turn = ToolTurn()
    handle_reply(turn, FakeUpdate(mid=7), {"id": 2, "text": bad_text}, {2: 42})
    assert turn.pending_reply == (42, "", 2)


def test_handle_reply_missing_text_key():
    turn = ToolTurn()
    handle_reply(turn, FakeUpdate(mid=7), {"id": 2}, {2: 42})
    assert turn.pending_reply == (42, "", 2)


# ---------- handle_react ----------

def test_handle_react_valid_emoji_sets_state_and_calls_api():
    turn = ToolTurn()
    upd, ctx, payload = FakeUpdate(mid=1), FakeContext(), []
    asyncio.run(handle_react(turn, payload, upd, ctx, TC, {"emoji": "🔥"}, {}, []))
    assert turn.reacted is True
    assert turn.reactions_made[0]["emoji"] == "🔥"
    assert turn.reactions_made[0]["on_mid"] == 1
    assert ctx.bot.reactions == [(555, 1, "🔥")]
    assert payload[-1]["role"] == "tool"
    assert "поставлена" in payload[-1]["content"]


def test_handle_react_disallowed_emoji_no_api_call():
    turn = ToolTurn()
    upd, ctx, payload = FakeUpdate(), FakeContext(), []
    asyncio.run(handle_react(turn, payload, upd, ctx, TC, {"emoji": "🍕"}, {}, []))
    assert turn.reacted is False
    assert ctx.bot.reactions == []
    assert "не разрешён" in payload[-1]["content"]


def test_handle_react_strips_fe0f_variation_selector():
    turn = ToolTurn()
    upd, ctx, payload = FakeUpdate(mid=1), FakeContext(), []
    # ❤️ с FE0F должен пройти как каноничный ❤
    asyncio.run(handle_react(turn, payload, upd, ctx, TC, {"emoji": "❤️"}, {}, []))
    assert ctx.bot.reactions == [(555, 1, "❤")]


def test_handle_react_rejects_duplicate_on_same_mid():
    turn = ToolTurn()
    hist = [{
        "role": "assistant",
        "content": "",
        "reactions": [{"emoji": "🤡", "on_mid": 99, "on_sid": 139, "on": "шутка"}],
    }]
    upd, ctx, payload = FakeUpdate(mid=1), FakeContext(), []
    asyncio.run(handle_react(
        turn, payload, upd, ctx, TC, {"emoji": "🔥", "id": 139}, {139: 99}, hist
    ))
    assert ctx.bot.reactions == []  # Telegram не трогали
    assert "УЖЕ" in payload[-1]["content"]
    assert "🤡" in payload[-1]["content"]
    assert turn.reacted is True  # прошлый факт — молчать текстом можно


# ---------- handle_find_stickers ----------

def test_handle_find_stickers_numbers_candidates(monkeypatch):
    monkeypatch.setattr(tool_handlers, "STICKER_ENABLED", True)
    monkeypatch.setattr(tool_handlers, "search_stickers", lambda q, c=6: [
        {"file_id": "f1", "description": "ржу", "emotion": "joy", "keywords": ["смех"]},
        {"file_id": "f2", "description": "грусть", "emotion": "sad", "keywords": []},
    ])
    turn = ToolTurn()
    payload = []
    asyncio.run(handle_find_stickers(turn, payload, FakeUpdate(), TC, {"query": "ржу"}))
    assert turn.sticker_seq == 2
    assert turn.sticker_candidates[1]["file_id"] == "f1"
    assert turn.sticker_candidates[2]["file_id"] == "f2"
    content = payload[-1]["content"]
    assert "#1" in content and "#2" in content
    assert "теги: смех" in content


def test_handle_find_stickers_empty_query(monkeypatch):
    monkeypatch.setattr(tool_handlers, "STICKER_ENABLED", True)
    turn = ToolTurn()
    payload = []
    asyncio.run(handle_find_stickers(turn, payload, FakeUpdate(), TC, {"query": "   "}))
    assert "Пустой запрос" in payload[-1]["content"]
    assert turn.sticker_seq == 0


def test_handle_find_stickers_disabled(monkeypatch):
    monkeypatch.setattr(tool_handlers, "STICKER_ENABLED", False)
    turn = ToolTurn()
    payload = []
    asyncio.run(handle_find_stickers(turn, payload, FakeUpdate(), TC, {"query": "ржу"}))
    assert payload[-1]["content"] == "Стикеры отключены."


def test_handle_find_stickers_respects_per_turn_limit(monkeypatch):
    monkeypatch.setattr(tool_handlers, "STICKER_ENABLED", True)
    monkeypatch.setattr(tool_handlers, "STICKER_FIND_MAX_PER_TURN", 3)
    calls = {"n": 0}

    def fake_search(q, c=6):
        calls["n"] += 1
        return [{"file_id": f"f{calls['n']}", "description": q, "emotion": "x", "keywords": []}]

    monkeypatch.setattr(tool_handlers, "search_stickers", fake_search)
    turn = ToolTurn()
    payload = []
    for i in range(3):
        asyncio.run(handle_find_stickers(turn, payload, FakeUpdate(), TC, {"query": f"q{i}"}))
    assert calls["n"] == 3
    assert turn.find_stickers_calls == 3
    # 4-й вызов — отказ без реального поиска
    asyncio.run(handle_find_stickers(turn, payload, FakeUpdate(), TC, {"query": "ещё"}))
    assert calls["n"] == 3
    assert "Лимит поиска" in payload[-1]["content"]
    assert turn.find_stickers_calls == 3


# ---------- handle_send_sticker ----------

def test_handle_send_sticker_by_id(monkeypatch):
    monkeypatch.setattr(tool_handlers, "STICKER_ENABLED", True)
    turn = ToolTurn()
    turn.sticker_candidates[3] = {"file_id": "fX", "desc": "d", "emotion": "e"}
    ctx, payload = FakeContext(), []
    asyncio.run(handle_send_sticker(turn, payload, FakeUpdate(), ctx, TC, {"id": 3}))
    assert turn.sticker_sent is True
    assert turn.stickers_made == [{"desc": "d", "emotion": "e"}]
    assert ctx.bot.stickers[0]["sticker"] == "fX"
    assert "Стикер отправлен" in payload[-1]["content"]


def test_handle_send_sticker_bad_id(monkeypatch):
    monkeypatch.setattr(tool_handlers, "STICKER_ENABLED", True)
    turn = ToolTurn()  # пустые кандидаты
    ctx, payload = FakeContext(), []
    asyncio.run(handle_send_sticker(turn, payload, FakeUpdate(), ctx, TC, {"id": 5}))
    assert turn.sticker_sent is False
    assert ctx.bot.stickers == []
    assert "Не поняла, какой стикер" in payload[-1]["content"]


# ---------- handle_web_search ----------

def _stub_search(monkeypatch, result="1. что-то\nописание\nhttps://example.com"):
    """Stubs out the network search and counts the real calls."""
    calls = {"n": 0}

    def fake_search(query, max_results=5, timelimit=None, region="ru-ru"):
        calls["n"] += 1
        return result

    monkeypatch.setattr(tool_handlers, "web_search", fake_search)
    return calls


def test_handle_web_search_respects_per_turn_limit(monkeypatch):
    """The prompt promises at most two searches per turn; the DDG limiter is
    process-global, so the ceiling has to live in code, not only in prompt text."""
    monkeypatch.setattr(tool_handlers, "WEB_SEARCH_MAX_PER_TURN", 2)
    calls = _stub_search(monkeypatch)
    turn = ToolTurn()
    payload = []

    for i in range(2):
        asyncio.run(handle_web_search(turn, payload, FakeUpdate(), TC, {"query": f"q{i}"}))
    assert calls["n"] == 2
    assert turn.web_search_calls == 2

    # third call — refused without touching the network
    asyncio.run(handle_web_search(turn, payload, FakeUpdate(), TC, {"query": "ещё"}))
    assert calls["n"] == 2
    assert turn.web_search_calls == 2
    assert "Лимит поисков" in payload[-1]["content"]
    assert payload[-1]["tool_call_id"] == TC["id"]


def test_handle_web_search_rate_limit_is_not_a_miss(monkeypatch):
    """A limiter refusal must not read to the model as "nothing found" and trigger a retry."""
    monkeypatch.setattr(tool_handlers, "WEB_SEARCH_MAX_PER_TURN", 2)
    _stub_search(monkeypatch, result=f"{RATE_LIMIT_PREFIX} (10/мин). Попробуйте позже.")
    turn = ToolTurn()
    payload = []
    asyncio.run(handle_web_search(turn, payload, FakeUpdate(), TC, {"query": "курс евро"}))
    content = payload[-1]["content"]
    assert "Не повторяй поиск в этом ходе" in content
    assert RATE_LIMIT_PREFIX not in content


def test_handle_web_search_status_message_hides_the_query(monkeypatch):
    """The banner sits in the chat next to the answer: showing the query in it
    exposes the mechanics exactly where the prompt requires them hidden."""
    monkeypatch.setattr(tool_handlers, "WEB_SEARCH_MAX_PER_TURN", 2)
    _stub_search(monkeypatch)
    turn = ToolTurn()
    update = FakeUpdate()
    asyncio.run(handle_web_search(turn, [], update, TC, {"query": "сколько лет Илону Маску"}))
    shown = [turn.status_message.message_id] and update.message.reply_msg.edits
    assert all("Илону" not in text for text in shown)
    assert turn.status_message is update.message.reply_msg


def test_two_searches_never_rewrite_the_banner_with_the_same_text(monkeypatch):
    """The banner text no longer contains the query, so two searches in a row ask
    for the same text. Telegram rejects that edit — so do not send it."""
    monkeypatch.setattr(tool_handlers, "WEB_SEARCH_MAX_PER_TURN", 2)
    _stub_search(monkeypatch)
    turn = ToolTurn()
    update = FakeUpdate()
    payload = []

    asyncio.run(handle_web_search(turn, payload, update, TC, {"query": "первый"}))
    asyncio.run(handle_web_search(turn, payload, update, TC, {"query": "второй"}))

    banner = update.message.reply_msg
    assert banner.rejected_edits == []
    assert banner.edits == []  # created once and never touched again


def test_read_url_after_search_does_update_the_banner(monkeypatch):
    """The "do not rewrite with the same text" guard must not stick: a different text still edits."""
    monkeypatch.setattr(tool_handlers, "WEB_SEARCH_MAX_PER_TURN", 2)
    _stub_search(monkeypatch)
    monkeypatch.setattr(tool_handlers, "read_url", lambda url: "текст страницы")
    turn = ToolTurn()
    update = FakeUpdate()
    payload = []

    asyncio.run(handle_web_search(turn, payload, update, TC, {"query": "первый"}))
    asyncio.run(tool_handlers.handle_read_url(turn, payload, update, TC, {"url": "https://e.com"}))

    banner = update.message.reply_msg
    assert banner.rejected_edits == []
    assert banner.edits == ["🔗 Читаю ссылку..."]


def test_handle_web_search_passes_region_through(monkeypatch):
    """The tool schema declares region — the handler must pass it through."""
    monkeypatch.setattr(tool_handlers, "WEB_SEARCH_MAX_PER_TURN", 2)
    seen = {}

    def fake_search(query, max_results=5, timelimit=None, region="ru-ru"):
        seen.update(query=query, region=region, timelimit=timelimit)
        return "ok"

    monkeypatch.setattr(tool_handlers, "web_search", fake_search)
    asyncio.run(handle_web_search(
        ToolTurn(), [], FakeUpdate(), TC,
        {"query": "python release", "region": "wt-wt", "timelimit": "w"},
    ))
    assert seen == {"query": "python release", "region": "wt-wt", "timelimit": "w"}


# ---------- dispatch_tool_call ----------

def test_dispatch_unknown_tool():
    turn = ToolTurn()
    payload = []
    tc = {"id": "t9", "function": {"name": "nope", "arguments": "{}"}}
    asyncio.run(dispatch_tool_call(turn, payload, FakeUpdate(), FakeContext(), tc, {}, []))
    assert payload[-1]["content"] == "Инструмент 'nope' не поддерживается."


def test_dispatch_routes_reply_to_message():
    turn = ToolTurn()
    payload = []
    tc = {"id": "t1", "function": {"name": "reply_to_message",
                                    "arguments": '{"id": 2, "text": "yo"}'}}
    asyncio.run(dispatch_tool_call(turn, payload, FakeUpdate(mid=1), FakeContext(), tc, {2: 40}, []))
    # терминальный инструмент: в payload ничего не пишет, только pending_reply
    assert payload == []
    assert turn.pending_reply == (40, "yo", 2)
