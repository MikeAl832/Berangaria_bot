"""
Юнит-тесты вынесенных обработчиков инструментов (tool_handlers.py).

Раньше эта логика жила замыканиями внутри send_llm_request и была непокрыта.
Telegram/сеть подменяются лёгкими фейками, async гоняем через asyncio.run.
"""
import asyncio

import tool_handlers
from tool_handlers import (
    ToolTurn,
    handle_reply,
    handle_react,
    handle_find_stickers,
    handle_send_sticker,
    dispatch_tool_call,
)


# ---------- фейки ----------

class FakeChat:
    def __init__(self):
        self.actions = []

    async def send_action(self, action):
        self.actions.append(action)


class FakeStatusMsg:
    def __init__(self, mid=999):
        self.message_id = mid
        self.edits = []
        self.deleted = False

    async def edit_text(self, text, **kw):
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


# ---------- handle_react ----------

def test_handle_react_valid_emoji_sets_state_and_calls_api():
    turn = ToolTurn()
    upd, ctx, payload = FakeUpdate(mid=1), FakeContext(), []
    asyncio.run(handle_react(turn, payload, upd, ctx, TC, {"emoji": "🔥"}, {}, []))
    assert turn.reacted is True
    assert turn.reactions_made == [{"emoji": "🔥", "on": None}]
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
