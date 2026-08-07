import asyncio
from types import SimpleNamespace

from berangaria.chat import handlers
from berangaria.core import state


class _Bot:
    id = 9001
    username = "berangaria_bot"
    first_name = "Ber"

    async def get_me(self):
        return SimpleNamespace(id=self.id, username=self.username, first_name=self.first_name)

    async def send_chat_action(self, **kwargs):
        return None

    async def send_message(self, **kwargs):
        return SimpleNamespace(message_id=1)


class _Chat:
    def __init__(self, chat_id=-1002263830880):
        self.id = chat_id
        self.type = "supergroup"
        self.title = "test"

    async def send_action(self, action="typing"):
        return None


class _User:
    def __init__(self, user_id=555, name="OtherBot"):
        self.id = user_id
        self.first_name = name
        self.username = "other_bot"
        self.is_bot = True


class _Message:
    def __init__(self, chat, user, mid=42, text="hi Ber"):
        self.chat = chat
        self.from_user = user
        self.message_id = mid
        self.text = text
        self.caption = None
        self.date = None
        self.message_thread_id = None
        self.reply_to_message = None

    async def reply_text(self, text, **kwargs):
        return SimpleNamespace(message_id=2)


class _Update:
    def __init__(self, chat_id=-1002263830880, text="hi Ber"):
        self.effective_chat = _Chat(chat_id)
        self.effective_user = _User()
        self.message = _Message(self.effective_chat, self.effective_user, text=text)


def test_bridge_queue_no_memory_and_bot_author(monkeypatch, isolated_db):
    """Bridge path must not enqueue memory and must tag history as Bot."""
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-1002263830880])
    monkeypatch.setattr(handlers, "MESSAGE_DEBOUNCE_SECONDS", 0)

    enqueued = []

    def _fake_enqueue(**kwargs):
        enqueued.append(kwargs)
        return 123

    monkeypatch.setattr(handlers, "enqueue_memory_source", _fake_enqueue)

    llm_calls = []

    async def _fake_llm(*args, **kwargs):
        llm_calls.append(args)

    monkeypatch.setattr(handlers, "send_llm_request", _fake_llm)
    monkeypatch.setattr(handlers, "should_reply_randomly", lambda chat_id: False)

    update = _Update(text="hey Ber whats up")
    context = SimpleNamespace(bot=_Bot())

    async def _run():
        await handlers.queue_bridge_bot_message(
            update,
            context,
            text="hey Ber whats up",
        )
        # Debounce 0 still schedules a task; let it finish.
        await asyncio.sleep(0.05)
        # Drain any remaining buffer tasks.
        for data in list(state.message_buffer.values()):
            task = data.get("task")
            if task is not None:
                await task

    asyncio.run(_run())

    assert enqueued == []

    key = state.get_history_key(-1002263830880, False)
    history = state.histories.get(key) or []
    assert history, "bridge message should land in group history"
    content = history[-1]["content"]
    assert content.startswith("[Bot: OtherBot]")
    assert "[Message: hey Ber whats up]" in content
    assert llm_calls, "mention of Ber should trigger LLM"


def test_bridge_queue_refuses_non_allowed_group(monkeypatch, isolated_db):
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-1002263830880])
    monkeypatch.setattr(handlers, "enqueue_memory_source", lambda **kw: 1)

    update = _Update(chat_id=-999)
    context = SimpleNamespace(bot=_Bot())

    asyncio.run(
        handlers.queue_bridge_bot_message(update, context, text="hello Ber")
    )

    assert state.histories == {} or not any(
        state.histories.get(k) for k in state.histories
    )


def test_bridge_queue_skips_memory_on_text_only_bot_chat(monkeypatch, isolated_db):
    """Even with text that looks like a personal fact, no memory source."""
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-1002263830880])
    monkeypatch.setattr(handlers, "MESSAGE_DEBOUNCE_SECONDS", 0)
    monkeypatch.setattr(handlers, "should_reply_randomly", lambda chat_id: False)

    # No mention / no random → still history, no LLM required.
    update = _Update(text="I live in Paris forever")
    # Avoid name mention
    update.message.text = "I live in Paris forever"
    context = SimpleNamespace(bot=_Bot())
    context.bot.first_name = "ZZZunique"
    context.bot.username = "zzz_unique_bot"

    enqueued = []
    monkeypatch.setattr(
        handlers, "enqueue_memory_source", lambda **kw: enqueued.append(kw) or 1
    )

    async def _run():
        await handlers.queue_bridge_bot_message(
            update,
            context,
            text="I live in Paris forever",
        )
        await asyncio.sleep(0.05)
        for data in list(state.message_buffer.values()):
            task = data.get("task")
            if task is not None:
                await task

    asyncio.run(_run())
    assert enqueued == []
    key = state.get_history_key(-1002263830880, False)
    history = state.histories.get(key) or []
    assert history
    assert "[Bot:" in history[-1]["content"]
