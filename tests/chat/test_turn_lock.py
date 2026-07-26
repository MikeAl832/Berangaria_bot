"""Регрессии на инварианты сериализации хода (AGENTS.md, «Concurrency and history invariants»).

До появления этого файла инварианты не были защищены ничем: удаление
`async with get_turn_lock(key)` из handlers.clear оставляло весь набор тестов
зелёным. Тесты намеренно проверяют поведение под конкуренцией, а не факт
наличия строки в исходнике.
"""
import asyncio

import pytest

from berangaria.chat import handlers
from berangaria.core import state


def test_turn_lock_is_stable_per_key():
    first = state.get_turn_lock("private_1")
    assert state.get_turn_lock("private_1") is first
    assert state.get_turn_lock("private_2") is not first


def test_turn_lock_serializes_two_turns_of_one_chat():
    """Два хода одного чата не могут перекрываться."""
    order = []

    async def turn(name, hold):
        async with state.get_turn_lock("private_1"):
            order.append(f"{name}:enter")
            await asyncio.sleep(hold)
            order.append(f"{name}:exit")

    async def run():
        # Первый успевает войти раньше и держит лок дольше, чем спит второй.
        await asyncio.gather(turn("A", 0.05), turn("B", 0))

    asyncio.run(run())

    # Ни одного перекрытия: каждый вход закрыт своим выходом.
    assert order in (
        ["A:enter", "A:exit", "B:enter", "B:exit"],
        ["B:enter", "B:exit", "A:enter", "A:exit"],
    )


def test_turn_locks_of_different_chats_do_not_block_each_other():
    """Ход в одном чате не должен задерживать другой чат."""
    events = []

    async def slow_chat():
        async with state.get_turn_lock("group_-100"):
            events.append("slow:enter")
            await asyncio.sleep(0.05)
            events.append("slow:exit")

    async def fast_chat():
        await asyncio.sleep(0)
        async with state.get_turn_lock("private_1"):
            events.append("fast:done")

    async def run():
        await asyncio.gather(slow_chat(), fast_chat())

    asyncio.run(run())

    # Быстрый чат прошёл, не дожидаясь медленного.
    assert events.index("fast:done") < events.index("slow:exit")


def test_history_lock_is_separate_from_turn_lock():
    assert state.get_turn_lock("private_1") is not state.get_history_lock("private_1")


def test_canonical_lock_order_completes():
    """Канонический порядок turn -> history проходит без блокировок."""

    async def canonical():
        # Вложенность намеренная, а не небрежность: тест про ПОРЯДОК взятия
        # локов, и два отдельных `async with` показывают его буквально.
        async with state.get_turn_lock("private_1"):  # noqa: SIM117
            async with state.get_history_lock("private_1"):
                return "ok"

    async def run():
        return await asyncio.wait_for(canonical(), timeout=1.0)

    assert asyncio.run(run()) == "ok"


def test_reversed_lock_order_deadlocks_two_tasks():
    """Фиксируем, ПОЧЕМУ порядок важен: обратный порядок реально виснет.

    Тест защищает от «безобидной» перестановки: если кто-то поменяет порядок
    в одном из восьми вложенных мест, это перестанет быть теорией.
    """

    async def canonical():
        async with state.get_turn_lock("private_1"):
            await asyncio.sleep(0.02)
            async with state.get_history_lock("private_1"):
                return "canonical"

    async def reversed_order():
        async with state.get_history_lock("private_1"):
            await asyncio.sleep(0.02)
            async with state.get_turn_lock("private_1"):
                return "reversed"

    async def run():
        return await asyncio.wait_for(
            asyncio.gather(canonical(), reversed_order()), timeout=0.5
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())


def test_clear_command_holds_the_turn_lock(monkeypatch):
    """`/clear` обязан брать turn-lock: он мутирует историю чата.

    Если убрать `async with get_turn_lock(key)` из handlers.clear, команда
    отработает, пока идёт LLM-ход, и сотрёт историю в середине транзакции.
    """
    from types import SimpleNamespace

    replies = []
    entered = asyncio.Event()

    monkeypatch.setattr(handlers, "ADMIN_MODE", False)
    monkeypatch.setattr(handlers, "_check_access_permissions", lambda *args: True)
    monkeypatch.setattr(state, "delete_history", lambda key: True)

    async def reply_text(text, **kwargs):
        replies.append(text)

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1, type="private"),
        effective_user=SimpleNamespace(id=42, first_name="Миша"),
        message=SimpleNamespace(message_id=1, reply_text=reply_text),
    )

    async def run():
        # get_history_key(chat_id, is_private=True, user_id) -> f"private_{user_id}"
        key = state.get_history_key(1, True, 42)
        handlers.histories[key] = [{"role": "user", "content": "привет"}]

        async def holder():
            async with state.get_turn_lock(key):
                entered.set()
                await asyncio.sleep(0.05)

        async def clear_call():
            await entered.wait()
            await handlers.clear(update, SimpleNamespace())

        holder_task = asyncio.create_task(holder())
        clear_task = asyncio.create_task(clear_call())
        await asyncio.sleep(0.01)
        # Пока turn-lock занят, /clear обязан ждать и не отвечать.
        clear_finished_early = clear_task.done()
        await asyncio.gather(holder_task, clear_task)
        return clear_finished_early

    async def coro_wrapper():
        return await asyncio.wait_for(run(), timeout=2.0)

    finished_early = asyncio.run(coro_wrapper())

    assert finished_early is False, "/clear не дождался turn-lock"
    assert replies, "/clear должен подтвердить очистку после получения лока"
