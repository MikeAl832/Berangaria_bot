import asyncio
from types import SimpleNamespace

import pytest

from berangaria.user_bridge import runtime


def test_start_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(runtime, "USER_BRIDGE_ENABLED", False)
    result = asyncio.run(runtime.start_user_bridge(SimpleNamespace(bot=object())))
    assert result is None


def test_start_missing_credentials_is_noop(monkeypatch):
    monkeypatch.setattr(runtime, "USER_BRIDGE_ENABLED", True)
    monkeypatch.setattr(runtime, "TELEGRAM_API_ID", 0)
    monkeypatch.setattr(runtime, "TELEGRAM_API_HASH", "")
    monkeypatch.setattr(runtime, "USER_BRIDGE_SESSION", "")
    result = asyncio.run(runtime.start_user_bridge(SimpleNamespace(bot=object())))
    assert result is None


def test_start_empty_allowlist_is_noop(monkeypatch):
    monkeypatch.setattr(runtime, "USER_BRIDGE_ENABLED", True)
    monkeypatch.setattr(runtime, "TELEGRAM_API_ID", 123)
    monkeypatch.setattr(runtime, "TELEGRAM_API_HASH", "hash")
    monkeypatch.setattr(runtime, "USER_BRIDGE_SESSION", "session")
    monkeypatch.setattr(runtime, "USER_BRIDGE_CHAT_IDS", [])
    monkeypatch.setattr(runtime, "ALLOWED_GROUPS", [])
    result = asyncio.run(runtime.start_user_bridge(SimpleNamespace(bot=object())))
    assert result is None


def test_supervisor_survives_inner_failure(monkeypatch):
    """A failing client run schedules reconnect instead of killing the task hard."""
    monkeypatch.setattr(runtime, "USER_BRIDGE_RECONNECT_SECONDS", 0.01)

    calls = {"n": 0}
    stop = asyncio.Event()

    async def _boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            stop.set()
        raise RuntimeError("simulated telethon death")

    monkeypatch.setattr(runtime, "_run_client_once", _boom)

    async def _run():
        task = asyncio.create_task(
            runtime._bridge_supervisor(
                bot=object(),
                allowed_chat_ids=(-1001,),
                stop_event=stop,
            )
        )
        await asyncio.wait_for(stop.wait(), timeout=2.0)
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert calls["n"] >= 2

    asyncio.run(_run())


class _UninitializedExtBot:
    """Mimics PTB ExtBot before ``initialize``: ``id`` raises RuntimeError."""

    @property
    def id(self):
        raise RuntimeError(
            "ExtBot is not properly initialized. Call `ExtBot.initialize` "
            "before accessing this property."
        )

    async def get_me(self):
        return SimpleNamespace(id=9001)


def test_getattr_id_does_not_catch_ptb_runtime_error():
    """Document why getattr(bot, 'id', None) is the wrong probe."""
    with pytest.raises(RuntimeError, match="not properly initialized"):
        getattr(_UninitializedExtBot(), "id", None)


def test_resolve_bot_id_falls_back_to_get_me_when_property_uninitialized():
    bot = _UninitializedExtBot()
    assert asyncio.run(runtime._resolve_our_bot_id(bot)) == 9001


def test_resolve_bot_id_uses_cached_int_without_get_me():
    class ReadyBot:
        id = 42
        get_me_calls = 0

        async def get_me(self):
            type(self).get_me_calls += 1
            return SimpleNamespace(id=99)

    bot = ReadyBot()
    assert asyncio.run(runtime._resolve_our_bot_id(bot)) == 42
    assert ReadyBot.get_me_calls == 0


def test_resolve_bot_id_returns_none_when_both_probes_fail():
    class DeadBot:
        @property
        def id(self):
            raise RuntimeError("not initialized")

        async def get_me(self):
            raise RuntimeError("network down")

    assert asyncio.run(runtime._resolve_our_bot_id(DeadBot())) is None
