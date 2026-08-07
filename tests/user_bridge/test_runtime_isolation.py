import asyncio
from types import SimpleNamespace

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
