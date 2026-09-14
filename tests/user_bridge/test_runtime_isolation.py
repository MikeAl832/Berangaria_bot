import asyncio
from types import SimpleNamespace

import pytest
import telethon
from telethon.sessions import StringSession

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


class _Client:
    def __init__(self, session, api_id, api_hash, **options):
        self.session = session
        self.options = options
        self.disconnected = asyncio.get_running_loop().create_future()
        self.ready = asyncio.Event()
        self.closed = False

    def on(self, event):
        return lambda handler: handler

    async def connect(self):
        pass

    async def is_user_authorized(self):
        return True

    async def get_me(self):
        self.ready.set()
        return SimpleNamespace(id=123, username="test")

    async def disconnect(self):
        self.closed = True
        if not self.disconnected.done():
            self.disconnected.set_result(None)


def _install_client(monkeypatch, client_type=_Client):
    clients = []

    def factory(*args, **kwargs):
        client = client_type(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(telethon, "TelegramClient", factory)
    monkeypatch.setattr(runtime, "USER_BRIDGE_SESSION", "")
    monkeypatch.setattr(runtime, "USER_BRIDGE_PORT", 0)
    return clients


@pytest.mark.parametrize("cancel", [False, True])
def test_client_closes_and_leaves_no_stop_waiter(monkeypatch, cancel):
    clients = _install_client(monkeypatch)

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(runtime._run_client_once(
            SimpleNamespace(id=42), allowed_chat_ids=(-1001,), stop_event=stop
        ))
        await asyncio.sleep(0)
        await clients[0].ready.wait()
        assert clients[0].options["auto_reconnect"] is False
        assert clients[0].options["connection_retries"] == 0
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            stop.set()
            await asyncio.wait_for(task, 1)
        assert clients[0].closed
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["connect", "is_user_authorized", "get_me"])
def test_startup_deadline_closes_stalled_client(monkeypatch, stage):
    class StalledClient(_Client):
        pass

    async def stall(self):
        await asyncio.Event().wait()

    setattr(StalledClient, stage, stall)
    clients = _install_client(monkeypatch, StalledClient)
    monkeypatch.setattr(runtime, "_CLIENT_START_TIMEOUT", 0.01)

    async def run():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(runtime._run_client_once(
                SimpleNamespace(id=42), allowed_chat_ids=(-1001,), stop_event=asyncio.Event()
            ), 1)
        assert clients[0].closed
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())


def test_disconnect_error_reaches_supervisor_after_cleanup(monkeypatch):
    class BrokenClient(_Client):
        async def get_me(self):
            self.disconnected.set_exception(ConnectionError("network lost"))
            return await super().get_me()

    clients = _install_client(monkeypatch, BrokenClient)

    async def run():
        with pytest.raises(ConnectionError, match="network lost"):
            await runtime._run_client_once(
                SimpleNamespace(id=42), allowed_chat_ids=(-1001,), stop_event=asyncio.Event()
            )
        assert clients[0].closed
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())


def test_failed_connect_consumes_disconnection_future(monkeypatch):
    class BrokenClient(_Client):
        async def connect(self):
            self.disconnected.set_exception(ConnectionError("HTTP gateway"))
            raise ConnectionError("HTTP gateway")

    clients = _install_client(monkeypatch, BrokenClient)

    async def run():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: unhandled.append(ctx))
        with pytest.raises(ConnectionError, match="HTTP gateway"):
            await runtime._run_client_once(
                SimpleNamespace(id=42), allowed_chat_ids=(-1001,), stop_event=asyncio.Event()
            )
        assert clients[0].closed
        clients.clear()
        import gc
        gc.collect()
        assert not unhandled

    asyncio.run(run())


def test_port_override_preserves_session_dc_and_auth(monkeypatch):
    from telethon.crypto import AuthKey

    session = StringSession()
    session.set_dc(2, "149.154.167.51", 443)
    session.auth_key = AuthKey(bytes(range(256)))
    serialized = session.save()
    clients = _install_client(monkeypatch)
    monkeypatch.setattr(runtime, "USER_BRIDGE_SESSION", serialized)
    monkeypatch.setattr(runtime, "USER_BRIDGE_PORT", 5222)

    async def run():
        stop = asyncio.Event()
        stop.set()
        await runtime._run_client_once(
            SimpleNamespace(id=42), allowed_chat_ids=(-1001,), stop_event=stop
        )
        configured = clients[0].session
        assert (configured.dc_id, configured.server_address, configured.port) == (
            2, "149.154.167.51", 5222
        )
        assert configured.auth_key.key == session.auth_key.key
        assert StringSession(serialized).port == 443

    asyncio.run(run())
