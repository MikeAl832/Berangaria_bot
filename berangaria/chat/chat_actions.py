"""Best-effort Telegram chat-action heartbeat for one selected bot turn."""

import asyncio
import logging
from typing import Any


logger = logging.getLogger(__name__)

CHAT_ACTION_REFRESH_SECONDS = 4.0


class ChatActionHeartbeat:
    """Keep one Telegram chat action alive and allow atomic action switches."""

    def __init__(
        self,
        chat: Any,
        *,
        action: str = "typing",
        message_thread_id: int | None = None,
        interval_seconds: float = CHAT_ACTION_REFRESH_SECONDS,
    ) -> None:
        self._chat = chat
        self._action = action
        self._message_thread_id = message_thread_id
        self._interval_seconds = max(0.001, float(interval_seconds))
        self._send_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    @property
    def action(self) -> str:
        return self._action

    async def __aenter__(self) -> "ChatActionHeartbeat":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.stop()

    async def _send(self, action: str) -> None:
        kwargs = {"action": action}
        if self._message_thread_id is not None:
            kwargs["message_thread_id"] = self._message_thread_id
        try:
            await self._chat.send_action(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Chat actions are cosmetic and must never break delivery.
            logger.debug("Telegram chat action %s failed: %s", action, error)

    async def _send_current(self) -> None:
        async with self._send_lock:
            await self._send(self._action)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_seconds)
            await self._send_current()

    async def start(self) -> None:
        if self._task is not None:
            return
        await self._send_current()
        self._task = asyncio.create_task(self._run())

    async def set_action(self, action: str) -> None:
        """Switch immediately; subsequent refreshes keep the new action alive."""
        if not action:
            return
        async with self._send_lock:
            if action == self._action:
                return
            self._action = action
            await self._send(action)

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
