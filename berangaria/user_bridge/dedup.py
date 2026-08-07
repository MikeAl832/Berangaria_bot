"""Short-TTL dedup for (chat_id, message_id) so a reconnect cannot double-ingest."""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock


class MessageDeduper:
    """Process-local dedup. Not shared across replicas (single bot assumed)."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 4096):
        self._ttl = max(1.0, float(ttl_seconds))
        self._max_size = max(16, int(max_size))
        self._seen: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._lock = Lock()

    def _purge(self, now: float) -> None:
        while self._seen:
            key, ts = next(iter(self._seen.items()))
            if now - ts <= self._ttl and len(self._seen) <= self._max_size:
                break
            if now - ts > self._ttl:
                self._seen.popitem(last=False)
                continue
            # Over max_size: drop oldest even if still inside TTL.
            if len(self._seen) > self._max_size:
                self._seen.popitem(last=False)
                continue
            break

    def seen_or_add(self, chat_id: int, message_id: int, *, now: float | None = None) -> bool:
        """Return True if this key was already seen (caller should drop).

        First observation records the key and returns False.
        """
        stamp = time.monotonic() if now is None else now
        key = (int(chat_id), int(message_id))
        with self._lock:
            self._purge(stamp)
            if key in self._seen:
                # Refresh timestamp so active floods keep the key warm.
                self._seen.move_to_end(key)
                self._seen[key] = stamp
                return True
            self._seen[key] = stamp
            self._seen.move_to_end(key)
            self._purge(stamp)
            return False

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)
