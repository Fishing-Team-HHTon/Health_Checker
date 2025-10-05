from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Iterable, Dict, Set

from fastapi import WebSocket

from ..models import SignalMode


class StreamManager:
    """Manage WebSocket subscriptions grouped by :class:`SignalMode`."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._connections: Dict[SignalMode, Set[WebSocket]] = defaultdict(set)

    async def subscribe(self, mode: SignalMode, websocket: WebSocket) -> None:
        """Register a WebSocket for the given signal mode."""
        async with self._lock:
            self._connections[mode].add(websocket)

    async def unsubscribe(self, mode: SignalMode, websocket: WebSocket) -> None:
        """Remove a WebSocket subscription for the given signal mode."""
        async with self._lock:
            clients = self._connections.get(mode)
            if not clients:
                return
            clients.discard(websocket)
            if not clients:
                self._connections.pop(mode, None)

    async def broadcast(self, mode: SignalMode, samples: Iterable[float | int]) -> None:
        """Send a batch of samples to all subscribers of the signal mode."""
        samples_list = list(samples)
        if not samples_list:
            return

        async with self._lock:
            targets = list(self._connections.get(mode, set()))

        if not targets:
            return

        payload = {"type": "batch", "mode": mode, "samples": samples_list}
        stale: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)

        if not stale:
            return

        async with self._lock:
            clients = self._connections.get(mode)
            if not clients:
                return
            for websocket in stale:
                clients.discard(websocket)
            if not clients:
                self._connections.pop(mode, None)


stream_manager = StreamManager()
