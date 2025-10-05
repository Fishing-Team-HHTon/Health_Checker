from __future__ import annotations

import asyncio

from ..models import SignalMode


class ModeService:
    def __init__(self, initial: SignalMode = SignalMode.ecg):
        self._lock = asyncio.Lock()
        self._mode = initial

    async def get(self) -> SignalMode:
        async with self._lock:
            return self._mode

    async def set(self, mode: SignalMode) -> SignalMode:
        async with self._lock:
            self._mode = mode
            return self._mode


mode_service = ModeService()
