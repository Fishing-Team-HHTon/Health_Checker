from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from ..models import Sample, RecordingInfo
from ..settings import settings

class RecordingState:
    __slots__ = ("id","started_at","duration_sec","closed","samples")
    def __init__(self, rid: str, duration_sec: int):
        self.id = rid
        self.started_at = datetime.utcnow()
        self.duration_sec = duration_sec
        self.closed = False
        self.samples: List[Sample] = []

class Recorder:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._active: Optional[RecordingState] = None
        self._store: Dict[str, RecordingState] = {}
        self._auto_task: Optional[asyncio.Task] = None

    async def start(self, duration_sec: Optional[int] = None) -> RecordingState:
        async with self._lock:
            if self._active and not self._active.closed:
                self._active.closed = True
                self._store[self._active.id] = self._active
                self._active = None
            rid = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:-3]
            state = RecordingState(rid, duration_sec or settings.default_duration_sec)
            self._active = state
            self._store[state.id] = state
            if self._auto_task:
                self._auto_task.cancel()
            self._auto_task = asyncio.create_task(self._auto_close(state.id, state.duration_sec))
            return state

    async def _auto_close(self, rid: str, duration_sec: int):
        try:
            await asyncio.sleep(duration_sec)
        except asyncio.CancelledError:
            return
        async with self._lock:
            st = self._store.get(rid)
            if st and not st.closed:
                st.closed = True
                if self._active and self._active.id == rid:
                    self._active = None

    async def stop(self) -> Optional[RecordingState]:
        async with self._lock:
            if not self._active:
                return None
            self._active.closed = True
            st = self._active
            self._active = None
            return st

    async def active(self) -> Optional[RecordingState]:
        async with self._lock:
            return self._active

    async def add_samples(self, rid: Optional[str], samples: List[Sample]) -> Optional[str]:
        async with self._lock:
            target = None
            if rid:
                target = self._store.get(rid)
            else:
                target = self._active
            if not target or target.closed:
                return None
            target.samples.extend(samples)
            return target.id

    async def get(self, rid: str) -> Optional[RecordingState]:
        async with self._lock:
            return self._store.get(rid)

    async def info(self, rid: str) -> Optional[RecordingInfo]:
        st = await self.get(rid)
        if not st:
            return None
        return RecordingInfo(
            recording_id=st.id,
            started_at=st.started_at,
            duration_sec=st.duration_sec,
            closed=st.closed,
            count=len(st.samples),
        )

recorder = Recorder()
