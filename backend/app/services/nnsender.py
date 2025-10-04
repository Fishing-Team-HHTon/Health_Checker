from __future__ import annotations
import asyncio
from typing import List, Optional
import httpx
from ..models import Sample
from ..settings import settings

class NNSender:
    def __init__(self):
        self._queue: asyncio.Queue[Sample] = asyncio.Queue(maxsize=4096)
        self._task: Optional[asyncio.Task] = None

    def start(self):
        if not settings.nn_url:
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._runner())

    async def _runner(self):
        async with httpx.AsyncClient(timeout=5.0) as client:
            batch: List[dict] = []
            while True:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                    batch.append(item.model_dump())
                    if len(batch) >= settings.max_batch_len:
                        await self._flush(client, batch); batch.clear()
                except asyncio.TimeoutError:
                    if batch:
                        await self._flush(client, batch); batch.clear()

    async def _flush(self, client: httpx.AsyncClient, batch: List[dict]):
        try:
            await client.post(settings.nn_url, json={"samples": batch})
        except Exception:
            pass

    async def enqueue_many(self, samples: List[Sample]):
        if not settings.nn_url:
            return
        for s in samples:
            try:
                self._queue.put_nowait(s)
            except asyncio.QueueFull:
                try:
                    _ = self._queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._queue.put_nowait(s)
                except Exception:
                    pass

nnsender = NNSender()
