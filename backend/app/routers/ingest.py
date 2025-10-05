from __future__ import annotations
from fastapi import APIRouter, HTTPException
from ..models import IngestRequest
from ..services.recorder import recorder
from ..services.nnsender import nnsender
from ..services.mode import mode_service
from ..services.stream import stream_manager

router = APIRouter(prefix="/api", tags=["ingest"])

@router.post("/ingest")
async def ingest(req: IngestRequest):
    if not req.samples:
        return {"accepted": 0, "recording_id": None}
    rid = await recorder.add_samples(req.recording_id, req.samples)
    if rid is None:
        if req.recording_id:
            raise HTTPException(
                status_code=400,
                detail="invalid recording_id; call /api/recordings/start first or pass valid recording_id",
            )
        # No active recording – continue processing so realtime clients still receive data.
        # Samples won't be persisted, but streaming and downstream processing should proceed.
        rid = None
    await nnsender.enqueue_many(req.samples)
    current_mode = await mode_service.get()
    amplitudes = [sample.mv if sample.mv is not None else sample.adc for sample in req.samples]
    await stream_manager.broadcast(current_mode, amplitudes)
    return {"accepted": len(req.samples), "recording_id": rid}
