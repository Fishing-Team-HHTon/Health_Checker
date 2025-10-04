from __future__ import annotations
from fastapi import APIRouter, HTTPException
from ..models import IngestRequest
from ..services.recorder import recorder
from ..services.nnsender import nnsender

router = APIRouter(prefix="/api", tags=["ingest"])

@router.post("/ingest")
async def ingest(req: IngestRequest):
    if not req.samples:
        return {"accepted": 0, "recording_id": None}
    rid = await recorder.add_samples(req.recording_id, req.samples)
    if not rid:
        raise HTTPException(status_code=400, detail="no active recording; call /api/recordings/start first or pass valid recording_id")
    await nnsender.enqueue_many(req.samples)
    return {"accepted": len(req.samples), "recording_id": rid}
