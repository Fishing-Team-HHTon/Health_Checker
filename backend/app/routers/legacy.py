from __future__ import annotations
from fastapi import APIRouter, HTTPException
from datetime import datetime
from . import export as export_router_mod  # reuse download helpers
from ..services.recorder import recorder
from ..services.nnsender import nnsender
from ..models import IngestRequest, RecordingInfo
# Note: legacy routes mirror the new API behavior without the /api prefix.

router = APIRouter(tags=["legacy"])

@router.post("/start")
async def legacy_start(duration_sec: int | None = None):
    st = await recorder.start(duration_sec=duration_sec)
    nnsender.start()
    return {"recording_id": st.id, "duration_sec": st.duration_sec, "started_at": st.started_at}

@router.get("/active")
async def legacy_active():
    st = await recorder.active()
    if not st:
        return {"recording_id": None}
    elapsed = (datetime.utcnow() - st.started_at).total_seconds()
    return {"recording_id": st.id, "started_at": st.started_at, "duration_sec": st.duration_sec, "elapsed_sec": int(elapsed), "closed": st.closed}

@router.get("/status/{rid}")
async def legacy_status(rid: str):
    info = await recorder.info(rid)
    if not info:
        raise HTTPException(status_code=404, detail="recording not found")
    return info

# Downloads (re-use internal helpers)
@router.get("/download/{rid}.json")
async def legacy_download_json(rid: str):
    return await export_router_mod.download_json(rid)

@router.get("/download/{rid}.csv")
async def legacy_download_csv(rid: str):
    return await export_router_mod.download_csv(rid)

@router.get("/download/{rid}.txt")
async def legacy_download_txt(rid: str):
    return await export_router_mod.download_txt(rid)

# Legacy ingest (no /api prefix), same payload as /api/ingest
@router.post("/ingest")
async def legacy_ingest(req: IngestRequest):
    if not req.samples:
        return {"accepted": 0, "recording_id": None}
    rid = await recorder.add_samples(req.recording_id, req.samples)
    if not rid:
        raise HTTPException(status_code=400, detail="no active recording; call /start first or pass valid recording_id")
    await nnsender.enqueue_many(req.samples)
    return {"accepted": len(req.samples), "recording_id": rid}
