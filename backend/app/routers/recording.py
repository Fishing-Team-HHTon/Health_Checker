from __future__ import annotations
from fastapi import APIRouter, HTTPException
from ..services.recorder import recorder
from ..services.nnsender import nnsender
from ..models import RecordingInfo
from datetime import datetime

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

@router.post("/start")
async def start_recording(duration_sec: int | None = None):
    # Запуск записи и фоновый авто-стоп по duration_sec (дефолт 20с внутри Recorder)
    st = await recorder.start(duration_sec=duration_sec)
    nnsender.start()  # поднимем отправку в NN (если NN_URL задан)
    return {"recording_id": st.id, "duration_sec": st.duration_sec, "started_at": st.started_at}

@router.get("/active")
async def active_recording():
    st = await recorder.active()
    if not st:
        return {"recording_id": None}
    elapsed = (datetime.utcnow() - st.started_at).total_seconds()
    return {
        "recording_id": st.id,
        "started_at": st.started_at,
        "duration_sec": st.duration_sec,
        "elapsed_sec": int(elapsed),
        "closed": st.closed
    }

@router.get("/{rid}/status", response_model=RecordingInfo)
async def recording_status(rid: str):
    info = await recorder.info(rid)
    if not info:
        raise HTTPException(status_code=404, detail="recording not found")
    return info
