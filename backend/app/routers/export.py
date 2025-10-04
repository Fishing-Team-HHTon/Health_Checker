from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..services.recorder import recorder
from ..services.storage import save_ndjson, save_csv, save_txt

router = APIRouter(prefix="/api/recordings", tags=["download"])

async def _paths_for(rid: str):
    st = await recorder.get(rid)
    if not st:
        raise HTTPException(status_code=404, detail="recording not found")
    p_json = save_ndjson(rid, st.samples)
    p_csv  = save_csv(rid, st.samples)
    p_txt  = save_txt(rid, st.samples)
    return p_json, p_csv, p_txt

@router.get("/{rid}/download.json")
async def download_json(rid: str):
    p_json, _, _ = await _paths_for(rid)
    return FileResponse(p_json, media_type="application/x-ndjson", filename=f"{rid}.ndjson")

@router.get("/{rid}/download.csv")
async def download_csv(rid: str):
    _, p_csv, _ = await _paths_for(rid)
    return FileResponse(p_csv, media_type="text/csv", filename=f"{rid}.csv")

@router.get("/{rid}/download.txt")
async def download_txt(rid: str):
    _, _, p_txt = await _paths_for(rid)
    return FileResponse(p_txt, media_type="text/plain", filename=f"{rid}.txt")
