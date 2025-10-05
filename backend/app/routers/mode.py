from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..models import SignalMode
from ..services.mode import mode_service


router = APIRouter(prefix="/api", tags=["mode"])


class ModePayload(BaseModel):
    mode: SignalMode


@router.get("/mode", response_model=ModePayload)
async def get_mode() -> ModePayload:
    mode = await mode_service.get()
    return ModePayload(mode=mode)


@router.post("/mode", response_model=ModePayload)
async def set_mode(payload: ModePayload) -> ModePayload:
    mode = await mode_service.set(payload.mode)
    return ModePayload(mode=mode)
