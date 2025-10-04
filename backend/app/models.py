from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Sample(BaseModel):
    t_ms: int = Field(..., description="Device relative time in ms")
    ts_unix_ms: int = Field(..., description="Unix epoch ms")
    adc: int = Field(..., ge=0, le=1023, description="Raw ADC 0..1023")
    lead_off: Optional[bool] = False
    hp: Optional[float] = None
    mv: Optional[float] = None

class IngestRequest(BaseModel):
    recording_id: Optional[str] = None
    samples: List[Sample]

class RecordingInfo(BaseModel):
    recording_id: str
    started_at: datetime
    duration_sec: int
    closed: bool
    count: int
