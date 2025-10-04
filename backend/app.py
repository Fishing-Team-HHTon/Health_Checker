from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist

# ---------------------------
# Конфигурация
# ---------------------------

# Максимальный объём буфера на сигнал (точек). Подберите под частоты (например, ~60 секунд истории).
DEFAULT_HISTORY_SECONDS = 60
MAX_EXPECTED_FS = 4000  # верхняя граница на всякий случай
RING_BUFFER_SIZE = DEFAULT_HISTORY_SECONDS * MAX_EXPECTED_FS  # 60 * 4000 = 240k точек

# Ограничим размер входного батча, чтобы не убить сервер одним запросом
MAX_BATCH_SAMPLES = 50_000

# Простейший API-ключ (опционально). Если переменная задана, потребуется заголовок X-API-Key
API_KEY = os.getenv("FASTAPI_API_KEY")  # например, export FASTAPI_API_KEY=secret123

SIGNAL = Literal["ecg", "emg", "ppg", "resp"]
ALLOWED_SIGNALS = {"ecg", "emg", "ppg", "resp"}

# ---------------------------
# Модели
# ---------------------------

class SampleBatch(BaseModel):
    """
    Унифицированный формат для всех четырёх сигналов.
    """
    device_id: str = Field(..., description="Произвольный идентификатор контроллера/моста")
    fs: int = Field(..., ge=1, le=MAX_EXPECTED_FS, description="Частота дискретизации, Гц")
    samples: conlist(float, min_length=1, max_length=MAX_BATCH_SAMPLES)  # список значений
    t0: Optional[datetime] = Field(None, description="UTC-время первого сэмпла; если не указано — серверное now()")
    unit: Optional[str] = Field("V", description="Единицы измерения, справочно (например, 'V', 'mV', 'a.u.')")
    channel: Optional[int] = Field(0, description="Номер канала (если много каналов на устройстве)")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Произвольные метаданные")

class SamplePoint(BaseModel):
    t: float  # Unix time (секунды, float)
    y: float  # значение

class SeriesResponse(BaseModel):
    signal: SIGNAL
    device_id: Optional[str] = None
    fs: Optional[int] = None
    unit: Optional[str] = None
    count: int
    points: List[SamplePoint]

class Ack(BaseModel):
    status: str
    signal: SIGNAL
    received: int
    buffer_size: int
    subscribers: int

# ---------------------------
# Хранилище в памяти
# ---------------------------

# По каждому сигналу держим:
# - deque с кортежами (t, y)
# - список активных WebSocket-клиентов
class MemoryStore:
    def __init__(self, maxlen: int):
        self.buffers: Dict[str, deque[Tuple[float, float]]] = {
            "ecg": deque(maxlen=maxlen),
            "emg": deque(maxlen=maxlen),
            "ppg": deque(maxlen=maxlen),
            "resp": deque(maxlen=maxlen),
        }
        self.subscribers: Dict[str, Set[WebSocket]] = {k: set() for k in self.buffers.keys()}

    def append_batch(self, signal: str, times: List[float], values: List[float]) -> int:
        buf = self.buffers[signal]
        for t, y in zip(times, values):
            buf.append((t, y))
        return len(buf)

    def latest(self, signal: str, limit: int) -> List[Tuple[float, float]]:
        buf = self.buffers[signal]
        if limit <= 0:
            return []
        if limit >= len(buf):
            return list(buf)
        # взять хвост
        return list(list(buf)[-limit:])

    def subscriber_count(self, signal: str) -> int:
        return len(self.subscribers[signal])

store = MemoryStore(maxlen=RING_BUFFER_SIZE)

# ---------------------------
# WebSocket менеджер
# ---------------------------

async def ws_register(signal: str, ws: WebSocket) -> None:
    await ws.accept()
    store.subscribers[signal].add(ws)

def ws_unregister(signal: str, ws: WebSocket) -> None:
    try:
        store.subscribers[signal].remove(ws)
    except KeyError:
        pass

async def ws_broadcast(signal: str, payload: Dict[str, Any]) -> None:
    dead: List[WebSocket] = []
    for ws in list(store.subscribers[signal]):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_unregister(signal, ws)

# ---------------------------
# Утилиты
# ---------------------------

def now_utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()

def ensure_signal(sig: str) -> str:
    sig = sig.lower()
    if sig not in ALLOWED_SIGNALS:
        raise HTTPException(status_code=404, detail=f"Unknown signal '{sig}'. Allowed: {sorted(ALLOWED_SIGNALS)}")
    return sig

def check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

def build_times(t0: Optional[datetime], n: int, fs: int) -> List[float]:
    if t0 is None:
        t0_ts = now_utc_ts()
    else:
        # приводим к utc timestamp
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        else:
            t0 = t0.astimezone(timezone.utc)
        t0_ts = t0.timestamp()
    dt = 1.0 / float(fs)
    # равномерная сетка: t0, t0+dt, ...
    return [t0_ts + i * dt for i in range(n)]

# ---------------------------
# Приложение
# ---------------------------

app = FastAPI(title="BioSignals Backend (FastAPI)",
              version="0.1.0",
              description="ECG/EMG/PPG/Resp ingest + history + WebSocket streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при необходимости ограничьте домены фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "ts": now_utc_ts()}

# ---------- Общий обработчик для 4 HTTP путей (GET/POST) ----------

def _handle_post(signal: str, payload: SampleBatch, x_api_key: Optional[str]) -> Ack:
    check_api_key(x_api_key)
    signal = ensure_signal(signal)
    n = len(payload.samples)
    times = build_times(payload.t0, n, payload.fs)
    buf_size = store.append_batch(signal, times, payload.samples)

    # рассылаем подписчикам "как есть"
    broadcast_payload = {
        "type": "batch",
        "signal": signal,
        "device_id": payload.device_id,
        "fs": payload.fs,
        "unit": payload.unit,
        "channel": payload.channel,
        "meta": payload.meta or {},
        "t0": times[0],
        "n": n,
        "samples": payload.samples,
    }

    # fire-and-forget — здесь можно не await, но пусть будет корректно
    import anyio
    anyio.from_thread.run(ws_broadcast, signal, broadcast_payload)

    return Ack(
        status="ok",
        signal=signal, received=n,
        buffer_size=buf_size,
        subscribers=store.subscriber_count(signal)
    )

def _handle_get(signal: str, limit: int) -> SeriesResponse:
    signal = ensure_signal(signal)
    data = store.latest(signal, limit)
    points = [SamplePoint(t=t, y=y) for (t, y) in data]
    return SeriesResponse(signal=signal, count=len(points), points=points)

# ECG
@app.post("/ecg", response_model=Ack)
def post_ecg(payload: SampleBatch, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    return _handle_post("ecg", payload, x_api_key)

@app.get("/ecg", response_model=SeriesResponse)
def get_ecg(limit: int = Query(2000, ge=1, le=RING_BUFFER_SIZE)):
    return _handle_get("ecg", limit)

# EMG
@app.post("/emg", response_model=Ack)
def post_emg(payload: SampleBatch, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    return _handle_post("emg", payload, x_api_key)

@app.get("/emg", response_model=SeriesResponse)
def get_emg(limit: int = Query(2000, ge=1, le=RING_BUFFER_SIZE)):
    return _handle_get("emg", limit)

# PPG
@app.post("/ppg", response_model=Ack)
def post_ppg(payload: SampleBatch, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    return _handle_post("ppg", payload, x_api_key)

@app.get("/ppg", response_model=SeriesResponse)
def get_ppg(limit: int = Query(2000, ge=1, le=RING_BUFFER_SIZE)):
    return _handle_get("ppg", limit)

# Resp
@app.post("/resp", response_model=Ack)
def post_resp(payload: SampleBatch, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    return _handle_post("resp", payload, x_api_key)

@app.get("/resp", response_model=SeriesResponse)
def get_resp(limit: int = Query(2000, ge=1, le=RING_BUFFER_SIZE)):
    return _handle_get("resp", limit)

# ---------- WebSocket (real-time) ----------

@app.websocket("/ws/{signal}")
async def ws_signal(ws: WebSocket, signal: str):
    signal = ensure_signal(signal)
    await ws_register(signal, ws)
    try:
        # Ничего не ждём от клиента — просто держим соединение,
        # чтоб можно было отсылать батчи из POST-хендлеров.
        while True:
            # Если фронт хочет пинговать/посылать команды — можно читать:
            _ = await ws.receive_text()
            # и/или отвечать ok/ignore
    except WebSocketDisconnect:
        ws_unregister(signal, ws)
    except Exception:
        ws_unregister(signal, ws)
        # необязательный лог, чтобы не шуметь
        # print("WS error on", signal)
