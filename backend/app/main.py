from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import recording, export, ingest, legacy

app = FastAPI(title="biosense-backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recording.router)
app.include_router(export.router)
app.include_router(ingest.router)
app.include_router(legacy.router)

@app.get("/")
def root():
    return {"ok": True, "name": "biosense-backend", "routes": [
        "/api/recordings/start",
        "/api/recordings/stop",
        "/api/recordings/active",
        "/api/recordings/{rid}/status",
        "/api/recordings/{rid}/download.json",
        "/api/recordings/{rid}/download.csv",
        "/api/recordings/{rid}/download.txt",
        "/api/ingest",
    ]}
