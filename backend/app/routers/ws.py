from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..models import SignalMode
from ..services.stream import stream_manager

router = APIRouter(tags=["ws"])


@router.websocket("/ws/{signal}")
async def websocket_endpoint(websocket: WebSocket, signal: SignalMode) -> None:
    await websocket.accept()
    await stream_manager.subscribe(signal, websocket)

    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await stream_manager.unsubscribe(signal, websocket)
