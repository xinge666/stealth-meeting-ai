"""
FastAPI WebSocket server for streaming LLM answers to mobile devices.
Provides a stealth, hardware-decoupled presentation layer.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config import ServerConfig
from ..event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class WebSocketServer:
    """
    FastAPI-based WebSocket server that broadcasts LLM responses
    to connected web clients.
    """

    def __init__(self, config: ServerConfig, event_bus: EventBus):
        self.config = config
        self.bus = event_bus
        self.app = FastAPI(title="Meeting Assistant", docs_url=None)
        self._active_connections: Set[WebSocket] = set()
        self._history = []  # List of dicts: {"type": "question"|"answer", "text": str}
        self._current_answer_buffer = ""

        self._setup_routes()

        # Subscribe to LLM events
        self.bus.subscribe(EventType.LLM_RESPONSE_CHUNK, self._handle_chunk)
        self.bus.subscribe(EventType.LLM_RESPONSE_DONE, self._handle_done)
        self.bus.subscribe(EventType.INTENT_QUESTION, self._handle_question)

    def _setup_routes(self):
        """Configure FastAPI routes."""

        @self.app.get("/")
        async def index():
            html_path = STATIC_DIR / "index.html"
            if html_path.exists():
                return FileResponse(html_path, media_type="text/html")
            return HTMLResponse("<h1>Meeting Assistant</h1><p>Static files not found.</p>")

        @self.app.get("/health")
        async def health():
            return {"status": "ok", "connections": len(self._active_connections)}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            # 1. Sync history and current state to the new client
            sync_payload = {
                "type": "sync",
                "history": self._history,
                "current_chunk": self._current_answer_buffer
            }
            await websocket.send_text(json.dumps(sync_payload, ensure_ascii=False))

            self._active_connections.add(websocket)
            logger.info("Client connected. Total: %d", len(self._active_connections))
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self._active_connections.discard(websocket)
                logger.info("Client disconnected. Total: %d", len(self._active_connections))

    async def _broadcast(self, message: dict):
        """Send a JSON message to all connected clients."""
        if not self._active_connections:
            return
        text = json.dumps(message, ensure_ascii=False)
        disconnected = set()
        for ws in self._active_connections:
            try:
                await ws.send_text(text)
            except Exception:
                disconnected.add(ws)
        self._active_connections -= disconnected

    async def _handle_question(self, event: Event):
        """Broadcast the detected question to clients and save to history."""
        text = event.data.get("text", "")
        self._history.append({"type": "question", "text": text})
        # Reset answer buffer for new question
        self._current_answer_buffer = ""
        
        await self._broadcast({
            "type": "question",
            "text": text,
            "confidence": event.data.get("confidence", 0),
        })

    async def _handle_chunk(self, event: Event):
        """Broadcast an LLM response chunk and append to buffer."""
        chunk = event.data.get("chunk", "")
        self._current_answer_buffer += chunk
        await self._broadcast({
            "type": "chunk",
            "text": chunk,
        })

    async def _handle_done(self, event: Event):
        """Broadcast LLM response completion and move buffer to history."""
        if self._current_answer_buffer:
            self._history.append({"type": "answer", "text": self._current_answer_buffer})
            # We don't clear buffer here so re-connecting clients can still see the last answer
            # It will be cleared on the next question.
        await self._broadcast({
            "type": "done",
        })

    async def start(self):
        """Start the uvicorn server."""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        logger.info(
            "WebSocket server starting on %s:%d",
            self.config.host, self.config.port
        )
        await server.serve()

    async def stop(self):
        """Close all connections."""
        for ws in self._active_connections.copy():
            try:
                await ws.close()
            except Exception:
                pass
        self._active_connections.clear()
        logger.info("WebSocket server stopped")
