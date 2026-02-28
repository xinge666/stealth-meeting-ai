import asyncio
import logging
from typing import Optional
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed

from ..shared.protocol import ClientMessage

logger = logging.getLogger(__name__)

class ClientConnection:
    """Manages the WebSocket connection from the Edge Client to the Server."""
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.ws: Optional[WebSocketClientProtocol] = None
        self._message_queue: asyncio.Queue[ClientMessage] = asyncio.Queue()
        self._running = False

    async def start(self):
        """Starts the connection and the background sender loop."""
        self._running = True
        logger.info(f"🔌 Connecting to Server at {self.server_url}...")
        try:
            self.ws = await connect(self.server_url)
            logger.info("✅ Connected to Server successfully.")
            
            # Start background sender task
            asyncio.create_task(self._sender_loop())
            
            # Start background receiver task (for server config updates/status)
            asyncio.create_task(self._receiver_loop())
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to server: {e}")
            self._running = False
            raise

    async def stop(self):
        self._running = False
        if self.ws:
            await self.ws.close()
            logger.info("🔌 Disconnected from Server.")

    async def send(self, message: ClientMessage):
        """Enqueues a message to be sent to the server."""
        await self._message_queue.put(message)

    async def _sender_loop(self):
        while self._running:
            try:
                msg = await self._message_queue.get()
                if self.ws and not self.ws.closed:
                    await self.ws.send(msg.to_json())
                    self._message_queue.task_done()
                else:
                    logger.warning("Tried to send message but WebSocket is closed.")
            except ConnectionClosed:
                logger.error("Connection closed while sending.")
                break
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    async def _receiver_loop(self):
        while self._running and self.ws and not self.ws.closed:
            try:
                response = await self.ws.recv()
                # Handle future Server->Client messages here
                logger.debug(f"Received from server: {response[:50]}")
            except ConnectionClosed:
                logger.warning("Connection closed by server.")
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                
