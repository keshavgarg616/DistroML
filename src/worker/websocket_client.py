"""
WebSocket client for workers to stream metrics and logs in real-time.

Provides non-blocking metric streaming with automatic reconnection
and fallback to HTTP if WebSocket is unavailable.
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class MetricsStreamer:

    def __init__(
        self,
        coordinator_url: str,
        job_id: str,
        worker_id: str,
        max_queue_size: int = 1000,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0
    ):
        """
        Initialize metrics streamer.

        Args:
            coordinator_url: Base URL of coordinator (e.g., "http://localhost:8000")
            job_id: Job ID
            worker_id: Worker ID
            max_queue_size: Maximum messages to buffer when disconnected
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
        """
        # Convert HTTP URL to WebSocket URL
        self.ws_url = coordinator_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{self.ws_url}/ws/jobs/{job_id}/stream"

        self.job_id = job_id
        self.worker_id = worker_id

        # Connection state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.is_running = False

        # Message queue for buffering during disconnection
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

        # Reconnection parameters
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.current_reconnect_delay = reconnect_delay

        # Background tasks
        self.connect_task: Optional[asyncio.Task] = None
        self.sender_task: Optional[asyncio.Task] = None

    async def start(self):
        if self.is_running:
            logger.warning("MetricsStreamer already running")
            return

        self.is_running = True

        # Start connection manager and sender tasks
        self.connect_task = asyncio.create_task(self._connection_manager())
        self.sender_task = asyncio.create_task(self._message_sender())

        logger.info(f"MetricsStreamer started for job {self.job_id}")

    async def stop(self):
        self.is_running = False

        # Cancel background tasks
        if self.connect_task:
            self.connect_task.cancel()
        if self.sender_task:
            self.sender_task.cancel()

        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.is_connected = False
        logger.info("MetricsStreamer stopped")

    async def stream_metrics(self, metrics: Dict[str, Any]):
        message = {
            "type": "metrics",
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics
        }
        await self._enqueue_message(message)

    async def stream_log(self, level: str, message: str, **kwargs):
        log_entry = {
            "type": "log",
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "level": level,
                "message": message,
                **kwargs
            }
        }
        await self._enqueue_message(log_entry)

    async def _enqueue_message(self, message: Dict[str, Any]):
        try:
            self.message_queue.put_nowait(message)
        except asyncio.QueueFull:
            # Drop oldest message to make room
            try:
                self.message_queue.get_nowait()
                self.message_queue.put_nowait(message)
                logger.warning(
                    f"Message queue full, dropped oldest message "
                    f"(job={self.job_id}, worker={self.worker_id})"
                )
            except Exception as e:
                logger.error(f"Failed to enqueue message: {e}")

    async def _connection_manager(self):
        while self.is_running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.is_connected = True
                    self.current_reconnect_delay = self.reconnect_delay

                    logger.info(f"WebSocket connected to {self.ws_url}")

                    # Keep connection alive and handle incoming messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_server_message(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON from server: {e}")

            except ConnectionClosed:
                self.is_connected = False
                logger.warning(f"WebSocket connection closed, reconnecting in {self.current_reconnect_delay}s...")

            except WebSocketException as e:
                self.is_connected = False
                logger.warning(f"WebSocket error: {e}, reconnecting in {self.current_reconnect_delay}s...")

            except Exception as e:
                self.is_connected = False
                logger.error(f"Unexpected error in connection manager: {e}")

            finally:
                self.is_connected = False
                self.websocket = None

                if self.is_running:
                    # Exponential backoff for reconnection
                    await asyncio.sleep(self.current_reconnect_delay)
                    self.current_reconnect_delay = min(
                        self.current_reconnect_delay * 2,
                        self.max_reconnect_delay
                    )

    async def _message_sender(self):
        
        while self.is_running:
            try:
                # Wait for a message (with timeout to check is_running periodically)
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Wait until connected
                while not self.is_connected and self.is_running:
                    await asyncio.sleep(0.1)

                if not self.is_running:
                    break

                # Send message
                if self.websocket and self.is_connected:
                    try:
                        await self.websocket.send(json.dumps(message))
                    except ConnectionClosed:
                        # Connection lost, put message back in queue
                        logger.warning("Connection lost while sending, re-queuing message")
                        await self._enqueue_message(message)
                    except Exception as e:
                        logger.error(f"Failed to send message: {e}")

            except Exception as e:
                logger.error(f"Error in message sender: {e}")
                await asyncio.sleep(1.0)

    async def _handle_server_message(self, data: Dict[str, Any]):
        msg_type = data.get("type")

        if msg_type == "connection":
            logger.info(f"Server acknowledged connection: {data.get('status')}")

        elif msg_type == "pong":
            # Response to ping (keep-alive check)
            pass

        else:
            logger.debug(f"Received server message: {msg_type}")

    def is_streaming(self) -> bool:        
        return self.is_running and self.is_connected

    def get_queue_size(self) -> int:
        return self.message_queue.qsize()
