import asyncio
import logging
import threading
import queue
from typing import Optional, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not installed. WebSocket streaming disabled.")


class WebSocketStreamingThread:

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
        Initialize WebSocket streaming thread.

        Args:
            coordinator_url: Coordinator base URL (e.g., "http://localhost:8000")
            job_id: Job ID
            worker_id: Worker ID
            max_queue_size: Maximum queued messages
            reconnect_delay: Initial reconnect delay (seconds)
            max_reconnect_delay: Max reconnect delay (seconds)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required for WebSocket streaming")

        # Convert HTTP to WebSocket URL
        self.ws_url = coordinator_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{self.ws_url}/ws/jobs/{job_id}/stream"

        self.job_id = job_id
        self.worker_id = worker_id

        # Thread-safe message queue (main thread → WebSocket thread)
        self.message_queue = queue.Queue(maxsize=max_queue_size)

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.stop_event = threading.Event()

        # Asyncio event loop (runs in WebSocket thread)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # WebSocket state
        self.websocket: Optional[Any] = None
        self.is_connected = False

        # Reconnection parameters
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.current_reconnect_delay = reconnect_delay

    def start(self):
        if self.is_running:
            logger.warning("WebSocket streaming thread already running")
            return

        self.is_running = True
        self.stop_event.clear()

        # Create and start background thread
        self.thread = threading.Thread(
            target=self._run_event_loop,
            name=f"WebSocket-{self.worker_id}",
            daemon=True
        )
        self.thread.start()

        logger.info(f"WebSocket streaming thread started for worker {self.worker_id}")

    def stop(self, timeout: float = 5.0):
        if not self.is_running:
            return

        logger.info("Stopping WebSocket streaming thread...")
        self.is_running = False
        self.stop_event.set()

        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

            if self.thread.is_alive():
                logger.warning("WebSocket thread did not stop gracefully")

        logger.info("WebSocket streaming thread stopped")

    def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        message = {
            "type": "metrics",
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics
        }

        try:
            # Non-blocking put with immediate return
            self.message_queue.put_nowait(message)
            return True
        except queue.Full:
            # Queue full, drop message
            logger.warning(f"Message queue full, dropping metrics for step {metrics.get('step')}")
            return False

    def send_log(self, level: str, message: str, **kwargs) -> bool:
        log_message = {
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

        try:
            self.message_queue.put_nowait(log_message)
            return True
        except queue.Full:
            logger.warning("Message queue full, dropping log message")
            return False

    def _run_event_loop(self):
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Run WebSocket connection manager
            self.loop.run_until_complete(self._connection_manager())
        except Exception as e:
            logger.error(f"WebSocket thread crashed: {e}", exc_info=True)
        finally:
            # Clean up event loop
            self.loop.close()
            self.loop = None

    async def _connection_manager(self):
        while self.is_running and not self.stop_event.is_set():
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

                    # Register worker with coordinator
                    await self._send_worker_registration(websocket)

                    # Run message sender and receiver concurrently
                    # Use create_task so they run in background
                    sender_task = asyncio.create_task(self._message_sender(websocket))
                    receiver_task = asyncio.create_task(self._message_receiver(websocket))

                    # Wait for either to complete (both run indefinitely until stopped)
                    done, pending = await asyncio.wait(
                        [sender_task, receiver_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()

            except ConnectionClosed:
                self.is_connected = False
                logger.warning(f"WebSocket closed, reconnecting in {self.current_reconnect_delay}s...")

            except WebSocketException as e:
                self.is_connected = False
                logger.warning(f"WebSocket error: {e}, reconnecting in {self.current_reconnect_delay}s...")

            except Exception as e:
                self.is_connected = False
                logger.error(f"Unexpected WebSocket error: {e}")

            finally:
                self.is_connected = False
                self.websocket = None

                if self.is_running and not self.stop_event.is_set():
                    # Exponential backoff
                    await asyncio.sleep(self.current_reconnect_delay)
                    self.current_reconnect_delay = min(
                        self.current_reconnect_delay * 2,
                        self.max_reconnect_delay
                    )

    async def _message_sender(self, websocket):
        logger.debug(f"WebSocket message sender started for worker {self.worker_id}")
        while self.is_running and not self.stop_event.is_set():
            try:
                # Get message from queue (non-blocking with timeout)
                try:
                    message = self.message_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Send message
                await websocket.send(json.dumps(message))
                logger.debug(f"Sent WebSocket message: type={message.get('type')}, step={message.get('data', {}).get('step')}")

                # Mark task as done
                self.message_queue.task_done()

            except ConnectionClosed:
                # Connection lost, exit sender loop
                logger.warning("Connection closed while sending message")
                break
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                await asyncio.sleep(0.1)

    def is_streaming(self) -> bool:
        return self.is_running and self.is_connected

    def get_queue_size(self) -> int:
        return self.message_queue.qsize()

    async def _send_worker_registration(self, websocket):
        registration_message = {
            "type": "worker_register",
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            await websocket.send(json.dumps(registration_message))
            logger.info(f"Sent worker registration for {self.worker_id}")
        except Exception as e:
            logger.warning(f"Failed to send worker registration: {e}")

    async def _message_receiver(self, websocket):
        logger.debug("WebSocket message receiver started")
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    message = json.loads(message_str)

                    message_type = message.get("type")

                    if message_type == "shutdown":
                        reason = message.get("reason", "unknown")
                        logger.error(f"🛑 Received shutdown command from coordinator: {reason}")
                        logger.error(f"   Worker {self.worker_id} terminating due to: {reason}")

                        # Trigger immediate shutdown
                        self.is_running = False
                        self.stop_event.set()

                        # Exit the worker process
                        import os
                        import signal
                        os.kill(os.getpid(), signal.SIGTERM)

                    elif message_type == "pong":
                        # Heartbeat response - ignore
                        pass

                    else:
                        logger.debug(f"Received message: {message_type}")

                except asyncio.TimeoutError:
                    # No message received, continue loop
                    continue
                except ConnectionClosed:
                    logger.info("WebSocket connection closed while receiving")
                    break
                except Exception as e:
                    logger.warning(f"Error receiving message: {e}")
                    break

        except Exception as e:
            logger.error(f"Message receiver error: {e}")
        finally:
            logger.debug("WebSocket message receiver stopped")

    def __repr__(self) -> str:
        return (
            f"WebSocketStreamingThread("
            f"worker={self.worker_id}, "
            f"running={self.is_running}, "
            f"connected={self.is_connected}, "
            f"queue_size={self.get_queue_size()})"
        )
