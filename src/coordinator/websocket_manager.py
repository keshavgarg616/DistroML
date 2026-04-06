import asyncio
import logging
import json
from typing import Dict, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    

    def __init__(self):
        # job_id -> Set[WebSocket] - Track all connections per job
        self.active_connections: Dict[str, Set[WebSocket]] = {}

        # WebSocket -> job_id - Reverse mapping for cleanup
        self.connection_to_job: Dict[WebSocket, str] = {}

        # worker_id -> WebSocket - Track worker-specific connections for shutdown commands
        self.worker_connections: Dict[str, WebSocket] = {}

        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str):
        
        await websocket.accept()

        async with self._lock:
            # Add to job-specific connection set
            if job_id not in self.active_connections:
                self.active_connections[job_id] = set()
            self.active_connections[job_id].add(websocket)

            # Add reverse mapping
            self.connection_to_job[websocket] = job_id

        logger.info(
            f"WebSocket connected for job {job_id}. "
            f"Total connections for this job: {len(self.active_connections[job_id])}"
        )

        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "job_id": job_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            websocket
        )

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            job_id = self.connection_to_job.get(websocket)

            worker_id_to_remove = None
            for worker_id, ws in self.worker_connections.items():
                if ws == websocket:
                    worker_id_to_remove = worker_id
                    break
            if worker_id_to_remove:
                del self.worker_connections[worker_id_to_remove]

            if job_id:
                # Remove from job-specific set
                if job_id in self.active_connections:
                    self.active_connections[job_id].discard(websocket)

                    # Clean up empty job sets
                    if not self.active_connections[job_id]:
                        del self.active_connections[job_id]

                # Remove reverse mapping
                del self.connection_to_job[websocket]

                logger.info(
                    f"WebSocket disconnected for job {job_id}. "
                    f"Remaining connections: {len(self.active_connections.get(job_id, []))}"
                )

    async def send_personal_message(self, message: dict, websocket: WebSocket):

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)

    async def broadcast_to_job(self, job_id: str, message: dict):
        
        if job_id not in self.active_connections:
            return  # No one watching this job

        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        # Create a copy of connections to avoid modification during iteration
        connections = list(self.active_connections.get(job_id, []))

        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                logger.info(f"Client disconnected during broadcast for job {job_id}")
                disconnected.append(connection)
            except Exception as e:
                logger.warning(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)

    async def broadcast_metrics(self, job_id: str, metrics: dict):
        message = {
            "type": "metrics",
            "job_id": job_id,
            "data": metrics,
        }
        await self.broadcast_to_job(job_id, message)

    async def broadcast_log(self, job_id: str, log_entry: dict):
        message = {
            "type": "log",
            "job_id": job_id,
            "data": log_entry,
        }
        await self.broadcast_to_job(job_id, message)

    async def broadcast_state_update(self, job_id: str, state_data: dict):
        
        message = {
            "type": "state_update",
            "job_id": job_id,
            "data": state_data,
        }
        await self.broadcast_to_job(job_id, message)

    def get_connection_count(self, job_id: str) -> int:
        return len(self.active_connections.get(job_id, []))

    def get_total_connections(self) -> int:
        return len(self.connection_to_job)

    def get_watched_jobs(self) -> Set[str]:
        return set(self.active_connections.keys())

    async def register_worker_connection(self, worker_id: str, websocket: WebSocket): 
        async with self._lock:
            self.worker_connections[worker_id] = websocket
            logger.debug(f"Registered worker connection for {worker_id}")

    async def send_shutdown_to_worker(self, worker_id: str, reason: str):
        async with self._lock:
            websocket = self.worker_connections.get(worker_id)
            if not websocket:
                logger.warning(f"Cannot send shutdown to {worker_id}: WebSocket not found")
                return

        shutdown_message = {
            "type": "shutdown",
            "reason": reason,
            "worker_id": worker_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            await self.send_personal_message(shutdown_message, websocket)
            logger.info(f"Sent shutdown command to worker {worker_id} (reason: {reason})")
        except Exception as e:
            logger.error(f"Failed to send shutdown to worker {worker_id}: {e}")
