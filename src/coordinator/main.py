import logging
from contextlib import asynccontextmanager
from typing import List
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect

from .coordinator import Coordinator, CoordinatorConfig
from .websocket_manager import ConnectionManager
from .models import (
    JobSpec,
    JobStatusResponse,
    JobInfo,
    JobState,
    WorkerRegistration,
    WorkerInfo,
    HeartbeatPayload,
)
from ..common import JobNotFoundError, WorkerNotFoundError, InvalidStateTransitionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coordinator: Coordinator = None
ws_manager: ConnectionManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global coordinator, ws_manager

    config = CoordinatorConfig()

    # Initialize WebSocket manager first
    ws_manager = ConnectionManager()
    logger.info("WebSocket manager initialized")

    # Initialize coordinator with WebSocket manager for state broadcasting
    logger.info("Initializing DistroML Coordinator...")
    coordinator = Coordinator(config, ws_manager=ws_manager)
    await coordinator.start()

    yield
    logger.info("Shutting down DistroML Coordinator...")
    await coordinator.shutdown()


app = FastAPI(
    title="DistroML Coordinator",
    description="Control plane for distributed ML training",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post(
    "/api/jobs",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    tags=["Jobs"],
    description="Submit a new training job",
)
async def submit_job(job_spec: JobSpec):
    try:
        job_id = await coordinator.submit_job(job_spec)
        return {"job_id": job_id, "status": "QUEUED"}
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get(
    "/api/jobs", response_model=List[JobInfo], tags=["Jobs"], summary="List all jobs"
)
async def list_jobs():
    return await coordinator.list_jobs()


async def get_job_status(job_id: str):
    try:
        return await coordinator.get_job_status(job_id)
    except JobNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/workers/register", status_code=status.HTTP_200_OK)
async def register_worker(registration: WorkerRegistration):
    try:
        await coordinator.register_worker(registration)
        return {"status": "registered", "worker_id": registration.worker_id}
    except Exception as e:
        logger.error(f"Worker registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# handle heartbeat


@app.post(
    "/api/workers/heartbeat",
    status_code=status.HTTP_200_OK,
    tags=["Workers"],
    summary="Receive worker heartbeat",
)
async def handle_heartbeat(payload: HeartbeatPayload):

    try:
        await coordinator.handle_heartbeat(payload)
        return {"status": "ok"}
    except WorkerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Worker not found. Please re-register.",
        )
    except Exception as e:
        logger.error(f"Heartbeat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workers/deregister", status_code=200, tags=["Workers"])
async def deregister_worker(payload: dict):
    try:
        await coordinator.handle_worker_exit(payload["worker_id"])
        return {"status": "deregistered"}
    except WorkerNotFoundError:
        raise HTTPException(status_code=404, detail="Worker not found")


@app.post(
    "/api/workers/complete",
    status_code=200,
    tags=["Workers"],
    summary="Worker finished training successfully",
)
async def worker_training_complete(payload: dict):
    """Remove worker from registry without triggering recovery."""
    try:
        worker_id = payload.get("worker_id")
        if not worker_id:
            raise HTTPException(
                status_code=400, detail="worker_id is required"
            )
        await coordinator.handle_worker_complete(worker_id)
        return {"status": "completed"}
    except WorkerNotFoundError:
        raise HTTPException(status_code=404, detail="Worker not found")


@app.get(
    "/api/cluster/workers",
    response_model=List[WorkerInfo],
    tags=["Cluster"],
    summary="List active workers",
)



async def list_workers():
    return await coordinator.worker_registry.list_workers()


@app.post(
    "/api/jobs/{job_id}/metrics",
    status_code=status.HTTP_200_OK,
    tags=["Metrics"],
    summary="Receive training metrics from workers",
)
async def receive_metrics(job_id: str, metrics: dict):
    
    try:
        # Add job_id to metrics for context
        metrics["job_id"] = job_id

        logger.info(f"Received metrics for job {job_id}: step={metrics.get('step')}, rank={metrics.get('rank')}")

        # Broadcast to all WebSocket clients watching this job
        await ws_manager.broadcast_metrics(job_id, metrics)

        logger.debug(f"Broadcast metrics for job {job_id}: step={metrics.get('step')}")

        return {"status": "received"}
    except Exception as e:
        logger.error(f"Failed to process metrics for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process metrics: {str(e)}"
        )


@app.websocket("/ws/jobs/{job_id}/stream")
async def websocket_stream(websocket: WebSocket, job_id: str):
    
    await ws_manager.connect(websocket, job_id)

    try:
        # Keep connection alive and listen for messages from workers/clients
        while True:
            # Wait for messages (metrics from workers, commands from clients)
            try:
                data = await websocket.receive_json()

                # Handle different message types
                message_type = data.get("type")

                if message_type == "worker_register":
                    # Worker identifying itself for shutdown commands
                    worker_id = data.get("worker_id")
                    if worker_id:
                        await ws_manager.register_worker_connection(worker_id, websocket)
                        logger.info(f"Worker {worker_id} registered its WebSocket connection")

                elif message_type == "metrics":
                    # Worker sending metrics - broadcast to all watchers
                    logger.debug(f"Received metrics via WebSocket for job {job_id}, step={data.get('data', {}).get('step')}")
                    try:
                        await ws_manager.broadcast_metrics(job_id, data.get("data", {}))
                    except Exception as e:
                        logger.error(f"Failed to broadcast metrics: {e}", exc_info=True)

                elif message_type == "log":
                    # Worker sending log - broadcast to all watchers
                    logger.debug(f"Received log via WebSocket for job {job_id}")
                    await ws_manager.broadcast_log(job_id, data.get("data", {}))

                elif data.get("command") == "ping":
                    # Client ping command - respond with pong
                    await ws_manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        websocket
                    )
                else:
                    logger.debug(f"Received unknown WebSocket message: type={message_type}, command={data.get('command')}")

            except WebSocketDisconnect:
                logger.info(f"Client disconnected from job {job_id} stream")
                break
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}", exc_info=True)
                break

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await ws_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("src.coordinator.main:app", host="0.0.0.0", port=8000, reload=True)
