import logging
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, status

from .coordinator import Coordinator, CoordinatorConfig
from .models import (
    JobSpec,
    JobStatusResponse,
    JobInfo,
    JobState,
    WorkerRegistration,
    WorkerInfo,
    HeartbeatPayload
)
from ..common import (
    JobNotFoundError,
    WorkerNotFoundError,
    InvalidStateTransitionError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coordinator : Coordinator = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    global coordinator
    
    config = CoordinatorConfig()
    
    logger.info("Initializing DistroML Coordinator...")
    coordinator = Coordinator(config)
    await coordinator.start()
    
    yield  
    logger.info("Shutting down DistroML Coordinator...")
    await coordinator.shutdown()

app = FastAPI(title = "DistroML Coordinator",description="Control plane for distributed ML training", version="0.1.0",lifespan=lifespan)

@app.post("api/jobs", response_model=dict,status_code=status.HTTP_201_CREATED,tags=["Jobs"],description="Submit a new training job")
async def submit_job(job_spec: JobSpec):
    try:
        job_id = await coordinator.submit_job(job_spec)
        return {"job_id":job_id,"status":"QUEUED"}
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        raise HTTPException(
            status_code= status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = str(e)
        )
    

@app.get("api/jobs",response_model=List[JobInfo],tags=["Jobs"],summary="List all jobs")
async def list_jobs():
    return await coordinator.list_jobs()

async def get_job_status(job_id: str):
    try:
        return await coordinator.get_job_status(job_id)
    except JobNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/api/workers/register", status_code = status.HTTP_200_OK)
async def register_worker(registration:WorkerRegistration):
    try:
        await coordinator.register_worker(registration)
        return {"status": "registered", "worker_id": registration.worker_id}
    except Exception as e:
        logger.error(f"Worker registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

#handle heartbeat

@app.post(
    "/api/workers/heartbeat",
    status_code=status.HTTP_200_OK,
    tags=["Workers"],
    summary="Receive worker heartbeat"
)
async def handle_heartbeat(payload: HeartbeatPayload):

    try:
        await coordinator.handle_heartbeat(payload)
        return {"status": "ok"}
    except WorkerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Worker not found. Please re-register."
        )
    except Exception as e:
        logger.error(f"Heartbeat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/cluster/workers",
    response_model=List[WorkerInfo],
    tags=["Cluster"],
    summary="List active workers"
)
async def list_workers():
    return await coordinator.worker_registry.list_workers()


if __name__ == "__main__":
    uvicorn.run("src.coordinator.main:app", host="0.0.0.0", port=8000, reload = True)