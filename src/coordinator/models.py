from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..common import JobState, WorkerStatus

class WorkerResources(BaseModel):
    cpu_count: int = Field(default=2, ge=1, description="Number of CPU cores")
    memory_gb: float = Field(default=4.0, ge=0.1, description="Memory in GB")
    gpu_count: int = Field(default=0, ge=0, description="Number of GPUs")
    gpu_type: Optional[str] = Field(default=None, description="GPU type (e.g., 'A100', 'V100')")

class WorkerInfo(BaseModel):
    worker_id: str = Field(description="Unique worker identifier")
    rank: int = Field(ge=0, description="Worker rank in distributed training")
    world_size: int = Field(ge=1, description="Total number of workers")
    host: str = Field(description="Worker host address")
    port: int = Field(ge=1, le=65535, description="Worker port")
    job_id: str = Field(description="Associated job ID")
    status: WorkerStatus = Field(default=WorkerStatus.IDLE, description="Current worker status")
    registered_at: datetime = Field(default_factory=datetime.utcnow, description="Registration timestamp")
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow, description="Last heartbeat timestamp")
    resources: WorkerResources = Field(default_factory=WorkerResources, description="Allocated resources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"WorkerInfo(worker_id={self.worker_id}, rank={self.rank}/{self.world_size}, "
            f"status={self.status.value}, job_id={self.job_id})"
        )


class WorkerRegistration(BaseModel):
    worker_id: str = Field(description="Unique worker identifier")
    rank: int = Field(ge=0, description="Worker rank")
    world_size: int = Field(ge=1, description="Total workers")
    host: str = Field(description="Worker host")
    port: int = Field(ge=1, le=65535, description="Worker port")
    job_id: str = Field(description="Job ID this worker belongs to")
    resources: Optional[WorkerResources] = Field(default=None, description="Worker resources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HeartbeatPayload(BaseModel):
    worker_id: str = Field(description="Worker identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Heartbeat timestamp")
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Training metrics (step, loss, throughput, etc.)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class JobInfo(BaseModel):
    id: str = Field(description="Unique job identifier")
    name: str = Field(description="Human-readable job name")
    status: JobState = Field(default=JobState.QUEUED, description="Current job state")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if job failed")
    recovery_attempts: int = Field(default=0, ge=0, description="Number of recovery attempts")
    recovery_started_at: Optional[datetime] = Field(default=None, description="When recovery mode started")
    workers: list[WorkerInfo] = Field(default_factory=list, description="Associated workers")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional job metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"JobInfo(id={self.id}, name={self.name}, status={self.status.value}, "
            f"workers={len(self.workers)})"
        )


class JobSpec(BaseModel):
    name: str = Field(description="Job name")
    world_size: int = Field(ge=1, description="Number of workers to allocate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional job parameters")


class JobStatusResponse(BaseModel):
    job_id: str
    name: str
    status: JobState
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Progress 0.0-1.0")
    current_step: Optional[int] = Field(default=None, description="Current training step")
    workers: list[WorkerInfo] = Field(default_factory=list, description="Worker list")
    latest_metrics: Dict[str, float] = Field(default_factory=dict, description="Latest metrics")
    error_message: Optional[str] = Field(default=None, description="Error if failed")
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }



class ExperimentMetadata(BaseModel):
    """
    Full experiment record stored at job submission time.
    Captures everything needed to reproduce or compare a training run.
    """
    job_id: str = Field(description="Coordinator-assigned job UUID")
    run_id: str = Field(description="Run UUID (stable across recovery attempts)")
    recorded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was created",
    )

    git_commit_hash: str = Field(
        description="Git HEAD commit hash at submission time"
    )
    git_branch: str = Field(
        default="unknown",
        description="Git branch at submission time",
    )

    job_spec: Dict[str, Any] = Field(
        description="Full copy of the JobSpec used to submit this job"
    )

    seeds: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Random seeds keyed by name, e.g. "
            "{'global': 42, 'torch': 42, 'numpy': 42}. "
            "Pass these in job_spec.metadata.seeds when submitting."
        ),
    )

    runtime_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Python version, PyTorch version, CUDA, platform info",
    )

    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional context the submitter wants to attach",
    )
 
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
 
 
class ExperimentCompareRequest(BaseModel):
    """Request body for POST /api/experiments/compare."""
    run_ids: List[str] = Field(
        min_length=2,
        description="List of run_ids to compare (minimum 2)",
    )
 
 
class ExperimentCompareResponse(BaseModel):
    """Response for POST /api/experiments/compare."""
    compared_runs: int = Field(description="Number of runs found and compared")
    missing_runs: List[str] = Field(description="run_ids that were not found in the store")
    records: List[Dict[str, Any]] = Field(description="Full experiment records")
    summary: Dict[str, Any] = Field(
        description=(
            "High-level diff: same_git_commit, seeds_identical, "
            "torch_versions, etc."
        )
    )