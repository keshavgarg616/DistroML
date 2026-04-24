"""
- CoordinatorConfig: Configuration management
- JobStateMachine: Job lifecycle state machine
- WorkerRegistry: Thread-safe worker tracking
- HeartbeatMonitor: Background heartbeat monitoring
- Coordinator: Main orchestration hub
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from src.coordinator.manifest import build_manifest, save_manifest

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from statemachine import StateMachine, State
import os
import subprocess
import sys
from pathlib import Path
import random
import numpy
import json

from .models import (
    WorkerResources,
    JobInfo,
    JobSpec,
    WorkerInfo,
    WorkerRegistration,
    HeartbeatPayload,
    JobStatusResponse,
)
from ..common import (
    JobState,
    WorkerStatus,
    HEARTBEAT_TIMEOUT_SECONDS,
    HEARTBEAT_CHECK_INTERVAL_SECONDS,
    MAX_RECOVERY_ATTEMPTS,
    RECOVERY_BACKOFF_SECONDS,
    WORKER_REGISTRATION_TIMEOUT_SECONDS,
    JobNotFoundError,
    WorkerNotFoundError,
    InvalidStateTransitionError,
)

logger = logging.getLogger(__name__)

# Project root (…/DistroML) for subprocess worker relaunch
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CoordinatorConfig(BaseSettings):

    heartbeat_timeout_seconds: int = Field(
        default=HEARTBEAT_TIMEOUT_SECONDS,
        description="Seconds before a worker is considered lost due to missing heartbeat",
        ge=1,
    )

    heartbeat_check_interval_seconds: int = Field(
        default=HEARTBEAT_CHECK_INTERVAL_SECONDS,
        description="Interval for checking worker heartbeats",
        ge=1,
    )

    max_recovery_attempts: int = Field(
        default=MAX_RECOVERY_ATTEMPTS,
        description="Maximum number of recovery attempts before marking job as FAILED",
        ge=0,
    )
    recovery_backoff_seconds: int = Field(
        default=RECOVERY_BACKOFF_SECONDS,
        description="Backoff time between recovery attempts",
        ge=0,
    )

    worker_registration_timeout_seconds: int = Field(
        default=WORKER_REGISTRATION_TIMEOUT_SECONDS,
        description="Timeout for worker registration after job start",
        ge=1,
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    coordinator_base_url: str = Field(
        default="http://127.0.0.1:8000",
        description="Base URL workers use to reach this coordinator (recovery relaunch)",
    )

    model_config = SettingsConfigDict(
        env_prefix="DISTROML_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def __repr__(self) -> str:
        return (
            f"CoordinatorConfig("
            f"heartbeat_timeout={self.heartbeat_timeout_seconds}s, "
            f"check_interval={self.heartbeat_check_interval_seconds}s, "
            f"max_recovery_attempts={self.max_recovery_attempts}, "
            f"log_level={self.log_level})"
        )


class JobStateMachine(StateMachine):
    """
    Job lifecycle state machine.

    States:
        QUEUED (initial) → RUNNING → COMPLETED/FAILED
        RUNNING → RECOVERING → RUNNING/FAILED
        QUEUED/RUNNING → CANCELLED

    Transitions:
        - start: QUEUED → RUNNING
        - complete: RUNNING → COMPLETED, RECOVERING → COMPLETED (all workers finished cleanly)
        - fail: RUNNING/RECOVERING → FAILED
        - recover: RUNNING → RECOVERING
        - resume: RECOVERING → RUNNING
        - cancel: QUEUED/RUNNING → CANCELLED
    """

    queued = State(JobState.QUEUED, initial=True, value=JobState.QUEUED)
    running = State(JobState.RUNNING, value=JobState.RUNNING)
    recovering = State(JobState.RECOVERING, value=JobState.RECOVERING)
    completed = State(JobState.COMPLETED, final=True, value=JobState.COMPLETED)
    failed = State(JobState.FAILED, final=True, value=JobState.FAILED)
    cancelled = State(JobState.CANCELLED, final=True, value=JobState.CANCELLED)

    start = queued.to(running)
    complete = running.to(completed) | recovering.to(completed)
    fail = running.to(failed) | recovering.to(failed)
    recover = running.to(recovering)
    resume = recovering.to(running)
    cancel = queued.to(cancelled) | running.to(cancelled) | recovering.to(cancelled)

    def __init__(
        self,
        job_id: str,
        on_transition: Optional[Callable[[str, str, str], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.job_id = job_id
        self._on_transition_callback = on_transition

    def on_transition(
        self, event: str, source: State, target: State, **kwargs: Any
    ) -> None:
        """Called on any state transition."""
        logger.info(
            f"Job {self.job_id} state transition: {source.value} → {target.value} (event: {event})"
        )

        if self._on_transition_callback:
            try:
                self._on_transition_callback(self.job_id, source.value, target.value)
            except Exception as e:
                logger.error(
                    f"Error in state transition callback for job {self.job_id}: {e}",
                    exc_info=True,
                )

    def on_enter_running(self) -> None:
        """Called when entering RUNNING state."""
        logger.info(f"Job {self.job_id} started running")

    def on_enter_recovering(self) -> None:
        """Called when entering RECOVERING state."""
        pass

    def on_enter_failed(self) -> None:
        """Called when entering FAILED state."""
        logger.error(f"Job {self.job_id} failed")

    def on_enter_completed(self) -> None:
        """Called when entering COMPLETED state."""
        logger.info(f"Job {self.job_id} completed successfully")

    def on_enter_cancelled(self) -> None:
        """Called when entering CANCELLED state."""
        logger.info(f"Job {self.job_id} cancelled")

    def get_current_state(self) -> JobState:
        """Get the current state value."""
        return self.current_state.value

    def can_transition(self, event_name: str) -> bool:
        """Check if a transition is valid from the current state."""
        try:
            transition = getattr(self, event_name, None)
            if transition is None:
                return False
            return transition._can_run(self)
        except Exception:
            return False

    def safe_transition(self, event_name: str, **kwargs: Any) -> bool:
        """Attempt a state transition, returning success status."""
        if not self.can_transition(event_name):
            raise InvalidStateTransitionError(
                from_state=self.current_state.value, to_state=event_name
            )

        try:
            transition = getattr(self, event_name)
            transition(**kwargs)
            return True
        except Exception as e:
            logger.error(
                f"Failed to execute transition {event_name} for job {self.job_id}: {e}",
                exc_info=True,
            )
            return False

    def is_terminal(self) -> bool:
        """Check if the current state is terminal (final)."""
        return self.current_state in [self.completed, self.failed, self.cancelled]

    def __repr__(self) -> str:
        return (
            f"JobStateMachine(job_id={self.job_id}, state={self.current_state.value})"
        )


class WorkerRegistry:

    def __init__(self):
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = asyncio.Lock()
        logger.info("WorkerRegistry initialized")

    async def register_worker(self, worker_info: WorkerInfo) -> None:
        async with self._lock:
            existing = self._workers.get(worker_info.worker_id)
            if existing:
                logger.info(
                    f"Re-registering worker {worker_info.worker_id} "
                    f"(previous status: {existing.status.value})"
                )
            else:
                logger.info(
                    f"Registering new worker {worker_info.worker_id} "
                    f"(rank {worker_info.rank}/{worker_info.world_size}, job {worker_info.job_id})"
                )

            worker_info.registered_at = datetime.now(timezone.utc)
            worker_info.last_heartbeat = datetime.now(timezone.utc)
            self._workers[worker_info.worker_id] = worker_info

    async def deregister_worker(self, worker_id: str) -> None:
        async with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotFoundError(worker_id)

            worker = self._workers[worker_id]
            logger.info(
                f"Deregistering worker {worker_id} "
                f"(rank {worker.rank}, job {worker.job_id})"
            )
            del self._workers[worker_id]

    async def update_heartbeat(
        self, worker_id: str, metrics: Optional[Dict[str, float]] = None
    ) -> None:
        async with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotFoundError(worker_id)

            worker = self._workers[worker_id]
            worker.last_heartbeat = datetime.now(timezone.utc)

            if metrics:
                if "latest_metrics" not in worker.metadata:
                    worker.metadata["latest_metrics"] = {}
                worker.metadata["latest_metrics"].update(metrics)

            logger.debug(f"Heartbeat updated for worker {worker_id}")

    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        async with self._lock:
            return self._workers.get(worker_id)

    async def list_workers(
        self, job_id: Optional[str] = None, status: Optional[WorkerStatus] = None
    ) -> List[WorkerInfo]:
        async with self._lock:
            workers = list(self._workers.values())

            if job_id is not None:
                workers = [w for w in workers if w.job_id == job_id]

            if status is not None:
                workers = [w for w in workers if w.status == status]

            return workers

    async def get_workers_by_job(self, job_id: str) -> List[WorkerInfo]:
        return await self.list_workers(job_id=job_id)

    async def mark_worker_lost(self, worker_id: str) -> None:
        async with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotFoundError(worker_id)

            worker = self._workers[worker_id]
            old_status = worker.status
            worker.status = WorkerStatus.LOST

            logger.warning(
                f"Worker {worker_id} marked as LOST "
                f"(previous status: {old_status.value}, job: {worker.job_id})"
            )

    async def get_stale_workers(self, timeout_seconds: int) -> List[WorkerInfo]:
        async with self._lock:
            now = datetime.now(timezone.utc)
            threshold = now - timedelta(seconds=timeout_seconds)

            stale_workers = [
                worker
                for worker in self._workers.values()
                if worker.last_heartbeat < threshold
                and worker.status != WorkerStatus.LOST
            ]

            if stale_workers:
                logger.debug(
                    f"Found {len(stale_workers)} stale workers "
                    f"(timeout threshold: {timeout_seconds}s)"
                )

            return stale_workers

    async def count_workers(
        self, job_id: Optional[str] = None, status: Optional[WorkerStatus] = None
    ) -> int:
        workers = await self.list_workers(job_id=job_id, status=status)
        return len(workers)

    async def get_all_worker_ids(self) -> List[str]:
        async with self._lock:
            return list(self._workers.keys())

    async def clear(self) -> None:
        async with self._lock:
            count = len(self._workers)
            self._workers.clear()
            logger.info(f"Cleared {count} workers from registry")

    def __repr__(self) -> str:
        return f"WorkerRegistry(workers={len(self._workers)})"


class HeartbeatMonitor:

    def __init__(
        self,
        worker_registry: WorkerRegistry,
        timeout_seconds: int = 30,
        check_interval_seconds: int = 5,
        on_worker_lost: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.worker_registry = worker_registry
        self.timeout_seconds = timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self._on_worker_lost = on_worker_lost

        self._task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"HeartbeatMonitor initialized "
            f"(timeout={timeout_seconds}s, check_interval={check_interval_seconds}s)"
        )

    async def start(self) -> None:
        if self._running:
            logger.warning("HeartbeatMonitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("HeartbeatMonitor started")

    async def stop(self) -> None:
        if not self._running:
            logger.warning("HeartbeatMonitor not running")
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("HeartbeatMonitor stopped")

    async def _monitor_loop(self) -> None:
        logger.info("HeartbeatMonitor loop started")

        while self._running:
            try:
                await self._check_heartbeats()
            except Exception as e:
                logger.error(f"Error in heartbeat check loop: {e}", exc_info=True)

            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                logger.info("HeartbeatMonitor loop cancelled")
                break

        logger.info("HeartbeatMonitor loop exited")

    async def _check_heartbeats(self) -> None:
        stale_workers = await self.worker_registry.get_stale_workers(
            timeout_seconds=self.timeout_seconds
        )

        if not stale_workers:
            return

        logger.warning(f"Detected {len(stale_workers)} workers with stale heartbeats")

        for worker in stale_workers:
            await self._handle_timeout(worker)

    async def _handle_timeout(self, worker: WorkerInfo) -> None:
        worker_id = worker.worker_id
        logger.error(
            f"Worker {worker_id} heartbeat timeout "
            f"(job: {worker.job_id}, rank: {worker.rank}, "
            f"last_heartbeat: {worker.last_heartbeat})"
        )

        try:
            await self.worker_registry.mark_worker_lost(worker_id)
        except Exception as e:
            logger.error(
                f"Failed to mark worker {worker_id} as lost: {e}", exc_info=True
            )

        if self._on_worker_lost:
            try:
                await self._on_worker_lost(worker_id)
            except Exception as e:
                logger.error(
                    f"Error in on_worker_lost callback for {worker_id}: {e}",
                    exc_info=True,
                )

    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        return (
            f"HeartbeatMonitor("
            f"timeout={self.timeout_seconds}s, "
            f"check_interval={self.check_interval_seconds}s, "
            f"running={self._running})"
        )


class Coordinator:

    def __init__(self, config: Optional[CoordinatorConfig] = None, ws_manager=None):
        self.config = config or CoordinatorConfig()

        self._jobs: Dict[str, Tuple[JobInfo, JobStateMachine]] = {}
        self._jobs_lock = asyncio.Lock()

        self.worker_registry = WorkerRegistry()
        # metrics_store holds per-job, per-step aggregated metrics
        # Structure:
        # { job_id: {
        #     "steps": {
        #         step_number: {
        #             "worker_metrics": { worker_id: metrics_dict },
        #             "aggregated": {"avg_loss": float|None, "total_throughput": float, "workers_reported": int}
        #         }
        #     },
        #     "run_summary": {"latest_step": int|None, "latest_loss": float|None, "best_loss": float|None, "avg_throughput": float|None}
        # } }
        self._metrics_store: Dict[str, Dict[str, Any]] = {}

        self.heartbeat_monitor = HeartbeatMonitor(
            worker_registry=self.worker_registry,
            timeout_seconds=self.config.heartbeat_timeout_seconds,
            check_interval_seconds=self.config.heartbeat_check_interval_seconds,
            on_worker_lost=self._on_worker_lost,
        )
        self._checkpoint_reports = {}

        # WebSocket manager for real-time state broadcasting
        self.ws_manager = ws_manager

        logger.info(f"Coordinator initialized with config: {self.config}")

    async def start(self) -> None:
        logger.info("Starting Coordinator...")
        await self.heartbeat_monitor.start()
        logger.info("Coordinator started successfully")

    async def shutdown(self) -> None:
        logger.info("Shutting down Coordinator...")
        await self.heartbeat_monitor.stop()

        async with self._jobs_lock:
            logger.info(
                f"Coordinator shutdown complete. "
                f"Total jobs: {len(self._jobs)}, "
                f"Total workers: {await self.worker_registry.count_workers()}"
            )

    async def submit_job(self, job_spec: JobSpec) -> str:
        job_id = str(uuid.uuid4())

        run_id = (
            job_spec.metadata.get("run_id")
            if job_spec.metadata and "run_id" in job_spec.metadata
            else str(uuid.uuid4())
        )

        deterministic = getattr(job_spec, "deterministic", False)

        # Global seed (deterministic OR random)
        if job_spec.metadata and "seed" in job_spec.metadata:
            global_seed = job_spec.metadata["seed"]
        elif deterministic:
            # deterministic run MUST be reproducible
            global_seed = 42  # or config-fixed seed
        else:
            global_seed = random.randint(0, 10_000)

        world_size = job_spec.world_size

        # Per-worker seeds (stable + reproducible)
        worker_seeds = {
            str(rank): global_seed + rank
            for rank in range(world_size)
        }

        experiment_metadata = {
            "job_id": job_id,
            "run_id": run_id,
            "deterministic": deterministic,
            "global_seed": global_seed,
            "world_size": world_size,
            "worker_seeds": worker_seeds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_dir = os.path.join("./checkpoints", run_id)
        metadata_path = os.path.join(metadata_dir, "experiment_metadata.json")
        if os.path.exists(metadata_dir):
            with open(metadata_path, "r") as f:
                experiment_metadata = json.load(f)
                global_seed = experiment_metadata["global_seed"]
                worker_seeds = experiment_metadata["worker_seeds"]
                if job_spec.world_size != experiment_metadata["world_size"]:
                    raise ValueError(f"World size mismatch: {job_spec.world_size} != {experiment_metadata['world_size']}")
                elif job_spec.deterministic != experiment_metadata["deterministic"]:
                    raise ValueError(f"Deterministic mismatch: {job_spec.deterministic} != {experiment_metadata['deterministic']}")
        else:
            os.makedirs(metadata_dir, exist_ok=True)

            with open(metadata_path, "w") as f:
                json.dump(experiment_metadata, f, indent=2)

            logger.info(f"Experiment metadata created at {metadata_path}")
        deterministic = experiment_metadata["deterministic"]

        merged_meta: Dict[str, Any] = dict(job_spec.metadata or {})

        merged_meta.update({
            "run_id": run_id,
            "seed": global_seed,
            "deterministic": deterministic,
            "experiment_metadata_path": metadata_path,
            "world_size": job_spec.world_size,
            "total_steps": 100,
            "checkpoint_dir": "./checkpoints",
        })

        job_info = JobInfo(
            id=job_id,
            name=job_spec.name,
            status=JobState.QUEUED,
            metadata=merged_meta,
        )

        state_machine = JobStateMachine(
            job_id=job_id,
            on_transition=self._on_job_state_transition,
        )

        async with self._jobs_lock:
            self._jobs[job_id] = (job_info, state_machine)

        logger.info(
            f"Job submitted: {job_id} (name: {job_spec.name}, "
            f"world_size: {job_spec.world_size})"
        )

        return job_id

    async def start_job(self, job_id: str) -> None:
        async with self._jobs_lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)

            job_info, state_machine = self._jobs[job_id]

            state_machine.start()
            job_info.status = JobState.RUNNING
            job_info.started_at = datetime.now(timezone.utc)
            job_info.updated_at = datetime.now(timezone.utc)

        logger.info(f"Job {job_id} started")

    async def cancel_job(self, job_id: str) -> None:
        async with self._jobs_lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)

            job_info, state_machine = self._jobs[job_id]

            try:
                state_machine.cancel()
            except Exception as e:
                logger.warning(
                    f"Cannot cancel job {job_id} in state {state_machine.get_current_state()}: {e}"
                )
                return

            job_info.status = JobState.CANCELLED
            job_info.completed_at = datetime.now(timezone.utc)
            job_info.updated_at = datetime.now(timezone.utc)

        logger.info(f"Job {job_id} cancelled")

    async def complete_job(self, job_id: str) -> None:
        async with self._jobs_lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)

            job_info, state_machine = self._jobs[job_id]

            state_machine.complete()
            job_info.status = JobState.COMPLETED
            job_info.completed_at = datetime.now(timezone.utc)
            job_info.updated_at = datetime.now(timezone.utc)

        logger.info(f"Job {job_id} completed")

    async def fail_job(self, job_id: str, error_message: str) -> None:
        async with self._jobs_lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)

            job_info, state_machine = self._jobs[job_id]

            state_machine.fail()
            job_info.status = JobState.FAILED
            job_info.error_message = error_message
            job_info.completed_at = datetime.now(timezone.utc)
            job_info.updated_at = datetime.now(timezone.utc)

        logger.error(f"Job {job_id} failed: {error_message}")

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        async with self._jobs_lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)

            job_info, state_machine = self._jobs[job_id]

            workers = await self.worker_registry.get_workers_by_job(job_id)

            latest_metrics = {}
            for worker in workers:
                if "latest_metrics" in worker.metadata:
                    latest_metrics.update(worker.metadata["latest_metrics"])

            return JobStatusResponse(
                job_id=job_info.id,
                name=job_info.name,
                status=job_info.status,
                workers=workers,
                latest_metrics=latest_metrics,
                error_message=job_info.error_message,
                created_at=job_info.created_at,
                updated_at=job_info.updated_at,
            )

    async def list_jobs(self) -> List[JobInfo]:
        async with self._jobs_lock:
            return [job_info for job_info, _ in self._jobs.values()]

    async def register_worker(self, registration: WorkerRegistration) -> None:
        worker_info = WorkerInfo(
            worker_id=registration.worker_id,
            rank=registration.rank,
            world_size=registration.world_size,
            host=registration.host,
            port=registration.port,
            job_id=registration.job_id,
            status=WorkerStatus.BUSY,
            resources=registration.resources or WorkerResources(),
            metadata=registration.metadata,
        )

        await self.worker_registry.register_worker(worker_info)

        logger.info(
            f"Worker registered: {registration.worker_id} "
            f"(job: {registration.job_id}, rank: {registration.rank})"
        )

    async def handle_heartbeat(self, heartbeat: HeartbeatPayload) -> None:
        await self.worker_registry.update_heartbeat(
            worker_id=heartbeat.worker_id,
            metrics=heartbeat.metrics,
        )

        # Process metrics for aggregation (per-step)
        try:
            await self._process_metrics_from_heartbeat(heartbeat)
        except Exception as e:
            logger.error(f"Error processing metrics from heartbeat: {e}", exc_info=True)

        logger.debug(f"Heartbeat received from worker {heartbeat.worker_id}")

    async def handle_worker_exit(self, worker_id: str) -> None:
        """Called when a worker explicitly notifies the coordinator it is exiting."""
        worker = await self.worker_registry.get_worker(worker_id)
        if not worker:
            raise WorkerNotFoundError(worker_id)

        job_id = worker.job_id
        await self.worker_registry.mark_worker_lost(worker_id)
        logger.warning(f"Worker {worker_id} reported exit (job: {job_id})")
        logger.info(f"Failure detected time: {datetime.now(timezone.utc)}")
        await self._trigger_recovery(job_id, worker_id)

    async def handle_worker_complete(self, worker_id: str) -> None:
        """
        Called when a worker finishes training successfully.
        Removes the worker from the registry (no recovery). If a tracked job
        exists and is RUNNING, marks the job COMPLETED when the last worker
        for that job shuts down cleanly.
        """
        worker = await self.worker_registry.get_worker(worker_id)
        if not worker:
            raise WorkerNotFoundError(worker_id)

        job_id = worker.job_id
        await self.worker_registry.deregister_worker(worker_id)
        logger.info(
            f"Worker {worker_id} completed training and deregistered (job {job_id})"
        )

        remaining = await self.worker_registry.get_workers_by_job(job_id)
        if len(remaining) > 0:
            return

        async with self._jobs_lock:
            if job_id not in self._jobs:
                return

            job_info, state_machine = self._jobs[job_id]
            if job_info.status not in (JobState.RUNNING, JobState.RECOVERING):
                logger.debug(
                    f"Job {job_id} not auto-completed (status={job_info.status})"
                )
                return

            if not state_machine.safe_transition("complete"):
                logger.warning(
                    f"Could not complete job {job_id} from state "
                    f"{state_machine.get_current_state()}"
                )
                return

            job_info.status = JobState.COMPLETED
            job_info.completed_at = datetime.now(timezone.utc)
            job_info.updated_at = datetime.now(timezone.utc)

        logger.info(f"Job {job_id} completed (all workers finished successfully)")

    async def _on_worker_lost(self, worker_id: str) -> None:
        logger.error(f"Worker lost detected: {worker_id}")
        logger.info(f"Failure detected time: {datetime.now(timezone.utc)}")

        worker = await self.worker_registry.get_worker(worker_id)
        if not worker:
            logger.warning(f"Cannot find worker {worker_id} to handle loss")
            return

        # Send shutdown command to worker via WebSocket
        if self.ws_manager:
            await self.ws_manager.send_shutdown_to_worker(worker_id, "heartbeat_timeout")

        job_id = worker.job_id
        await self._trigger_recovery(job_id, worker_id)

    async def _trigger_recovery(self, job_id: str, failed_worker_id: str) -> None:
        failed_worker = await self.worker_registry.get_worker(failed_worker_id)
        if failed_worker is None:
            logger.error(
                f"Recovery aborted: failed worker {failed_worker_id} not in registry"
            )
            return
        failed_rank = failed_worker.rank

        async with self._jobs_lock:
            if job_id not in self._jobs:
                logger.warning(f"Cannot trigger recovery - job {job_id} not found")
                return

            job_info, state_machine = self._jobs[job_id]

            # Check if all workers are lost
            all_workers = await self.worker_registry.get_workers_by_job(job_id)
            alive_workers = [w for w in all_workers if w.status != WorkerStatus.LOST]

            if len(alive_workers) == 0:
                # Catastrophic failure - all workers dead
                logger.error(
                    f"All workers lost for job {job_id}. Marking as FAILED immediately."
                )
                try:
                    state_machine.fail()
                except Exception:
                    pass
                job_info.status = JobState.FAILED
                job_info.error_message = f"All workers lost (last: {failed_worker_id})"
                job_info.completed_at = datetime.now(timezone.utc)
                return

            # Try to enter recovery mode
            try:
                state_machine.recover()
            except Exception as e:
                logger.warning(f"Cannot trigger recovery for job {job_id}: {e}")
                return

            job_info.status = JobState.RECOVERING

            meta = job_info.metadata or {}

            run_id = meta.get("run_id")
            world_size = meta.get("world_size", 2)
            total_steps = meta.get("total_steps", 100)
            checkpoint_dir = meta.get("checkpoint_dir", "./checkpoints")

            if not run_id:
                logger.error("Recovery failed: run_id missing in metadata")
                return

            logger.warning(
                f"[RECOVERY] Relaunching worker rank {failed_rank} only "
                f"(failed_worker_id={failed_worker_id}) for job={job_id}"
            )
            logger.info(f"Recovery started: {datetime.now(timezone.utc)}")
            logger.info(
                "Recovery: relaunched worker loads checkpoints under run_id=%s "
                "checkpoint_dir=%s (find_latest_checkpoint / restore_checkpoint)",
                run_id,
                checkpoint_dir,
            )

            root = str(_PROJECT_ROOT)
            env = os.environ.copy()
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = root if not pp else f"{root}{os.pathsep}{pp}"

            base_url = self.config.coordinator_base_url.rstrip("/")

            cmd = [
                sys.executable,
                "-m",
                "src.worker.runtime",
                "--rank",
                str(failed_rank),
                "--world-size",
                str(world_size),
                "--job-id",
                job_id,
                "--run-id",
                run_id,
                "--coordinator-url",
                base_url,
                "--backend",
                "gloo",
                "--total-steps",
                str(total_steps),
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
            if bool(meta.get("deterministic", False)):
                cmd.append("--deterministic")

            subprocess.Popen(cmd, cwd=root, env=env)

            job_info.recovery_attempts += 1
            job_info.updated_at = datetime.now(timezone.utc)

            # Track when recovery started (first attempt only)
            if job_info.recovery_started_at is None:
                job_info.recovery_started_at = datetime.now(timezone.utc)

            # Check if max attempts exceeded
            if job_info.recovery_attempts >= self.config.max_recovery_attempts:
                logger.error(
                    f"Max recovery attempts ({self.config.max_recovery_attempts}) "
                    f"exceeded for job {job_id}. Marking as FAILED."
                )
                state_machine.fail()
                job_info.status = JobState.FAILED
                job_info.error_message = f"Max recovery attempts exceeded"
                job_info.completed_at = datetime.now(timezone.utc)
                return

            # Schedule recovery check after backoff period
            logger.warning(
                f"Job {job_id} entering recovery (attempt {job_info.recovery_attempts}, "
                f"{len(alive_workers)}/{len(all_workers)} workers alive)"
            )

            # Schedule async task to check recovery status after backoff
            asyncio.create_task(
                self._check_recovery_status(job_id, job_info.recovery_attempts)
            )

    async def _check_recovery_status(self, job_id: str, attempt_number: int) -> None:
        """
        Check recovery status after backoff period and decide to resume or fail.

        This runs after RECOVERY_BACKOFF_SECONDS to give workers time to:
        1. Re-register after relaunch
        2. Restore from checkpoint on the worker (same run_id in job metadata)
        3. Rejoin the training process

        Current behavior:
        - If alive workers still exist → resume job
        - If all workers lost → fail job
        - If timeout exceeded → fail job
        """
        await asyncio.sleep(self.config.recovery_backoff_seconds)

        async with self._jobs_lock:
            if job_id not in self._jobs:
                return

            job_info, state_machine = self._jobs[job_id]

            # Skip if job already exited RECOVERING state
            if state_machine.get_current_state() != JobState.RECOVERING:
                return

            # Skip if this is an old recovery attempt (newer one happened)
            if job_info.recovery_attempts != attempt_number:
                return

            # Check worker status
            all_workers = await self.worker_registry.get_workers_by_job(job_id)
            alive_workers = [w for w in all_workers if w.status != WorkerStatus.LOST]

            if len(alive_workers) == 0:
                # All workers died during recovery
                logger.error(
                    f"All workers lost during recovery for job {job_id}. Failing."
                )
                state_machine.fail()
                job_info.status = JobState.FAILED
                job_info.error_message = "All workers lost during recovery"
                job_info.completed_at = datetime.now(timezone.utc)
                return

            # Check recovery timeout (max_attempts * backoff = total time limit)
            recovery_started = job_info.recovery_started_at or job_info.updated_at
            recovery_duration = (
                datetime.now(timezone.utc) - recovery_started
            ).total_seconds()
            max_recovery_time = (
                self.config.max_recovery_attempts * self.config.recovery_backoff_seconds
            )

            if recovery_duration > max_recovery_time:
                logger.error(
                    f"Recovery timeout for job {job_id} "
                    f"({recovery_duration:.0f}s > {max_recovery_time}s). Failing."
                )
                state_machine.fail()
                job_info.status = JobState.FAILED
                job_info.error_message = f"Recovery timeout ({recovery_duration:.0f}s)"
                job_info.completed_at = datetime.now(timezone.utc)
                return

            logger.info(
                f"Resuming job {job_id} with {len(alive_workers)}/{len(all_workers)} workers "
                f"after recovery attempt {attempt_number}"
            )
            logger.info(
                "Training resumed (coordinator): job %s back to RUNNING; "
                "workers continue from checkpoint if present for run_id in metadata",
                job_id,
            )

            try:
                state_machine.resume()
                job_info.status = JobState.RUNNING
                job_info.updated_at = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(f"Failed to resume job {job_id}: {e}")
                state_machine.fail()
                job_info.status = JobState.FAILED
                job_info.error_message = f"Resume failed: {e}"
                job_info.completed_at = datetime.now(timezone.utc)

    async def _process_metrics_from_heartbeat(self, heartbeat: HeartbeatPayload) -> None:
        """Aggregate metrics by job and step.

        Rules:
        - loss: average across workers that reported loss for the step
        - throughput: sum across workers for the step
        - workers_reported: number of distinct workers reporting for the step

        The metrics_store layout is described in __init__.
        """
        # find worker and job
        worker = await self.worker_registry.get_worker(heartbeat.worker_id)
        if not worker:
            # Unknown worker — nothing to aggregate
            return

        job_id = worker.job_id
        metrics = heartbeat.metrics or {}

        # step must be present to perform per-step aggregation
        step = metrics.get("step")

        # ensure job store exists
        if job_id not in self._metrics_store:
            self._metrics_store[job_id] = {"steps": {}, "run_summary": {"latest_step": None, "latest_loss": None, "best_loss": None, "avg_throughput": None}}

        job_store = self._metrics_store[job_id]
        steps_store: Dict[int, Dict[str, Any]] = job_store["steps"]

        if step is None:
            # No per-step info: update nothing further (we do not append raw entries)
            return

        # normalize step to int (if possible)
        try:
            step_key = int(step)
        except Exception:
            # non-integer step — ignore for aggregation
            return

        # ensure step store exists
        if step_key not in steps_store:
            steps_store[step_key] = {"worker_metrics": {}, "aggregated": {"avg_loss": None, "total_throughput": 0.0, "workers_reported": 0}}

        step_entry = steps_store[step_key]
        worker_metrics: Dict[str, Dict[str, Any]] = step_entry["worker_metrics"]

        # update worker's metrics for this step (replace previous if present)
        worker_metrics[heartbeat.worker_id] = metrics

        # recompute aggregated values for this step from worker_metrics
        losses = []
        total_throughput = 0.0

        for wm in worker_metrics.values():
            # loss: may be missing
            try:
                if wm.get("loss") is not None:
                    losses.append(float(wm.get("loss")))
            except Exception:
                # ignore malformed loss
                pass

            try:
                if wm.get("throughput") is not None:
                    total_throughput += float(wm.get("throughput"))
            except Exception:
                # ignore malformed throughput
                pass

        workers_reported = len(worker_metrics)
        avg_loss = (sum(losses) / len(losses)) if losses else None

        step_entry["aggregated"] = {
            "avg_loss": avg_loss,
            "total_throughput": total_throughput,
            "workers_reported": workers_reported,
        }

        # update run summary using aggregated step values (not raw entries)
        run_summary = job_store["run_summary"]

        # latest_step: choose max seen step
        prev_latest = run_summary.get("latest_step")
        if prev_latest is None or step_key >= prev_latest:
            run_summary["latest_step"] = step_key
            run_summary["latest_loss"] = avg_loss

        # best_loss: min across aggregated step avg_loss values (skip None)
        best = None
        tot_throughputs = []
        for sdata in steps_store.values():
            agg = sdata.get("aggregated", {})
            al = agg.get("avg_loss")
            if al is not None:
                if best is None or al < best:
                    best = al
            tt = agg.get("total_throughput")
            if tt is not None:
                tot_throughputs.append(tt)

        run_summary["best_loss"] = best

        # avg_throughput across steps: average of per-step total_throughput
        if tot_throughputs:
            run_summary["avg_throughput"] = sum(tot_throughputs) / len(tot_throughputs)
        else:
            run_summary["avg_throughput"] = None

        # commit back
        job_store["run_summary"] = run_summary

    def _on_job_state_transition(
        self, job_id: str, from_state: str, to_state: str
    ) -> None:
        """
        Callback invoked on job state transitions.
        Broadcasts state changes via WebSocket to connected clients.
        """
        logger.info(f"Job {job_id} state transition: {from_state} → {to_state}")

        # Broadcast state change via WebSocket
        if self.ws_manager:
            state_data = {
                "old_state": from_state,
                "new_state": to_state,
                "transition_time": datetime.now(timezone.utc).isoformat(),
            }

            # Schedule broadcast
            asyncio.create_task(
                self.ws_manager.broadcast_state_update(job_id, state_data)
            )

    def __repr__(self) -> str:
        return f"Coordinator(jobs={len(self._jobs)}, config={self.config})"
    
    async def handle_checkpoint_complete(self, payload: dict) -> None:
        """
        Called when a worker reports checkpoint completion.
        Aggregates worker shards and generates manifest
        once all workers for a job + step have reported.
        """

        step = payload["step"]
        run_id = payload["run_id"]
        job_id = payload["job_id"]

        key = (job_id, run_id, step)

        if key not in self._checkpoint_reports:
            self._checkpoint_reports[key] = []

        self._checkpoint_reports[key].append(payload)

        # Determine expected worker count from registry (source of truth)
        workers = await self.worker_registry.get_workers_by_job(job_id)
        expected_world_size = len(workers)

        if len(self._checkpoint_reports[key]) == expected_world_size:
            await self._generate_manifest(job_id, run_id, step, key)

    async def _generate_manifest(
        self,
        job_id: str,
        run_id: str,
        step: int,
        key: tuple
    ) -> None:
        """
        Build and persist manifest.json for a completed checkpoint step.
        """

        worker_reports = self._checkpoint_reports.get(key, [])
        world_size = len(worker_reports)

        worker_shards = []
        for report in worker_reports:
            worker_shards.append({
                "worker_id": report["worker_id"],
                "rank": report["rank"],
                "path": report["checkpoint_path"],
                "file_size_bytes": report["file_size_bytes"],
                "sha256": report["sha256"],
            })

        manifest = build_manifest(
            step=step,
            run_id=run_id,
            job_id=job_id,
            world_size=world_size,
            worker_shards=worker_shards,
        )

        output_dir = os.path.join(
            "checkpoints",
            run_id,
            f"ckpt_step_{step:06d}"
        )

        manifest_path = save_manifest(manifest, output_dir)

        logger.info(f"Manifest generated: {manifest_path}")

        # Clean up memory for this checkpoint
        del self._checkpoint_reports[key]
