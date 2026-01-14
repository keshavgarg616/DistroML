from enum import Enum

class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class WorkerStatus(str, Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    LOST = "LOST"


HEARTBEAT_TIMEOUT_SECONDS = 30
HEARTBEAT_CHECK_INTERVAL_SECONDS = 5
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_BACKOFF_SECONDS = 10
WORKER_REGISTRATION_TIMEOUT_SECONDS = 120


class DistroMLError(Exception):
    pass


class CoordinatorError(DistroMLError):
    pass


class WorkerNotFoundError(CoordinatorError):
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        super().__init__(f"Worker not found: {worker_id}")


class JobNotFoundError(CoordinatorError):
    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}")


class InvalidStateTransitionError(CoordinatorError):
    def __init__(self, from_state: str, to_state: str):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid state transition: {from_state} → {to_state}")
