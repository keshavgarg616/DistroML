"""
DistroML Worker Runtime
Implements worker registration, heartbeat emission, and stubbed training loop.
"""

import os
import time
import socket
import logging
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json

import torch
import torch.distributed as dist
import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker lifecycle states"""

    INITIALIZING = "initializing"
    REGISTERED = "registered"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    RECOVERING = "recovering"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class WorkerConfig:
    """Worker configuration"""

    worker_id: str
    rank: int
    world_size: int
    coordinator_url: str
    job_id: str
    run_id: str
    backend: str = "gloo"  # gloo for CPU, nccl for GPU
    heartbeat_interval: int = 5  # seconds
    metrics_interval: int = 10  # seconds
    checkpoint_dir: str = "./checkpoints"


@dataclass
class WorkerMetrics:
    """Training metrics emitted by worker"""

    step: int
    epoch: int
    loss: float
    throughput: float  # samples/sec
    timestamp: float


class WorkerRuntime:
    """
    Core Worker Runtime
    Handles registration, heartbeat, and training coordination.
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.state = WorkerState.INITIALIZING
        self.hostname = socket.gethostname()
        self.pid = os.getpid()

        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.is_running = False

        # Threading
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Distributed context
        self.dist_initialized = False

        logger.info(
            f"Worker initialized: id={config.worker_id}, "
            f"rank={config.rank}/{config.world_size}, "
            f"backend={config.backend}"
        )

    def register(self) -> bool:
        """
        Register worker with Coordinator.
        Returns True if registration successful.
        """
        registration_data = {
            "worker_id": self.config.worker_id,
            "rank": self.config.rank,
            "world_size": self.config.world_size,
            "job_id": self.config.job_id,
            "run_id": self.config.run_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "backend": self.config.backend,
            "resources": {
                "cpus": os.cpu_count(),
                "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
        }

        try:
            url = f"{self.config.coordinator_url}/api/workers/register"
            response = requests.post(url, json=registration_data, timeout=10)

            if response.status_code == 200:
                self.state = WorkerState.REGISTERED
                logger.info(f"Worker {self.config.worker_id} registered successfully")
                return True
            else:
                logger.error(
                    f"Registration failed: {response.status_code} - {response.text}"
                )
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Registration request failed: {e}")
            return False

    def _emit_heartbeat(self):
        """Send heartbeat to Coordinator"""
        heartbeat_data = {
            "worker_id": self.config.worker_id,
            "rank": self.config.rank,
            "state": self.state.value,
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "timestamp": time.time(),
        }

        try:
            url = f"{self.config.coordinator_url}/api/workers/heartbeat"
            response = requests.post(url, json=heartbeat_data, timeout=5)

            if response.status_code != 200:
                logger.warning(f"Heartbeat failed: {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Heartbeat request failed: {e}")

    def _heartbeat_loop(self):
        """Background thread for periodic heartbeat emission"""
        logger.info(
            f"Heartbeat loop started (interval={self.config.heartbeat_interval}s)"
        )

        while not self.stop_event.is_set():
            self._emit_heartbeat()
            self.stop_event.wait(self.config.heartbeat_interval)

        logger.info("Heartbeat loop stopped")

    def start_heartbeat(self):
        """Start heartbeat background thread"""
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.stop_event.clear()
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self.heartbeat_thread.start()
            logger.info("Heartbeat thread started")

    def stop_heartbeat(self):
        """Stop heartbeat background thread"""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.stop_event.set()
            self.heartbeat_thread.join(timeout=10)
            logger.info("Heartbeat thread stopped")

    def init_distributed(self) -> bool:
        """
        Initialize PyTorch distributed context.
        Uses environment variables or manual setup.
        """
        try:
            # Check if already initialized
            if dist.is_initialized():
                logger.info("Distributed context already initialized")
                self.dist_initialized = True
                return True

            # Determine backend
            backend = self.config.backend
            if backend == "auto":
                backend = "nccl" if torch.cuda.is_available() else "gloo"

            # Initialize process group
            # In production, these would come from environment or config
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            dist.init_process_group(
                backend=backend,
                rank=self.config.rank,
                world_size=self.config.world_size,
            )

            self.dist_initialized = True
            logger.info(
                f"Distributed initialized: backend={backend}, "
                f"rank={self.config.rank}, world_size={self.config.world_size}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize distributed context: {e}")
            self.state = WorkerState.FAILED
            return False

    def emit_metrics(self, metrics: WorkerMetrics):
        """Send training metrics to Coordinator"""
        metrics_data = asdict(metrics)
        metrics_data["worker_id"] = self.config.worker_id
        metrics_data["rank"] = self.config.rank

        try:
            url = f"{self.config.coordinator_url}/api/jobs/{self.config.job_id}/metrics"
            requests.post(url, json=metrics_data, timeout=5)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to emit metrics: {e}")

    def train_step_stub(self, step: int) -> WorkerMetrics:
        """
        Stubbed training step for MVP testing.
        In production, this would call the framework adapter.
        """
        # Simulate some work
        time.sleep(0.1)

        # Simulate gradient allreduce
        if self.dist_initialized:
            dummy_tensor = torch.randn(100)
            dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM)

        # Generate fake metrics
        fake_loss = 2.5 - (step * 0.001)  # Decreasing loss
        fake_throughput = 450 + (step % 50)  # Varying throughput

        metrics = WorkerMetrics(
            step=step,
            epoch=self.current_epoch,
            loss=max(0.1, fake_loss),
            throughput=fake_throughput,
            timestamp=time.time(),
        )

        return metrics

    def training_loop(self, total_steps: int = 100, steps_per_epoch: int = 50):
        """
        Main training loop (stubbed for MVP).
        Demonstrates step execution, metric emission, and checkpoint logic.
        """
        logger.info(f"Starting training loop: {total_steps} steps")
        self.state = WorkerState.TRAINING
        self.is_running = True

        try:
            for step in range(1, total_steps + 1):
                if not self.is_running:
                    logger.info("Training stopped by external signal")
                    break

                # Update epoch
                self.current_epoch = (step - 1) // steps_per_epoch
                self.current_step = step

                # Execute training step
                metrics = self.train_step_stub(step)

                # Emit metrics periodically
                if step % (self.config.metrics_interval) == 0:
                    self.emit_metrics(metrics)
                    logger.info(
                        f"Step {step}/{total_steps} | "
                        f"Epoch {self.current_epoch} | "
                        f"Loss: {metrics.loss:.4f} | "
                        f"Throughput: {metrics.throughput:.1f} samples/s"
                    )

                # Checkpoint logic (stubbed)
                if step % 200 == 0:
                    logger.info(f"Checkpoint triggered at step {step}")
                    self._save_checkpoint_stub(step)

            logger.info("Training loop completed successfully")
            self.state = WorkerState.STOPPED

        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            self.state = WorkerState.FAILED
            raise

    def _save_checkpoint_stub(self, step: int):
        """
        Stubbed checkpoint save.
        In production, this would serialize model/optimizer state.
        """
        self.state = WorkerState.CHECKPOINTING

        # Simulate checkpoint save
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.config.run_id}",
            f"ckpt_step_{step:06d}",
            f"worker_{self.config.rank}.pt",
        )

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Write dummy checkpoint
        checkpoint_data = {
            "step": step,
            "epoch": self.current_epoch,
            "rank": self.config.rank,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.state = WorkerState.TRAINING

    def shutdown(self):
        """Clean shutdown of worker"""
        logger.info("Shutting down worker...")
        self.is_running = False
        self.stop_heartbeat()

        if self.dist_initialized:
            dist.destroy_process_group()
            logger.info("Distributed context destroyed")

        logger.info("Worker shutdown complete")


def main():
    """
    Example worker launch.
    In production, config would come from Coordinator via API or env vars.
    """
    import argparse

    parser = argparse.ArgumentParser(description="DistroML Worker")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--coordinator-url", type=str, default="http://localhost:8000")
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--total-steps", type=int, default=100)

    args = parser.parse_args()

    # Create worker config
    config = WorkerConfig(
        worker_id=f"worker_{args.rank}",
        rank=args.rank,
        world_size=args.world_size,
        job_id=args.job_id,
        run_id=args.run_id,
        coordinator_url=args.coordinator_url,
        backend=args.backend,
    )

    # Initialize worker
    worker = WorkerRuntime(config)

    try:
        # Registration
        if not worker.register():
            logger.error("Registration failed, exiting")
            return 1

        # Start heartbeat
        worker.start_heartbeat()

        # Initialize distributed
        if not worker.init_distributed():
            logger.error("Distributed initialization failed, exiting")
            return 1

        # Run training
        worker.training_loop(total_steps=args.total_steps)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        return 1
    finally:
        worker.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
