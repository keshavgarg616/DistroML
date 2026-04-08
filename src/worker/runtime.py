"""
DistroML Worker Runtime
Implements worker registration, heartbeat emission, and stubbed training loop.
"""

import os
import sys
import random
import time
import socket
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import requests
from time import perf_counter

import hashlib



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
    step_time: float  # seconds per step
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

        # Failure injection config
        self.kill_at_step = None
        self.pause_at_step = None
        self.pause_duration = 0
        self.drop_heartbeat_rate = 0.0

        # Initialize dummy model/optimizer for checkpointing
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # WebSocket streaming thread for real-time metrics/logs
        self.ws_streamer = None
        self._enable_websocket = True  # Can be disabled for testing/fallback

        logger.info(
            f"Worker initialized: id={config.worker_id}, "
            f"rank={config.rank}/{config.world_size}, "
            f"backend={config.backend}"
        )

    def configure_failure_injection(
        self,
        kill_at_step: Optional[int] = None,
        pause_at_step: Optional[int] = None,
        pause_duration: int = 0,
        drop_heartbeat_rate: float = 0.0,
    ):
        """Configure failure injection parameters"""
        self.kill_at_step = kill_at_step
        self.pause_at_step = pause_at_step
        self.pause_duration = pause_duration
        self.drop_heartbeat_rate = drop_heartbeat_rate
        logger.info(
            f"Failure injection configured: kill_at={kill_at_step}, "
            f"pause_at={pause_at_step} ({pause_duration}s), "
            f"drop_rate={drop_heartbeat_rate}"
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
            "host": self.hostname,
            "port": int(os.environ.get("MASTER_PORT", "29500")),
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

    def notify_training_complete(self) -> None:
        """Tell the coordinator this worker finished training so it is not marked LOST."""
        url = f"{self.config.coordinator_url.rstrip('/')}/api/workers/complete"
        delays = (0.0, 0.5, 1.0)
        last_err: Optional[str] = None
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                response = requests.post(
                    url,
                    json={"worker_id": self.config.worker_id},
                    timeout=10,
                )
                if response.status_code == 200:
                    logger.info("Notified coordinator of clean training shutdown")
                    return
                last_err = f"HTTP {response.status_code} {response.text}"
                logger.warning(
                    f"Complete notification attempt {attempt + 1}/{len(delays)}: {last_err}"
                )
            except requests.exceptions.RequestException as e:
                last_err = str(e)
                logger.warning(
                    f"Complete notification attempt {attempt + 1}/{len(delays)}: {e}"
                )
        logger.warning(
            f"Could not notify coordinator of clean shutdown after retries: {last_err}"
        )

    def _emit_heartbeat(self):
        """Send heartbeat to Coordinator"""
        # Failure Injection: Drop Heartbeat
        if self.drop_heartbeat_rate > 0 and random.random() < self.drop_heartbeat_rate:
            logger.warning("💓 INJECTED FAILURE: Dropping heartbeat")
            return

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

    def start_websocket_streaming(self):
        """
        Initialize and start WebSocket streaming thread.

        This runs WebSocket operations in a background thread, allowing
        synchronous training code to stream metrics without blocking.
        """
        if not self._enable_websocket:
            logger.info("WebSocket streaming disabled")
            return

        try:
            from src.worker.websocket_thread import WebSocketStreamingThread

            self.ws_streamer = WebSocketStreamingThread(
                coordinator_url=self.config.coordinator_url,
                job_id=self.config.job_id,
                worker_id=self.config.worker_id
            )
            self.ws_streamer.start()
            logger.info("WebSocket streaming started in background thread")
        except ImportError:
            logger.warning("websockets library not installed. Falling back to HTTP only.")
            self.ws_streamer = None
        except Exception as e:
            logger.warning(f"Failed to start WebSocket streaming: {e}. Falling back to HTTP only.")
            self.ws_streamer = None

    def stop_websocket_streaming(self):
        if self.ws_streamer:
            try:
                self.ws_streamer.stop(timeout=5.0)
                logger.info("WebSocket streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping WebSocket streamer: {e}")

    def emit_metrics(self, metrics: WorkerMetrics):
        metrics_data = asdict(metrics)
        metrics_data["worker_id"] = self.config.worker_id
        metrics_data["rank"] = self.config.rank

        logger.debug(f"emit_metrics called for step {metrics.step}")

        # Try WebSocket first (non-blocking, thread-safe)
        if self.ws_streamer and self.ws_streamer.is_streaming():
            # Queue message for WebSocket thread to send
            success = self.ws_streamer.send_metrics(metrics_data)
            if not success:
                logger.warning(f"WebSocket queue full, using HTTP fallback for step {metrics.step}")
                # Queue full, use HTTP fallback
                self._emit_metrics_http(metrics_data)
        else:
            # WebSocket not available, use HTTP
            logger.debug(f"WebSocket not active, using HTTP fallback")
            self._emit_metrics_http(metrics_data)

    def _emit_metrics_http(self, metrics_data: dict):
        """
        HTTP fallback for metrics emission.
        """
        try:
            url = f"{self.config.coordinator_url}/api/jobs/{self.config.job_id}/metrics"
            requests.post(url, json=metrics_data, timeout=5)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to emit metrics via HTTP: {e}")

    def stream_log(self, level: str, message: str, **kwargs):
        """
        Stream a log message via WebSocket.

        Thread-safe: Can be called from synchronous code.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message
            **kwargs: Additional context
        """
        if self.ws_streamer and self.ws_streamer.is_streaming():
            try:
                self.ws_streamer.send_log(level, message, **kwargs)
            except Exception as e:
                logger.debug(f"Failed to stream log: {e}")

    def sync_gradients(self, model: nn.Module):
        """
        Placeholder for gradient synchronization across workers.
        This will be implemented with either DDP or manual AllReduce.

        Args:
            model: The neural network model with gradients to sync
        """
        # TO BE IMPLEMENTED
        pass

    def train_step(
        self,
        model: nn.Module,
        batch: tuple,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> tuple[float, float]:
        """
        Execute one training step: forward → loss → backward → sync → optimizer step.

        Args:
            model: Neural network model
            batch: Tuple of (inputs, targets)
            criterion: Loss function
            optimizer: Optimizer (e.g., SGD, Adam)

        Returns:
            tuple: (loss_value, step_time) where step_time is in seconds
        """
        step_start = perf_counter()

        inputs, targets = batch

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        # Gradient synchronization (placeholder)
        self.sync_gradients(model)

        optimizer.step()

        step_time = perf_counter() - step_start

        return loss.item(), step_time

    def train_step_stub(self, step: int) -> WorkerMetrics:
        """
        Stubbed training step for MVP testing.
        In production, this would call train_step() with real model/data.
        """
        step_start = perf_counter()

        # Simulate some work
        time.sleep(0.1)

        # Simulate gradient allreduce
        if self.dist_initialized:
            dummy_tensor = torch.randn(100)
            dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM)

        # Calculate step time
        step_time = perf_counter() - step_start

        # Generate fake metrics
        fake_loss = 2.5 - (step * 0.001)  # Decreasing loss
        fake_throughput = 450 + (step % 50)  # Varying throughput

        metrics = WorkerMetrics(
            step=step,
            epoch=self.current_epoch,
            loss=max(0.1, fake_loss),
            step_time=step_time,
            throughput=fake_throughput,
            timestamp=time.time(),
        )

        return metrics
        
    def training_loop(self, total_steps: int = 100, steps_per_epoch: int = 50, start_step: int = 0):
    
        if start_step > 0:
            logger.info(f"Resuming training from step {start_step} to {total_steps}")
            logger.info(f"Training resumed: {datetime.now(timezone.utc)}")
        else:
            logger.info(f"Starting training loop: {total_steps} steps")

        self.state = WorkerState.TRAINING
        self.is_running = True

        try:
            # Start from start_step + 1 (next step to train) or 1 if fresh start
            for step in range(max(1, start_step + 1), total_steps + 1):
                # Failure Injection: Kill
                if self.kill_at_step == step:
                    logger.critical(f"💥 INJECTED FAILURE: Killing worker at step {step}")
                    sys.exit(1)

                # Failure Injection: Pause
                if self.pause_at_step == step:
                    logger.warning(f"⏸️ INJECTED FAILURE: Pausing worker for {self.pause_duration}s")
                    time.sleep(self.pause_duration)

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
                        f"Step time: {metrics.step_time:.3f}s | "
                        f"Throughput: {metrics.throughput:.1f} samples/s"
                    )

                # Checkpoint logic (stubbed)
                if step % 200 == 0:
                    logger.info(f"Checkpoint triggered at step {step}")
                    self.save_checkpoint(step)

            logger.info("Training loop completed successfully")
            self.state = WorkerState.STOPPED

        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            self.state = WorkerState.FAILED
            raise
    
    def _compute_sha256(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def find_latest_checkpoint(self) -> Optional[tuple]:
        from src.coordinator.manifest import load_manifest

        checkpoint_base = os.path.join(
            self.config.checkpoint_dir,
            f"{self.config.run_id}"
        )

        if not os.path.exists(checkpoint_base):
            logger.info(f"No checkpoint directory found at {checkpoint_base}")
            return None

        checkpoint_dirs = []
        for entry in os.listdir(checkpoint_base):
            if entry.startswith("ckpt_step_"):
                try:
                    step_str = entry.replace("ckpt_step_", "")
                    step = int(step_str)
                    full_path = os.path.join(checkpoint_base, entry)
                    checkpoint_dirs.append((step, full_path))
                except ValueError:
                    continue

        if not checkpoint_dirs:
            logger.info("No checkpoint directories found")
            return None

        checkpoint_dirs.sort(reverse=True, key=lambda x: x[0])

        for step, checkpoint_dir in checkpoint_dirs:
            try:
                manifest = load_manifest(checkpoint_dir)
                logger.info(f"Found valid checkpoint at step {step}")
                return (checkpoint_dir, manifest)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Checkpoint at step {step} has no valid manifest: {e}")
                continue

        logger.info("No valid checkpoints found (no manifests)")
        return None

    def restore_checkpoint(self, checkpoint_dir: str, manifest: dict) -> int:

        if manifest["world_size"] != self.config.world_size:
            raise ValueError(
                f"Checkpoint world_size ({manifest['world_size']}) "
                f"doesn't match current world_size ({self.config.world_size})"
            )

        worker_shard = None
        for shard in manifest["worker_shards"]:
            if shard["rank"] == self.config.rank:
                worker_shard = shard
                break

        if worker_shard is None:
            raise ValueError(
                f"Rank {self.config.rank} not found in checkpoint manifest"
            )

        shard_path = worker_shard["path"]

        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Checkpoint shard not found: {shard_path}")

        logger.info(f"Loading checkpoint from {shard_path}")
        checkpoint = torch.load(shard_path, map_location="cpu")

        if checkpoint["rank"] != self.config.rank:
            raise ValueError(
                f"Checkpoint rank ({checkpoint['rank']}) "
                f"doesn't match worker rank ({self.config.rank})"
            )

        self.model.load_state_dict(checkpoint["model_state"])
        logger.info("Model state restored")

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info("Optimizer state restored")

        torch.set_rng_state(checkpoint["rng_state"])
        if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        logger.info("RNG state restored")

        self.current_epoch = checkpoint["epoch"]
        restored_step = checkpoint["step"]

        logger.info(
            f"Checkpoint restored successfully from step {restored_step}, "
            f"epoch {self.current_epoch}"
        )

        return restored_step + 1

    def save_checkpoint(self, step: int):
        """
        Save checkpoint with model, optimizer, and RNG state.
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

        # Save full state
        state = {
            "step": step,
            "epoch": self.current_epoch,
            "rank": self.config.rank,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        torch.save(state, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.state = WorkerState.TRAINING


def main():
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description="DistroML Worker Runtime")
    parser.add_argument("--rank", type=int, required=True, help="Worker rank")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of workers")
    parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default="http://localhost:8000",
        help="Coordinator API URL",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend",
    )
    parser.add_argument("--total-steps", type=int, default=100, help="Total training steps")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory"
    )

    args = parser.parse_args()

    # Create worker configuration
    worker_id = f"worker_{args.rank}_{uuid.uuid4().hex[:8]}"
    config = WorkerConfig(
        worker_id=worker_id,
        rank=args.rank,
        world_size=args.world_size,
        coordinator_url=args.coordinator_url,
        job_id=args.job_id,
        run_id=args.run_id,
        backend=args.backend,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create and run worker
    worker = WorkerRuntime(config)

    try:
        # Step 1: Register with coordinator
        logger.info(f"Worker {args.rank} starting registration...")
        worker.register()

        # Step 2: Start WebSocket streaming for real-time metrics/logs
        logger.info("Starting WebSocket streaming...")
        worker.start_websocket_streaming()

        # Step 3: Check for checkpoint and restore if found
        start_step = 0
        checkpoint_info = worker.find_latest_checkpoint()

        if checkpoint_info:
            checkpoint_dir, manifest = checkpoint_info
            logger.info(f"Found checkpoint at step {manifest['step']}, restoring...")
            start_step = worker.restore_checkpoint(checkpoint_dir, manifest)
            logger.info(f"Restored from checkpoint, resuming from step {start_step}")
        else:
            logger.info("No checkpoint found, starting fresh")

        # Step 4: Initialize distributed training
        logger.info("Initializing distributed training...")
        distributed_initialised = worker.init_distributed()
        if not distributed_initialised:
            logger.error("Failed to initialize distributed training")
            sys.exit(1)
        # Step 5: Synchronize all workers after restore (barrier)
        if config.world_size > 1:
            logger.info("Waiting for all workers to reach barrier...")
            dist.barrier()
            logger.info("All workers synchronized")

        # Step 6: Start heartbeat thread
        logger.info("Starting heartbeat thread...")
        worker.start_heartbeat()

        # Step 7: Run training loop
        logger.info("Starting training...")
        worker.training_loop(total_steps=args.total_steps, start_step=start_step)

        worker.notify_training_complete()
        logger.info("Worker completed successfully")

    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        worker.state = WorkerState.FAILED
        sys.exit(1)

    finally:
        # Cleanup
        worker.stop_heartbeat()
        worker.stop_websocket_streaming()
        if worker.dist_initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()