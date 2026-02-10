"""
Local Worker Launcher
Spawns multiple worker processes for local testing without a full cluster.
"""

import subprocess
import time
import signal
import sys
import uuid
from typing import List


class WorkerLauncher:
    """Manages multiple worker processes for local testing"""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.job_id = str(uuid.uuid4())[:8]
        self.run_id = str(uuid.uuid4())[:8]

    def launch_workers(
        self,
        world_size: int = 2,
        backend: str = "gloo",
        coordinator_url: str = "http://localhost:8000",
        total_steps: int = 100,
    ):
        """Launch multiple worker processes"""

        print(f"🚀 Launching {world_size} workers")
        print(f"   Job ID: {self.job_id}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Backend: {backend}")
        print(f"   Coordinator: {coordinator_url}")
        print("-" * 60)

        # Launch each worker process
        for rank in range(world_size):
            cmd = [
                sys.executable,
                "-m",
                "runtime",
                "--rank",
                str(rank),
                "--world-size",
                str(world_size),
                "--job-id",
                self.job_id,
                "--run-id",
                self.run_id,
                "--coordinator-url",
                coordinator_url,
                "--backend",
                backend,
                "--total-steps",
                str(total_steps),
            ]

            print(f"Starting worker {rank}...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self.processes.append(process)
            time.sleep(0.5)  # Stagger launches

        print(f"✅ All {world_size} workers launched")
        print("-" * 60)

    def monitor(self):
        """Monitor worker output"""
        print("📊 Monitoring workers (Ctrl+C to stop)...\n")

        try:
            # Simple round-robin output reading
            while any(p.poll() is None for p in self.processes):
                for i, process in enumerate(self.processes):
                    if process.poll() is None and process.stdout:
                        line = process.stdout.readline()
                        if line:
                            print(f"[Worker {i}] {line.rstrip()}")
                time.sleep(0.01)

            # Check exit codes
            print("\n" + "=" * 60)
            for i, process in enumerate(self.processes):
                code = process.poll()
                if code == 0:
                    print(f"✅ Worker {i} completed successfully")
                else:
                    print(f"❌ Worker {i} exited with code {code}")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown all workers"""
        print("\n🛑 Shutting down workers...")

        for i, process in enumerate(self.processes):
            if process.poll() is None:
                print(f"   Stopping worker {i}...")
                process.terminate()

        # Wait for graceful shutdown
        time.sleep(2)

        # Force kill if needed
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                print(f"   Force killing worker {i}...")
                process.kill()

        print("✅ All workers stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch multiple DistroML workers locally"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of workers (world size)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend",
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default="http://localhost:8000",
        help="Coordinator API URL",
    )
    parser.add_argument("--steps", type=int, default=100, help="Total training steps")

    args = parser.parse_args()

    launcher = WorkerLauncher()

    try:
        launcher.launch_workers(
            world_size=args.workers,
            backend=args.backend,
            coordinator_url=args.coordinator_url,
            total_steps=args.steps,
        )
        launcher.monitor()
    except Exception as e:
        print(f"❌ Error: {e}")
        launcher.shutdown()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
