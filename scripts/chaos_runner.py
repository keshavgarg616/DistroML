"""
Week 6: Local chaos runner (random failure injection + time-to-recovery).

Runs:
1) Starts the coordinator (uvicorn) as a subprocess
2) Submits a job
3) Launches N local workers with failure-injection flags
4) Polls job status and measures:
   - failure_detected_time (first RECOVERING observation)
   - resumed_time (first RUNNING after RECOVERING)
   - time_to_recovery (seconds)

This is a lightweight harness intended for laptop MVP use.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

import requests


def _wait_for_http_ready(base_url: str, timeout_s: int = 20) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/api/jobs", timeout=2)
            if r.status_code in (200, 404):
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("Coordinator did not become ready in time")


def _submit_job(base_url: str, world_size: int, total_steps: int) -> Tuple[str, str]:
    payload = {
        "name": "chaos-run",
        "world_size": world_size,
        "metadata": {
            "world_size": world_size,
            "total_steps": total_steps,
            "checkpoint_dir": "./checkpoints",
        },
    }
    r = requests.post(f"{base_url.rstrip('/')}/api/jobs", json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data["job_id"], data.get("run_id", "unknown")


def _poll_recovery(base_url: str, job_id: str, timeout_s: int = 120) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    first_recovering: Optional[float] = None
    first_resumed: Optional[float] = None

    while time.time() < deadline:
        r = requests.get(f"{base_url.rstrip('/')}/api/jobs/{job_id}", timeout=5)
        if r.status_code == 404:
            time.sleep(0.5)
            continue
        r.raise_for_status()
        status = r.json()
        st = status.get("status")

        now = time.time()
        if st == "RECOVERING" and first_recovering is None:
            first_recovering = now
        if first_recovering is not None and st == "RUNNING" and first_resumed is None:
            first_resumed = now
            break

        if st in ("FAILED", "COMPLETED", "CANCELLED"):
            break

        time.sleep(0.5)

    return {
        "job_id": job_id,
        "first_recovering_time": first_recovering,
        "first_resumed_time": first_resumed,
        "time_to_recovery_s": (first_resumed - first_recovering) if (first_resumed and first_recovering) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DistroML chaos runner (Week 6)")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    # Choose a simple injected failure: kill one worker at a random step.
    kill_step = random.randint(10, max(10, args.steps // 2))

    print("=== DistroML Chaos Runner (Week 6) ===")
    print(f"Coordinator URL: {args.coordinator_url}")
    print(f"Workers: {args.workers}")
    print(f"Total steps: {args.steps}")
    print(f"Injected failure: kill_at_step={kill_step}")
    print()

    env = os.environ.copy()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    coord_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.coordinator.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    coordinator_proc: Optional[subprocess.Popen] = None
    try:
        coordinator_proc = subprocess.Popen(coord_cmd, cwd=root, env=env)
        _wait_for_http_ready(args.coordinator_url)

        job_id, run_id = _submit_job(args.coordinator_url, world_size=args.workers, total_steps=args.steps)
        print(f"Submitted job_id={job_id} run_id={run_id}")

        # Launch workers locally
        launch_cmd = [
            sys.executable,
            "-m",
            "src.worker.launcher",
            "--workers",
            str(args.workers),
            "--steps",
            str(args.steps),
            "--coordinator-url",
            args.coordinator_url,
            "--kill-at-step",
            str(kill_step),
        ]
        workers_proc = subprocess.Popen(launch_cmd, cwd=root, env=env)

        result = _poll_recovery(args.coordinator_url, job_id, timeout_s=args.timeout)
        ttr = result["time_to_recovery_s"]

        print()
        print("=== Results ===")
        print(result)
        if ttr is not None:
            print(f"Time-to-recovery: {ttr:.2f}s (target < 60s)")
        else:
            print("Time-to-recovery: N/A (did not observe resume to RUNNING)")

        # Best-effort cleanup
        try:
            workers_proc.terminate()
        except Exception:
            pass

        return 0

    finally:
        if coordinator_proc and coordinator_proc.poll() is None:
            try:
                coordinator_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

