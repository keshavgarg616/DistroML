"""Quick validation script for Coordinator per-step aggregation.

Usage:
    Activate virtualenv, then run from repo root:
        python scripts/test_metrics.py

This script:
 - Instantiates a Coordinator
 - Registers two workers for the same job
 - Sends heartbeats for the same step from both workers
 - Prints the resulting metrics_store and checks aggregates
"""

import sys
from pathlib import Path
import asyncio
import json

# Ensure project root is importable when running the script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coordinator.coordinator import Coordinator, CoordinatorConfig
from src.coordinator.models import WorkerRegistration, HeartbeatPayload


async def main():
    coord = Coordinator(CoordinatorConfig())

    # create two workers for job 'job-test'
    job_id = "job-test"
    w1 = WorkerRegistration(
        worker_id="worker-1",
        rank=0,
        world_size=2,
        host="127.0.0.1",
        port=10001,
        job_id=job_id,
    )

    w2 = WorkerRegistration(
        worker_id="worker-2",
        rank=1,
        world_size=2,
        host="127.0.0.1",
        port=10002,
        job_id=job_id,
    )

    await coord.register_worker(w1)
    await coord.register_worker(w2)

    # Heartbeats for step 1
    hb1 = HeartbeatPayload(worker_id="worker-1", metrics={"step": 1, "loss": 2.0, "throughput": 10.0})
    hb2 = HeartbeatPayload(worker_id="worker-2", metrics={"step": 1, "loss": 1.0, "throughput": 12.0})

    await coord.handle_heartbeat(hb1)
    await coord.handle_heartbeat(hb2)

    # inspect metrics_store
    ms = coord._metrics_store.get(job_id)
    print("METRICS STORE:")
    print(json.dumps(ms, indent=2))

    # sanity checks (basic assertions)
    step_entry = ms["steps"][1]
    agg = step_entry["aggregated"]

    assert agg["workers_reported"] == 2, "workers_reported should be 2"
    assert abs(agg["avg_loss"] - 1.5) < 1e-6, f"avg_loss expected 1.5, got {agg['avg_loss']}"
    assert abs(agg["total_throughput"] - 22.0) < 1e-6, f"total_throughput expected 22.0, got {agg['total_throughput']}"

    run_summary = ms["run_summary"]
    assert run_summary["latest_step"] == 1
    assert abs(run_summary["latest_loss"] - 1.5) < 1e-6
    assert abs(run_summary["best_loss"] - 1.5) < 1e-6
    assert abs(run_summary["avg_throughput"] - 22.0) < 1e-6

    print("All assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
