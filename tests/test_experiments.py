"""
Test script for Week 8 experiment metadata.

Usage (from repo root):
    python scripts/test_experiments.py
"""

import sys
from pathlib import Path
import asyncio
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coordinator.coordinator import Coordinator, CoordinatorConfig
from src.coordinator.models import WorkerRegistration, HeartbeatPayload
from src.coordinator.experiments import ExperimentStore, get_git_commit_hash, get_runtime_config
from src.coordinator.models import JobSpec


async def main():
    print("=" * 60)
    print("Week 8 — Experiment Metadata Tests")
    print("=" * 60)

    print("\n[1] Git commit hash")
    git_hash = get_git_commit_hash()
    assert git_hash != "", "git hash should not be empty"
    print(f"    git hash: {git_hash}")
    print("    ✅ passed")

    print("\n[2] Runtime config")
    runtime = get_runtime_config()
    assert "python_version" in runtime
    assert "torch_version" in runtime
    assert "cuda_available" in runtime
    print(f"    python: {runtime['python_version'].split()[0]}")
    print(f"    torch:  {runtime['torch_version']}")
    print(f"    cuda:   {runtime['cuda_available']}")
    print("    ✅ passed")
    print("\n[3] Record experiment")
    store = ExperimentStore()  # no persistence for test

    job_spec = JobSpec(
        name="test-job",
        world_size=2,
        metadata={
            "run_id": "run-aaa",
            "seeds": {"global": 42, "torch": 42, "numpy": 42},
        }
    )

    record = store.record(
        job_id="job-aaa",
        run_id="run-aaa",
        job_spec=job_spec.model_dump(),
    )

    assert record["job_id"] == "job-aaa"
    assert record["run_id"] == "run-aaa"
    assert record["seeds"] == {"global": 42, "torch": 42, "numpy": 42}
    assert record["git_commit_hash"] == git_hash
    assert "python_version" in record["runtime_config"]
    print(f"    recorded run_id: {record['run_id']}")
    print(f"    seeds: {record['seeds']}")
    print("    ✅ passed")

    print("\n[4] Get by run_id")
    fetched = store.get_by_run_id("run-aaa")
    assert fetched is not None
    assert fetched["job_id"] == "job-aaa"
    print("    ✅ passed")

    print("\n[5] Get by job_id")
    fetched = store.get_by_job_id("job-aaa")
    assert fetched is not None
    assert fetched["run_id"] == "run-aaa"
    print("    ✅ passed")
    
    print("\n[6] List all")
    # add a second record
    store.record(
        job_id="job-bbb",
        run_id="run-bbb",
        job_spec=JobSpec(name="test-job-2", world_size=4, metadata={"run_id": "run-bbb"}).model_dump(),
    )
    all_records = store.list_all()
    assert len(all_records) == 2
    print(f"    total records: {len(all_records)}")
    print("    ✅ passed")

    print("\n[7] Compare two runs")
    result = store.compare(["run-aaa", "run-bbb"])
    assert result["compared_runs"] == 2
    assert result["missing_runs"] == []
    assert result["summary"]["same_git_commit"] == True  # both from same repo
    print(f"    compared_runs: {result['compared_runs']}")
    print(f"    same_git_commit: {result['summary']['same_git_commit']}")
    print(f"    seeds_identical: {result['summary']['seeds_identical']}")
    print("    ✅ passed")

    print("\n[8] Compare with missing run_id")
    result = store.compare(["run-aaa", "run-DOESNOTEXIST"])
    assert result["compared_runs"] == 1
    assert "run-DOESNOTEXIST" in result["missing_runs"]
    print(f"    missing_runs: {result['missing_runs']}")
    print("    ✅ passed")
    print("\n[9] Record without seeds (non-deterministic run)")
    store.record(
        job_id="job-ccc",
        run_id="run-ccc",
        job_spec=JobSpec(name="no-seeds-job", world_size=2, metadata={"run_id": "run-ccc"}).model_dump(),
    )
    fetched = store.get_by_run_id("run-ccc")
    assert fetched["seeds"] == {}
    print("    seeds: {} (non-deterministic, expected)")
    print("    ✅ passed")
    print("\n[10] Full coordinator submit + experiment record")
    import uuid

    coord = Coordinator(CoordinatorConfig())
    exp_store = ExperimentStore()

    spec = JobSpec(
        name="integration-test",
        world_size=2,
        metadata={"seeds": {"global": 99}},
    )

    if "run_id" not in spec.metadata:
        spec.metadata["run_id"] = str(uuid.uuid4())
    run_id = spec.metadata["run_id"]

    job_id = await coord.submit_job(spec)

    exp_store.record(
        job_id=job_id,
        run_id=run_id,
        job_spec=spec.model_dump(),
    )

    record = exp_store.get_by_job_id(job_id)
    assert record is not None
    assert record["run_id"] == run_id
    assert record["seeds"] == {"global": 99}
    print(f"    job_id:  {job_id}")
    print(f"    run_id:  {run_id}")
    print(f"    seeds:   {record['seeds']}")
    print("    ✅ passed")

    print("\n" + "=" * 60)
    print("All assertions passed ✅")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())