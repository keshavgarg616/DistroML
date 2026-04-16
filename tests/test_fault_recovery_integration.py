"""
Week 5 integration test:
- Simulate worker loss mid-job
- Verify Coordinator transitions RUNNING -> RECOVERING -> RUNNING
- Verify Coordinator attempts to relaunch the failed rank

Note: This test runs the Coordinator in-process and patches subprocess.Popen so
no real worker processes are spawned.
"""

import asyncio
from unittest.mock import patch

from src.coordinator.coordinator import Coordinator, CoordinatorConfig
from src.coordinator.models import JobSpec, WorkerRegistration
from src.common import JobState


def test_recovery_orchestration_transitions_and_relaunch():
    async def _run():
        config = CoordinatorConfig(
            heartbeat_timeout_seconds=1,
            heartbeat_check_interval_seconds=1,
            recovery_backoff_seconds=1,
            max_recovery_attempts=3,
            coordinator_base_url="http://127.0.0.1:8000",
        )
        coordinator = Coordinator(config=config, ws_manager=None)
        # Intentionally do NOT start the heartbeat monitor in this test.
        # We want to deterministically simulate a single worker loss without
        # background timeouts marking other workers LOST.
        try:
            job_id = await coordinator.submit_job(
                JobSpec(
                    name="recovery-test",
                    world_size=2,
                    metadata={
                        "run_id": "test-run",
                        "world_size": 2,
                        "total_steps": 10,
                        "checkpoint_dir": "./checkpoints",
                    },
                )
            )
            await coordinator.start_job(job_id)

            w0 = WorkerRegistration(
                worker_id="worker-0",
                rank=0,
                world_size=2,
                host="localhost",
                port=29500,
                job_id=job_id,
                metadata={},
            )
            w1 = WorkerRegistration(
                worker_id="worker-1",
                rank=1,
                world_size=2,
                host="localhost",
                port=29501,
                job_id=job_id,
                metadata={},
            )

            await coordinator.register_worker(w0)
            await coordinator.register_worker(w1)

            with patch("src.coordinator.coordinator.subprocess.Popen") as popen_mock:
                # Mirror heartbeat monitor behavior: mark LOST then trigger callback.
                await coordinator.worker_registry.mark_worker_lost("worker-0")
                await coordinator._on_worker_lost("worker-0")

                status = await coordinator.get_job_status(job_id)
                assert status.status == JobState.RECOVERING

                # Ensure relaunch was attempted for the failed rank.
                assert popen_mock.call_count == 1
                args, kwargs = popen_mock.call_args
                cmd = args[0]
                assert "-m" in cmd and "src.worker.runtime" in cmd
                assert "--rank" in cmd and cmd[cmd.index("--rank") + 1] == "0"

            # Allow recovery backoff check to run and resume the job.
            await asyncio.sleep(1.25)
            status2 = await coordinator.get_job_status(job_id)
            assert status2.status == JobState.RUNNING

        finally:
            # Ensure any background monitor is stopped if it was started elsewhere.
            await coordinator.heartbeat_monitor.stop()

    asyncio.run(_run())

