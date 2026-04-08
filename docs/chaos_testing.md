# Chaos testing & failure behaviour — audit (Week 5–6)

## Summary

- **What exists in code:** heartbeat-based failure detection, optional explicit worker exit handling, a recovery path that can move a job to `RECOVERING`, relaunch worker processes, and log failure/recovery timestamps. Workers support **failure injection** (kill at step, pause, random heartbeat drops) inside `WorkerRuntime`, but **not** from the normal CLI entrypoint.
- **What is missing or weak:** no dedicated chaos test suite (random failures, multi-failure runs). No pytest that kills a worker mid-training and asserts end-to-end recovery. Recovery often **does nothing** unless a job record exists in the coordinator (see scenario below). WebSocket streaming can be ignored for chaos scope, as requested.
- **Automated tests run:** `pytest tests/` — **19 passed** (temp venv with torch + project deps; root `requirements.txt` lists invalid PyPI names like `logging`, `asyncio`, so installs must skip those lines or use a fixed file).

---

## Code map (week tasks → where it lives)

| Task | Present? | Location / notes |
|------|----------|-------------------|
| Heartbeat timeout | Yes | `HeartbeatMonitor` in `src/coordinator/coordinator.py` — stale workers → `mark_worker_lost` → `_on_worker_lost` |
| Worker process exit detection | Partial | **Explicit:** `POST /api/workers/deregister` → `handle_worker_exit`. **Crash / kill:** only via missing heartbeats (no OS-level hook) |
| Job → `RECOVERING`, relaunch, checkpoint resume | Partial | `_trigger_recovery` sets `RECOVERING`, logs `Recovery started`, spawns workers with `subprocess.Popen`. Checkpoint resume is implemented in **worker** (`find_latest_checkpoint` / `restore_checkpoint`); coordinator recovery comments still say “future: checkpoint logic” in `_check_recovery_status` |
| Integration test: kill mid-training | No | Not found under `tests/` |
| Time-to-recovery &lt; 60s | Not measured | Would need a job in `_jobs`, a real failure, and timestamps; not automated here |
| Failure injection: kill step N, pause T, drop heartbeats | Yes (in-process) | `WorkerRuntime.configure_failure_injection` + `training_loop` / `_emit_heartbeat` in `src/worker/runtime.py` |
| Recovery timeline logging | Partial | **Failure detected:** `Failure detected time:` in `_on_worker_lost` / `handle_worker_exit`. **Recovery started:** `Recovery started:` in `_trigger_recovery`. **Training resumed:** log when resuming from checkpoint (`start_step > 0`) in `training_loop` |
| Chaos: random failures, multiple failures | No | No scenarios in repo tests |

---

## Automated test run (what we executed)

- **Command:** `PYTHONPATH=. python -m pytest tests/ -q`
- **Environment:** Python 3.13 venv with `pytest`, `torch`, `requests`, `pydantic`, `python-statemachine`, etc. (invalid requirements lines omitted).
- **Result:** **19 passed**, ~6.4s.
- **Coverage:** mostly **checkpoint restore/manifest** (`tests/test_checkpoint_restore.py`), plus `tests/test_dataloader.py` and `tests/test_websocket_streaming.py`. **None** of these are end-to-end chaos or kill-and-recovery tests.

---

## Scenario 1 — Happy path (launcher + coordinator, 4 workers, 100 steps)

**What you did:** Coordinator on `127.0.0.1:8000`, workers via `python -m src.worker.launcher`, `gloo`, same job/run IDs.

**What went right**

- All four workers registered and finished training.
- Heartbeats and HTTP metrics returned 200 during the run.
- Barrier and training loop completed; launcher reported all workers successful.

**What went wrong / gaps**

- WebSocket URL returned 404 and uvicorn warned about missing WebSocket extras — **out of scope** if we ignore streaming.
- After workers exited, the coordinator later logged workers as **heartbeat stale** and **LOST**, plus **“Cannot trigger recovery - job not found”**.

**Why**

- Workers stopped sending heartbeats when processes ended; the monitor still treats that as failure after the timeout window.
- **Jobs are not created** when workers-only register. `register_worker` adds workers to the registry but does **not** call `submit_job`, so `self._jobs[job_id]` is empty → recovery cannot run. That matches the warning in your logs.

---

## Scenario 2 — Heartbeat timeout detection

**What the code does**

- Periodically finds workers whose last heartbeat is older than the configured timeout, marks them **LOST**, logs **failure detected**, then tries `_trigger_recovery`.

**What went right**

- Timeout path is implemented and logged clearly in real runs (stale heartbeats → LOST).

**What went wrong**

- Without a submitted job, recovery stops at **“job … not found”**.
- Treating **normal shutdown** the same as **crash** (no deregister) produces noisy LOST events after a successful run.

---

## Scenario 3 — Explicit worker exit (`/api/workers/deregister`)

**What the code does**

- Marks worker lost, logs failure time, calls `_trigger_recovery`.

**What went right**

- Clear hook for “worker told us it is exiting” (different from silent crash).

**What went wrong**

- Workers in the current launcher/runtime path do not appear to call this on clean exit, so this path is easy to miss in local runs.

---

## Scenario 4 — Recovery orchestration (`RECOVERING`, relaunch)

**What the code does**

- If the job exists in `_jobs`, transitions to `RECOVERING`, logs recovery start, spawns one subprocess per rank with `python src/worker/runtime.py ...` (working directory must allow that path; differs from `python -m src.worker.runtime`).

**What went right**

- State transition and subprocess relaunch are implemented; metadata can carry `run_id`, `world_size`, `total_steps` for relaunch.

**What went wrong**

- **Job must be created first** via `POST /api/jobs` with metadata your recovery code expects (`run_id` in `job_spec.metadata`, etc.). Launcher-only workflows never create that job.
- Relaunch command may fail if cwd / Python path does not match how you normally start workers.
- Full **checkpoint-driven** recovery is not fully wired in the coordinator’s `_check_recovery_status` (comments still describe future work).

---

## Scenario 5 — Failure injection (kill / pause / drop heartbeat)

**What exists**

- `kill_at_step` → `sys.exit(1)` inside the training loop.
- `pause_at_step` + `pause_duration` → sleep in-loop.
- `drop_heartbeat_rate` → skip sending some heartbeats.

**What went right**

- Hooks are easy to find and reason about for controlled experiments.

**What went wrong**

- **`main()` does not expose CLI flags** for these; injection requires calling `configure_failure_injection` from code or a custom script.
- No automated tests in `tests/` drive these hooks.

---

## Scenario 6 — “Chaos” (random failures, multiple failures per run)

**What exists**

- No random scheduler, no multi-failure scenario tests in the repository.

**What went wrong**

- Week 6 chaos objectives are **not** covered by automated tests yet.

---

## Recommendations for future test docs (not implemented here)

- Add a **job submission** step before workers so recovery can run, or auto-create/link jobs on worker registration.
- Add pytest or a script that: submits job → starts workers → injects kill → asserts coordinator state and worker resume within a time budget.
- Expose failure-injection flags on `src.worker.runtime` CLI for repeatable chaos runs.
- Clean shutdown: call deregister or stop heartbeat monitor expectations on normal completion to avoid false LOST signals.

---

## Quick reference — files

| Area | File(s) |
|------|---------|
| Heartbeat monitor & recovery | `src/coordinator/coordinator.py` |
| Worker failure injection | `src/worker/runtime.py` |
| API (jobs, workers, deregister) | `src/coordinator/main.py` |
| Checkpoint tests | `tests/test_checkpoint_restore.py` |
