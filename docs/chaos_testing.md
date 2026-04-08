# Chaos testing and failure behaviour (Week 5-6)

## Summary

- Heartbeat timeout detection is implemented and still the main crash detector.
- Clean worker finish is now handled via `POST /api/workers/complete`, which removes workers from registry and helps avoid false LOST status after successful runs.
- Recovery flow exists: job can move to `RECOVERING`, failed rank can be relaunched, and worker-side checkpoint restore can resume from latest checkpoint for the same `run_id`.
- Job metadata defaults are now filled on submit (`run_id`, `world_size`, `total_steps`, `checkpoint_dir`) so recovery has required fields.
- Main gaps remain: no end-to-end chaos integration tests, no automated time-to-recovery checks, and failure injection hooks are not exposed as CLI flags.

---

## What is implemented vs pending

| Task | Status | Notes |
|------|--------|-------|
| Heartbeat timeout | Implemented | `HeartbeatMonitor` checks stale workers, marks LOST, calls recovery callback |
| Worker exit detection | Partial | Explicit exit endpoint exists (`/api/workers/deregister`), crash/kill still inferred through missed heartbeat |
| Graceful success shutdown | Implemented | Worker calls `/api/workers/complete`; coordinator deregisters worker without triggering recovery |
| Job `RECOVERING` transition | Implemented | Coordinator transitions RUNNING -> RECOVERING on loss |
| Relaunch worker(s) | Implemented (single failed rank) | Recovery now relaunches failed rank only |
| Resume from checkpoint | Partial but wired | Worker restores from latest checkpoint using `run_id` and `checkpoint_dir`; coordinator does not deeply validate checkpoint progress |
| Kill worker mid-training integration test | Missing | Not present in `tests/` |
| Recovery continuation verification test | Missing | Not present in `tests/` |
| Time-to-recovery measurement (<60s) | Missing | No automated metric or assertion |
| Failure injection hooks (kill/pause/drop heartbeat) | Implemented in runtime | Available in code, not exposed in worker CLI |
| Recovery timeline logging | Implemented | Logs include failure detected and recovery started; training resumed logged by worker when restoring |
| Chaos scenarios (random/multi-failure) | Missing | No dedicated suite/script in repo |

---

## Test run (latest)

- Command: `PYTHONPATH=. python -m pytest tests/ -q`
- Result: **19 passed**
- Coverage focus:
  - checkpoint restore and manifest behaviour
  - dataloader tests
  - websocket tests (not relevant for this doc scope)
- Not covered by automated tests:
  - kill mid-training then recover
  - multiple failures across runs
  - recovery time target

---

## Scenario details (WebSockets intentionally ignored)

## 1) Normal training run (4 workers, no injected failure)

**What went right**

- Workers register, train, and finish.
- Heartbeats and metrics APIs are used as expected.
- Workers call `/api/workers/complete` on successful finish.
- Workers are removed from registry instead of waiting for heartbeat timeout.

**What can still go wrong**

- If `/api/workers/complete` fails repeatedly (network/coordinator issue), worker exits anyway and may later be marked LOST by heartbeat timeout.
- In launcher-only flow (no submitted job record), there is still no job lifecycle state to complete; only worker-level cleanup happens.

---

## 2) Heartbeat timeout failure

**What went right**

- Stale worker detection is active and marks worker LOST.
- Recovery callback is triggered with clear timeline logs.

**What can still go wrong**

- If job is not present in coordinator `_jobs` (workers registered directly), recovery cannot proceed (`job not found`).

---

## 3) Explicit failure exit (`/api/workers/deregister`)

**What went right**

- Clear endpoint path to mark worker exit as failure and trigger recovery.

**What can still go wrong**

- This path is for failure semantics, not clean completion.
- If used incorrectly for normal finish, it can trigger unnecessary recovery.

---

## 4) Recovery orchestration (`RECOVERING` and relaunch)

**What went right**

- Coordinator transitions to `RECOVERING`.
- Relaunch command now uses module form with correct cwd and `PYTHONPATH`.
- Recovery uses metadata defaults and passes `checkpoint_dir`.
- Failed rank relaunch avoids immediate duplicate restart of all ranks.

**What can still go wrong**

- Single-rank relaunch can still stall distributed training if other ranks are already blocked in collectives.
- Recovery health check is still coarse (based on worker status, not full rank sync/progress validation).

---

## 5) Failure injection hooks (kill, pause, drop heartbeat)

**What went right**

- All three hooks exist in worker runtime and are easy to call from code/tests.

**What can still go wrong**

- No CLI flags in worker entrypoint for these hooks.
- No automated integration tests currently use them end-to-end.

---

## 6) Chaos scenarios (random and multiple failures)

**Current state**

- No dedicated chaos runner or integration suite in repository.
- No repeated-run harness to validate stability after multiple random failures.

---

## Remaining high-priority gaps

- Add integration test: kill one worker mid-training, verify recovery and continued progress.
- Add integration test: multiple failures across runs, verify no deadlock and correct final state.
- Add recovery timing instrumentation and assert `<60s` target.
- Add CLI flags for failure injection hooks for repeatable local chaos runs.

---

## Key files

| Area | File(s) |
|------|---------|
| Heartbeat and recovery orchestration | `src/coordinator/coordinator.py` |
| Coordinator APIs for worker lifecycle | `src/coordinator/main.py` |
| Worker runtime and failure injection hooks | `src/worker/runtime.py` |
| Checkpoint tests | `tests/test_checkpoint_restore.py` |
