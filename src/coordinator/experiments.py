"""
Experiment metadata store for DistroML.

Captures and persists per-run metadata at job submission time:
  - JobSpec (full copy of what was submitted)
  - Git commit hash (reproducibility)
  - Seeds (global, numpy, torch — extracted from job metadata or defaults)
  - Runtime configuration (Python version, PyTorch version, platform)

Storage: in-memory dict + optional JSON file persistence (matches the
rest of the MVP which avoids a live Postgres dependency for local runs).
"""

import json
import logging
from logging import config
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

#helpers 

def get_git_commit_hash() -> str:
    """Return the current HEAD git commit hash, or 'unknown' on failure."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(project_root),
        )
        return result.decode().strip()
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    """Return the current git branch name, or 'unknown' on failure."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        result = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(project_root),
        )
        return result.decode().strip()
    except Exception:
        return "unknown"


def get_runtime_config() -> Dict[str, Any]:
    """Capture the current runtime environment."""
    config: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
    }

    try:
        import torch
        config["torch_version"] = torch.__version__
        config["cuda_available"] = torch.cuda.is_available()
        config["cuda_version"] = torch.version.cuda if (torch.cuda.is_available() and torch.version.cuda) else None
        config["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        config["torch_version"] = "not_installed"
        config["cuda_available"] = False

    try:
        import numpy as np
        config["numpy_version"] = np.__version__
    except ImportError:
        config["numpy_version"] = "not_installed"

    return config


def extract_seeds(job_metadata: Dict[str, Any]) -> Dict[str, int]:
    """
    Pull seed values out of job metadata.

    Workers are expected to pass seeds in job_spec.metadata like:
        { "seeds": { "global": 42, "torch": 42, "numpy": 42 } }

    If not present we record an empty dict — the run is non-deterministic.
    """
    raw = job_metadata.get("seeds", {})
    if not isinstance(raw, dict):
        logger.warning("job_metadata['seeds'] is not a dict, ignoring")
        return {}
    seeds: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            seeds[str(k)] = int(v)
        except (TypeError, ValueError):
            logger.warning(f"Seed '{k}' value '{v}' is not an int, skipping")
    return seeds


# experimentStore 
class ExperimentStore:
    """
    In-memory experiment metadata store with optional JSON persistence.

    One record is written per run at job submission time. Records are
    keyed by run_id (which is also stored on the coordinator's JobInfo).

    Persistence: if persist_path is set, the store is written atomically
    to a JSON file after every mutation so restarts do not lose history.
    """

    def __init__(self, persist_path: Optional[str] = None):
        # run_id → experiment record dict
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._persist_path = persist_path

        if persist_path:
            Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.info(
            f"ExperimentStore initialized "
            f"(persist_path={persist_path or 'disabled'}, "
            f"loaded={len(self._experiments)} records)"
        )

#write 

    def record(
        self,
        job_id: str,
        run_id: str,
        job_spec: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create and store an experiment record at job submission time.

        Args:
            job_id:         Coordinator-assigned job UUID.
            run_id:         Run UUID (stable across recovery).
            job_spec:       Serialised JobSpec dict.
            extra_metadata: Any caller-supplied extra context.

        Returns the full experiment record dict.
        """
        job_metadata: Dict[str, Any] = job_spec.get("metadata", {}) or {}

        record: Dict[str, Any] = {
            "job_id": job_id,
            "run_id": run_id,
            "recorded_at": datetime.utcnow().isoformat(),
            # ── reproducibility 
            "git_commit_hash": get_git_commit_hash(),
            "git_branch": get_git_branch(),
            # ── what was submitted 
            "job_spec": job_spec,
            # ── random seeds 
            "seeds": extract_seeds(job_metadata),
            # ── environment
            "runtime_config": get_runtime_config(),
            # ── optional caller context 
            "extra_metadata": extra_metadata or {},
        }

        self._experiments[run_id] = record
        logger.info(
            f"Experiment recorded: run_id={run_id}, job_id={job_id}, "
            f"git={record['git_commit_hash'][:8]}"
        )
        self._save_to_disk()
        return record

    #read

    def get_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a record by run_id. Returns None if not found."""
        return self._experiments.get(run_id)

    def get_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the record for a given job_id (first match)."""
        for record in self._experiments.values():
            if record["job_id"] == job_id:
                return record
        return None

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all experiment records, newest first."""
        records = list(self._experiments.values())
        records.sort(key=lambda r: r.get("recorded_at", ""), reverse=True)
        return records

    #compare 

    def compare(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Build a side-by-side comparison of two or more runs.

        Returns a dict with:
            compared_runs   - number of runs successfully found
            missing_runs    - run_ids that were not in the store
            records         - the full record dicts
            summary         - high-level diff (seeds, git, runtime diffs)
        """
        found: List[Dict[str, Any]] = []
        missing: List[str] = []

        for run_id in run_ids:
            record = self._experiments.get(run_id)
            if record:
                found.append(record)
            else:
                missing.append(run_id)

        return {
            "compared_runs": len(found),
            "missing_runs": missing,
            "records": found,
            "summary": self._build_summary(found),
        }

    def _build_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {}

        git_commits = list({r["git_commit_hash"] for r in records})
        git_branches = list({r["git_branch"] for r in records})

        # Collect all seed keys across runs
        all_seed_keys = set()
        for r in records:
            all_seed_keys.update(r.get("seeds", {}).keys())

        seeds_by_run: Dict[str, Dict[str, int]] = {
            r["run_id"]: r.get("seeds", {}) for r in records
        }

        # Collect torch versions
        torch_versions = list({
            r.get("runtime_config", {}).get("torch_version", "unknown")
            for r in records
        })

        return {
            "run_ids": [r["run_id"] for r in records],
            "recorded_at": [r["recorded_at"] for r in records],
            # reproducibility flags
            "same_git_commit": len(git_commits) == 1,
            "git_commits": git_commits,
            "git_branches": git_branches,
            # seed comparison
            "seed_keys_present": sorted(all_seed_keys),
            "seeds_by_run": seeds_by_run,
            "seeds_identical": len({
                json.dumps(v, sort_keys=True) for v in seeds_by_run.values()
            }) == 1,
            # runtime comparison
            "torch_versions": torch_versions,
            "same_torch_version": len(torch_versions) == 1,
        }

    #persistence 

    def _save_to_disk(self) -> None:
        if not self._persist_path:
            return
        try:
            tmp = self._persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._experiments, f, indent=2, default=str)
            os.replace(tmp, self._persist_path)
            logger.debug(f"ExperimentStore persisted to {self._persist_path}")
        except Exception as e:
            logger.warning(f"Failed to persist ExperimentStore: {e}")

    def _load_from_disk(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r") as f:
                self._experiments = json.load(f)
            logger.info(
                f"Loaded {len(self._experiments)} experiment records "
                f"from {self._persist_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load ExperimentStore from disk: {e}")
            self._experiments = {}

    def __len__(self) -> int:
        return len(self._experiments)

    def __repr__(self) -> str:
        return f"ExperimentStore(records={len(self._experiments)}, persist={self._persist_path})"