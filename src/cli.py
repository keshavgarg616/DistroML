import argparse
import sys
from pathlib import Path

import requests
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Any, Dict, Optional

class JobConfig(BaseModel):
    name: str = Field(description="Human-readable job name")
    world_size: int = Field(ge=1, description="Number of workers")
    backend: str = Field(default="gloo", description="Distributed backend: gloo or nccl")
    coordinator_url: str = Field(
        default="http://localhost:8000",
        description="Coordinator API base URL",
    )
    total_steps: int = Field(default=100, ge=1, description="Total training steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    @field_validator("backend")
    @classmethod
    def _backend_choices(cls, v: str) -> str:
        allowed = {"gloo", "nccl"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {sorted(allowed)}, got '{v}'")
        return v

    @field_validator("coordinator_url")
    @classmethod
    def _url_not_empty(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("coordinator_url must start with http:// or https://")
        return v.rstrip("/")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        _die(f"Job file not found: {path}")
    if not path.is_file():
        _die(f"Path is not a file: {path}")
    if path.suffix not in {".yaml", ".yml"}:
        _die(f"Expected a .yaml / .yml file, got: {path.name}")

    try:
        with path.open() as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        _die(f"YAML parse error in {path}:\n  {exc}")

    if not isinstance(raw, dict):
        _die(f"{path} must contain a YAML mapping at the top level, got {type(raw).__name__}")

    return raw


def _validate_config(raw: dict, path: Path) -> JobConfig:
    try:
        return JobConfig(**raw)
    except ValidationError as exc:
        lines = [f"Validation errors in {path}:"]
        for err in exc.errors():
            field = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  {field}: {err['msg']}")
        _die("\n".join(lines))


def _die(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)

def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.job_file)
    raw = _load_yaml(path)
    cfg = _validate_config(raw, path)

    print(f"Job:          {cfg.name}")
    print(f"Workers:      {cfg.world_size}")
    print(f"Backend:      {cfg.backend}")
    print(f"Coordinator:  {cfg.coordinator_url}")
    print(f"Steps:        {cfg.total_steps}")

    # Submit job to coordinator
    job_spec_payload = {
        "name": cfg.name,
        "world_size": cfg.world_size,
        "metadata": cfg.metadata,
    }

    try:
        resp = requests.post(
            f"{cfg.coordinator_url}/api/jobs",
            json=job_spec_payload,
            timeout=10,
        )
    except requests.ConnectionError:
        _die(
            f"Could not connect to coordinator at {cfg.coordinator_url}.\n"
            "  Is the coordinator running?  Start it with:\n"
            "    python -m src.coordinator.main"
        )
    except requests.Timeout:
        _die(f"Request to coordinator timed out ({cfg.coordinator_url})")

    if resp.status_code != 201:
        _die(
            f"Coordinator rejected the job (HTTP {resp.status_code}):\n"
            f"  {resp.text}"
        )

    result = resp.json()
    job_id = result["job_id"]
    run_id = result.get("run_id", "—")
    print(f"\nJob submitted — job_id={job_id}  run_id={run_id}")

    # Launch workers locally
    from src.worker.launcher import WorkerLauncher

    launcher = WorkerLauncher()
    launcher.job_id = job_id
    launcher.run_id = run_id

    try:
        launcher.launch_workers(
            world_size=cfg.world_size,
            backend=cfg.backend,
            coordinator_url=cfg.coordinator_url,
            total_steps=cfg.total_steps,
        )
        launcher.monitor()
    except Exception as exc:
        launcher.shutdown()
        _die(f"Worker error: {exc}")

    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distroml",
        description="DistroML — distributed ML training CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    run_p = sub.add_parser("run", help="Submit and run a training job from a YAML file")
    run_p.add_argument("job_file", metavar="job.yaml", help="Path to job definition file")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
