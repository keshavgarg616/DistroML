import json
import hashlib
import os
from datetime import datetime
from pathlib import Path


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_manifest(
    step: int,
    run_id: str,
    job_id: str,
    world_size: int,
    worker_shards: list,
    backend: str = "gloo",
):
    return {
        "step": step,
        "run_id": run_id,
        "job_id": job_id,
        "world_size": world_size,
        "timestamp": datetime.utcnow().isoformat(),
        "worker_shards": worker_shards,
        "metadata": {
            "backend": backend
        },
    }


def save_manifest(manifest: dict, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")

    tmp_path = manifest_path + ".tmp"

    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=4)

    os.replace(tmp_path, manifest_path)

    return manifest_path
