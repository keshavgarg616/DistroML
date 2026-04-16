import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.coordinator.main import app


def test_get_job_status_endpoint_returns_404_for_unknown_job():
    # Use context manager so FastAPI lifespan runs (coordinator is initialized).
    with TestClient(app) as client:
        r = client.get("/api/jobs/does-not-exist")
        assert r.status_code == 404

