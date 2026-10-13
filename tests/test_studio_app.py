"""Tests for FastAPI Web Studio endpoints."""

import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from mergestudio.studio.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def test_models(tmp_path):
    m1 = tmp_path / "model1"
    m2 = tmp_path / "model2"
    m1.mkdir()
    m2.mkdir()

    save_file({"weight": torch.tensor([1.0, 2.0])}, m1 / "model.safetensors")
    save_file({"weight": torch.tensor([3.0, 4.0])}, m2 / "model.safetensors")

    with open(m1 / "config.json", "w") as f:
        json.dump({"num_hidden_layers": 8}, f)

    return m1, m2


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "mergestudio"


def test_inspect_endpoint(client, test_models):
    m1, _ = test_models
    res = client.post("/api/inspect", json={"path": str(m1)})
    assert res.status_code == 200
    data = res.json()
    assert data["total_tensors"] == 1
    assert data["num_hidden_layers"] == 8


def test_plan_validate_endpoint(client, test_models):
    m1, m2 = test_models
    plan = {
        "name": "validation-test",
        "method": "slerp",
        "models": [
            {"path": str(m1), "name": "m1"},
            {"path": str(m2), "name": "m2"},
        ],
        "slerp_config": {"alpha": 0.5},
    }
    res = client.post("/api/plan/validate", json=plan)
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True


def test_merge_start_and_status(client, test_models, tmp_path):
    m1, m2 = test_models
    out_dir = tmp_path / "web_merged"

    plan = {
        "name": "web-slerp",
        "method": "slerp",
        "models": [
            {"path": str(m1), "name": "m1"},
            {"path": str(m2), "name": "m2"},
        ],
        "slerp_config": {"alpha": 0.5},
        "output_dir": str(out_dir),
    }

    res = client.post("/api/merge/start", json=plan)
    assert res.status_code == 200
    data = res.json()
    job_id = data["job_id"]
    assert job_id is not None

    status_res = client.get(f"/api/merge/status/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ("queued", "running", "completed")


def test_serve_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "MergeStudio" in res.text


def test_api_analyze_divergence(client, tmp_path):
    m1 = tmp_path / "div1"
    m2 = tmp_path / "div2"
    m1.mkdir()
    m2.mkdir()
    save_file({"model.layers.0.mlp.weight": torch.ones(2, 2)}, m1 / "model.safetensors")
    save_file({"model.layers.0.mlp.weight": torch.ones(2, 2)}, m2 / "model.safetensors")

    res = client.post("/api/analyze/divergence", json={"model_a": str(m1), "model_b": str(m2)})
    assert res.status_code == 200
    data = res.json()
    assert data["total_layers"] == 1
    assert data["mean_cosine_similarity"] == 1.0


def test_api_evaluate(client, test_models):
    m1, _ = test_models
    res = client.post("/api/evaluate", json={"model_dir": str(m1)})
    assert res.status_code == 200
    data = res.json()
    assert data["healthy"] is True
    assert data["total_tensors"] == 1


def test_api_export_ollama(client, test_models):
    m1, _ = test_models
    res = client.post("/api/export/ollama", json={"model_dir": str(m1)})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "Modelfile" in data["modelfile"]

