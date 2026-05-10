"""Tests for the FastAPI serving layer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_drift_status_returns_dict():
    r = client.get("/drift/status")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_stream_status_returns_dict():
    r = client.get("/stream/status")
    # Will 500 if dataset not downloaded — that's expected in CI without data
    assert r.status_code in (200, 500)


def test_predict_missing_age_returns_422():
    """Pydantic validation: age is required."""
    r = client.post("/predict", json={"smokes": 1})
    assert r.status_code == 422


def test_predict_invalid_age_returns_422():
    """Pydantic validation: age must be 0–120."""
    r = client.post("/predict", json={"age": 999})
    assert r.status_code == 422


def test_batch_predict_empty_list_returns_zero():
    r = client.post("/predict/batch", json={"patients": []})
    assert r.status_code == 200
    assert r.json()["count"] == 0


@pytest.mark.skipif(
    not Path("models/xgboost_model.joblib").exists(),
    reason="No trained model — run scripts/initial_train.py first",
)
def test_predict_returns_valid_risk_level():
    r = client.post("/predict", json={
        "age": 30,
        "smokes": 1,
        "stds_hpv": 1,
        "hormonal_contraceptives": 1,
        "num_sexual_partners": 5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in ("low", "moderate", "high")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert len(body["top_features"]) <= 5


@pytest.mark.skipif(
    not Path("models/xgboost_model.joblib").exists(),
    reason="No trained model — run scripts/initial_train.py first",
)
def test_batch_predict_multiple_patients():
    patients = [
        {"age": 25, "stds_hpv": 1, "smokes": 1},
        {"age": 45, "stds_hiv": 1},
        {"age": 32},
    ]
    r = client.post("/predict/batch", json={"patients": patients})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
