import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: these tests require a trained + promoted model to exist at
# config.MLFLOW_TRACKING_URI (run train.py first, or point
# MLFLOW_TRACKING_URI at a test database with a promoted model).
# They are skipped automatically if no Production model is available.

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SERVING_TESTS") != "1",
    reason="Serving tests require a trained model registry; set RUN_SERVING_TESTS=1 "
    "and a valid MLFLOW_TRACKING_URI to run them.",
)

VALID_PAYLOAD = {
    "tenure_months": 5,
    "monthly_charges": 85.5,
    "total_charges": 427.5,
    "contract_type": "month-to-month",
    "internet_service": "Fiber optic",
    "has_tech_support": False,
    "senior_citizen": False,
}


@pytest.fixture
def client():
    from src.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["production_model_loaded"] is True


def test_predict_requires_api_key(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 401


def test_predict_with_valid_key_returns_prediction(client):
    api_key = os.environ.get("API_KEY", "dev-key-change-me")
    r = client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": api_key})
    assert r.status_code == 200
    body = r.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_invalid_contract_type(client):
    api_key = os.environ.get("API_KEY", "dev-key-change-me")
    bad_payload = dict(VALID_PAYLOAD, contract_type="lifetime")
    r = client.post("/predict", json=bad_payload, headers={"X-API-Key": api_key})
    assert r.status_code == 422  # pydantic validation error


def test_dashboard_serves_html(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_metrics_endpoints_require_api_key(client):
    # These return real feature data and must be behind the same auth as
    # /predict — verifies the gap from an earlier version is actually closed.
    for path in ["/metrics/summary", "/metrics/predictions", "/metrics/shadow", "/metrics/drift"]:
        r = client.get(path)
        assert r.status_code == 401, f"{path} should require an API key but returned {r.status_code}"


def test_metrics_endpoints_work_with_valid_key(client):
    api_key = os.environ.get("API_KEY", "dev-key-change-me")
    r = client.get("/metrics/summary", headers={"X-API-Key": api_key})
    assert r.status_code == 200
