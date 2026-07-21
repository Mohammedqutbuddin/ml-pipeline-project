"""
FastAPI serving layer with two model-loading modes, controlled by the
MODEL_SOURCE environment variable:

  MODEL_SOURCE=mlflow (default, for local development)
    Loads Production/Staging models live from the MLflow registry over HTTP.
    Requires a running `mlflow server` — gives you the full live-registry
    workflow (shadow deployment, promote_staging.py, hot /reload-model).

  MODEL_SOURCE=local (used by the Dockerfile for deployment)
    Loads models from a self-contained local directory (model_artifacts/,
    produced by `python src/training/export_model.py`). No live MLflow
    server needed at runtime — this is what makes the container deployable
    to a free-tier platform that only gives you one service.

Run locally with: uvicorn src.serving.app:app --reload --port 8000
"""
import json
import os
import sys
import time
from typing import Literal, Optional

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.logging_config import get_logger
from src.monitoring.log_predictions import (
    get_dashboard_summary,
    load_recent_predictions,
    load_shadow_comparison,
    log_prediction,
    log_shadow_prediction,
)

logger = get_logger("churn_pipeline.serving")

app = FastAPI(
    title="Churn Prediction API",
    description="Serves predictions from the current Production model, with shadow "
    "scoring against any Staging challenger.",
    version="1.0.0",
)

_production_model = None
_production_version = "unknown"
_shadow_model = None
_shadow_version = None
_explainer = None

_MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "mlflow")  # "mlflow" | "local"
_ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")  # "development" | "production"
_LOCAL_MODEL_DIR = os.path.join(config.BASE_DIR, "model_artifacts")

# --- API key auth ---------------------------------------------------------
# Set the API_KEY environment variable / platform secret before deploying.
# The default below is fine for local dev, but the app REFUSES TO START if
# ENVIRONMENT=production and API_KEY is still the default — this catches
# the single most common deployment mistake (forgetting to set a real key)
# at boot time instead of silently shipping an open API.
_DEFAULT_DEV_KEY = "dev-key-change-me"
_API_KEY = os.environ.get("API_KEY", _DEFAULT_DEV_KEY)

if _ENVIRONMENT == "production" and _API_KEY == _DEFAULT_DEV_KEY:
    raise RuntimeError(
        "Refusing to start: ENVIRONMENT=production but API_KEY is still the default "
        "dev value. Set a real API_KEY as an environment variable / platform secret "
        "(Render: Environment tab; Hugging Face Spaces: Settings > Repository secrets) "
        "before deploying."
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    return True


class ChurnRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    total_charges: float = Field(ge=0, le=50000)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    has_tech_support: bool
    senior_citizen: bool

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_months": 5,
                "monthly_charges": 85.5,
                "total_charges": 427.5,
                "contract_type": "month-to-month",
                "internet_service": "Fiber optic",
                "has_tech_support": False,
                "senior_citizen": False,
            }
        }


class ChurnResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    model_version: str


class ExplanationResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    model_version: str
    feature_contributions: dict


# --- Model loading: two backends behind one interface ---------------------

def _load_stage_from_mlflow(stage: str):
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(config.REGISTERED_MODEL_NAME, stages=[stage])
    if not versions:
        return None, None
    version = versions[0].version
    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, str(version)


def _load_stage_from_local(stage_dirname: str):
    path = os.path.join(_LOCAL_MODEL_DIR, stage_dirname)
    if not os.path.isdir(path):
        return None, None
    model = mlflow.sklearn.load_model(path)  # loading a local dir needs no tracking server

    version = None
    meta_path = os.path.join(_LOCAL_MODEL_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        key = "production" if stage_dirname == "production" else "staging"
        version = meta.get(key, {}).get("version")
    return model, version


def _load_models():
    global _production_model, _production_version, _shadow_model, _shadow_version, _explainer

    if _MODEL_SOURCE == "local":
        _production_model, _production_version = _load_stage_from_local("production")
        if _production_model is None:
            raise RuntimeError(
                "MODEL_SOURCE=local but no baked-in model found at model_artifacts/production. "
                "Run `python src/training/export_model.py` locally, commit model_artifacts/, "
                "and rebuild the image."
            )
        _shadow_model, _shadow_version = _load_stage_from_local("staging")
        logger.info("model_loaded", extra={"source": "local", "stage": "Production", "version": _production_version})
    else:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        _production_model, _production_version = _load_stage_from_mlflow("Production")
        if _production_model is None:
            raise RuntimeError(
                "No Production model found in the registry. Run `python src/training/train.py` first."
            )
        _shadow_model, _shadow_version = _load_stage_from_mlflow("Staging")
        logger.info("model_loaded", extra={"source": "mlflow", "stage": "Production", "version": _production_version})

    if _shadow_model is not None:
        logger.info("model_loaded", extra={"stage": "Staging", "version": _shadow_version})
    else:
        logger.info("no_staging_model_found")

    _explainer = None  # rebuilt lazily on first /explain call, per loaded production model


@app.on_event("startup")
def startup_event():
    _load_models()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_source": _MODEL_SOURCE,
        "environment": _ENVIRONMENT,
        "production_model_loaded": _production_model is not None,
        "production_version": _production_version,
        "shadow_model_loaded": _shadow_model is not None,
        "shadow_version": _shadow_version,
    }


@app.post("/predict", response_model=ChurnResponse, dependencies=[Depends(require_api_key)])
def predict(request: ChurnRequest):
    if _production_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    input_df = pd.DataFrame([request.model_dump()])[config.ALL_FEATURE_COLUMNS]

    start = time.perf_counter()
    prediction = int(_production_model.predict(input_df)[0])
    probability = float(_production_model.predict_proba(input_df)[0][1])
    latency_ms = (time.perf_counter() - start) * 1000

    request_id = log_prediction(
        features=request.model_dump(),
        prediction=prediction,
        probability=probability,
        model_version=str(_production_version),
        latency_ms=round(latency_ms, 3),
    )

    # Shadow scoring: score the challenger too, but never let it affect the
    # response the caller sees. This is what lets you validate a candidate
    # model against real traffic risk-free.
    if _shadow_model is not None:
        shadow_prediction = int(_shadow_model.predict(input_df)[0])
        shadow_probability = float(_shadow_model.predict_proba(input_df)[0][1])
        log_shadow_prediction(
            request_id=request_id,
            production_prediction=prediction,
            production_probability=probability,
            production_version=str(_production_version),
            shadow_prediction=shadow_prediction,
            shadow_probability=shadow_probability,
            shadow_version=str(_shadow_version),
        )

    return ChurnResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        model_version=str(_production_version),
    )


@app.post("/explain", response_model=ExplanationResponse, dependencies=[Depends(require_api_key)])
def explain(request: ChurnRequest):
    """
    Returns per-feature SHAP contributions for this specific prediction —
    answers "why did the model predict this?" for a single customer, which
    is what a support/retention team would actually need, not just a global
    feature importance ranking.
    """
    global _explainer
    if _production_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    import shap
    import numpy as np

    input_df = pd.DataFrame([request.model_dump()])[config.ALL_FEATURE_COLUMNS]

    preprocessor = _production_model.named_steps["prep"]
    classifier = _production_model.named_steps["clf"]
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    if _explainer is None:
        # Use SHAP's unified Explainer API (auto-selects TreeExplainer for
        # tree models, a fast linear explainer for linear models, and falls
        # back to a model-agnostic permutation explainer otherwise) — this
        # works consistently regardless of which candidate model won
        # training, so /explain doesn't break when the champion model type
        # changes after a retrain. Background sample is read from the
        # training set, which must be present locally OR baked into the
        # deploy image (the Dockerfile copies data/ in for this reason).
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        background_df = train_df.sample(n=min(50, len(train_df)), random_state=0)[config.ALL_FEATURE_COLUMNS]
        background = preprocessor.transform(background_df)
        if hasattr(background, "toarray"):
            background = background.toarray()
        _explainer = shap.Explainer(classifier.predict_proba, background)

    explanation = _explainer(transformed)
    values = np.array(explanation.values)[0]  # first (only) row
    if values.ndim > 1:
        values = values[:, 1]  # class-1 (churn) contributions

    contributions = {
        str(name): round(float(val), 4)
        for name, val in zip(feature_names, np.ravel(values))
    }
    # sort by absolute contribution, most influential first
    contributions = dict(
        sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )

    prediction = int(_production_model.predict(input_df)[0])
    probability = float(_production_model.predict_proba(input_df)[0][1])

    return ExplanationResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        model_version=str(_production_version),
        feature_contributions=contributions,
    )


@app.post("/reload-model", dependencies=[Depends(require_api_key)])
def reload_model():
    """
    Re-reads models from the current source (live registry, or the local
    model_artifacts/ directory). In MODEL_SOURCE=local mode this does NOT
    pull anything new over the network — it just re-reads local disk, which
    only changes if you rebuild/redeploy the image. That's a deliberate
    limitation of the deployed mode, not a bug: the deployed container is
    meant to be self-contained.
    """
    _load_models()
    return {"status": "reloaded", "production_version": _production_version, "shadow_version": _shadow_version}


# --- Monitoring dashboard endpoints ---------------------------------------
# SECURITY NOTE: these return real (or real-shaped) customer feature data
# and are therefore behind the same API key as the scoring endpoints — this
# was a gap in an earlier version of this project (metrics were public) that
# got closed here. Only /health and /dashboard (the static HTML shell, which
# contains no data itself) remain unauthenticated. The dashboard's JavaScript
# prompts for the key once and attaches it to every /metrics/* call.

@app.get("/metrics/summary", dependencies=[Depends(require_api_key)])
def metrics_summary():
    return get_dashboard_summary()


@app.get("/metrics/predictions", dependencies=[Depends(require_api_key)])
def metrics_predictions(limit: int = 200):
    df = load_recent_predictions(limit=limit)
    return df.to_dict(orient="records")


@app.get("/metrics/shadow", dependencies=[Depends(require_api_key)])
def metrics_shadow(limit: int = 200):
    df = load_shadow_comparison(limit=limit)
    return df.to_dict(orient="records")


@app.get("/metrics/drift", dependencies=[Depends(require_api_key)])
def metrics_drift():
    from src.monitoring.drift_check import run_drift_check

    try:
        return run_drift_check()
    except FileNotFoundError as e:
        return {"status": "no_baseline", "message": str(e), "drift_detected": False}


_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")


@app.get("/dashboard")
def dashboard():
    """Serves the static dashboard shell — contains no data itself, so no
    auth required here. The page's own JavaScript prompts for an API key
    before it can successfully call any /metrics/* endpoint."""
    return FileResponse(_DASHBOARD_PATH)
