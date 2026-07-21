"""
Logs every prediction request/response to SQLite. This log is what the drift
checker and monitoring dashboard read to see what the model is actually
seeing in production. Also logs shadow (challenger) model predictions
separately, so a staged candidate model's behavior can be compared against
the live production model before it ever serves real traffic.
"""
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

logger = logging.getLogger("churn_pipeline.logging")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    features_json TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    model_version TEXT,
    latency_ms REAL
);

CREATE TABLE IF NOT EXISTS shadow_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    request_id INTEGER NOT NULL,
    production_prediction INTEGER NOT NULL,
    production_probability REAL NOT NULL,
    production_version TEXT,
    shadow_prediction INTEGER NOT NULL,
    shadow_probability REAL NOT NULL,
    shadow_version TEXT,
    agree INTEGER NOT NULL
);
"""


def get_connection():
    os.makedirs(os.path.dirname(config.PREDICTIONS_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.PREDICTIONS_DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def log_prediction(
    features: dict,
    prediction: int,
    probability: float,
    model_version: str = "unknown",
    latency_ms: float = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO predictions (timestamp, features_json, prediction, probability, model_version, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(features),
            int(prediction),
            float(probability),
            model_version,
            latency_ms,
        ),
    )
    conn.commit()
    request_id = cursor.lastrowid
    conn.close()
    logger.info(
        "prediction_logged",
        extra={
            "request_id": request_id,
            "model_version": model_version,
            "prediction": prediction,
            "latency_ms": latency_ms,
        },
    )
    return request_id


def log_shadow_prediction(
    request_id: int,
    production_prediction: int,
    production_probability: float,
    production_version: str,
    shadow_prediction: int,
    shadow_probability: float,
    shadow_version: str,
):
    agree = int(production_prediction == shadow_prediction)
    conn = get_connection()
    conn.execute(
        "INSERT INTO shadow_predictions "
        "(timestamp, request_id, production_prediction, production_probability, production_version, "
        "shadow_prediction, shadow_probability, shadow_version, agree) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            request_id,
            int(production_prediction),
            float(production_probability),
            production_version,
            int(shadow_prediction),
            float(shadow_probability),
            shadow_version,
            agree,
        ),
    )
    conn.commit()
    conn.close()
    if not agree:
        logger.info(
            "shadow_disagreement",
            extra={
                "request_id": request_id,
                "production_version": production_version,
                "shadow_version": shadow_version,
            },
        )


def load_recent_predictions(limit: int = 1000):
    import pandas as pd

    conn = get_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    if len(df):
        features_df = pd.json_normalize(df["features_json"].apply(json.loads))
        df = pd.concat([df.drop(columns=["features_json"]), features_df], axis=1)
    return df


def load_shadow_comparison(limit: int = 1000):
    import pandas as pd

    conn = get_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM shadow_predictions ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return df


def get_dashboard_summary() -> dict:
    """Aggregate stats used by the monitoring dashboard's /metrics/summary endpoint."""
    preds = load_recent_predictions(limit=2000)
    shadow = load_shadow_comparison(limit=2000)

    summary = {
        "total_predictions": int(len(preds)),
        "churn_rate": round(float(preds["prediction"].mean()), 4) if len(preds) else None,
        "avg_probability": round(float(preds["probability"].mean()), 4) if len(preds) else None,
        "avg_latency_ms": round(float(preds["latency_ms"].dropna().mean()), 2)
        if len(preds) and preds["latency_ms"].notna().any()
        else None,
        "current_model_version": str(preds.iloc[0]["model_version"]) if len(preds) else None,
        "shadow_active": bool(len(shadow)),
        "shadow_agreement_rate": round(float(shadow["agree"].mean()), 4) if len(shadow) else None,
        "shadow_n_compared": int(len(shadow)),
    }
    return summary
