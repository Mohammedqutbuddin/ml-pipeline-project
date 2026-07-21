"""
Trains candidate models, logs everything to MLflow, and promotes the best
candidate to the "Production" stage in the MLflow Model Registry — but only
if it beats the current production model's F1 score. This promotion gate is
what prevents silent regressions from reaching the serving layer.
"""
import json
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.data.validate_data import validate, DataValidationError


def build_preprocessor():
    numeric_cols = list(config.NUMERIC_FEATURES.keys())
    categorical_cols = list(config.CATEGORICAL_FEATURES.keys())
    boolean_cols = config.BOOLEAN_FEATURES

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("bool", "passthrough", boolean_cols),
        ]
    )


def get_candidates():
    """Return a dict of candidate pipelines to try. Adding a new model = one line."""
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            [("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        "random_forest": Pipeline(
            [("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42))]
        ),
        "gradient_boosting": Pipeline(
            [("prep", preprocessor), ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42))]
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def get_current_production_f1(client, model_name: str) -> float:
    """Look up the F1 of whatever is currently in Production, if anything."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return -1.0
        run = client.get_run(versions[0].run_id)
        return run.data.metrics.get("f1", -1.0)
    except Exception:
        return -1.0


def main():
    # 1. Validate before training anything — fail fast on bad data
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    report = validate(train_df)
    if not report["passed"]:
        failed = [c["check"] for c in report["checks"] if c["status"] == "FAIL"]
        raise DataValidationError(f"Refusing to train: validation failed on {failed}")

    test_df = pd.read_csv(config.TEST_DATA_PATH)

    X_train = train_df[config.ALL_FEATURE_COLUMNS]
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df[config.ALL_FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COLUMN]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    candidates = get_candidates()
    results = []

    for name, pipeline in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_params({f"n_train_rows": len(X_train)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            results.append({"name": name, "run_id": run.info.run_id, **metrics})
            print(f"[{name}] f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    # 2. Pick the best candidate by F1
    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest candidate: {best['name']} (f1={best['f1']:.4f})")

    # 3. Promotion gate: only register + promote if it clears the bar AND beats prod
    current_prod_f1 = get_current_production_f1(client, config.REGISTERED_MODEL_NAME)
    print(f"Current production F1: {current_prod_f1:.4f}" if current_prod_f1 >= 0 else "No production model yet.")

    if best["f1"] < config.MIN_F1_TO_PROMOTE:
        print(f"Best candidate F1 {best['f1']:.4f} below minimum threshold "
              f"{config.MIN_F1_TO_PROMOTE} — NOT registering.")
        return

    if best["f1"] <= current_prod_f1:
        print("Best candidate does not beat current production model — NOT promoting. "
              "(Still logged in MLflow for comparison.)")
        return

    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)

    if current_prod_f1 < 0:
        # No production model exists yet — nothing to shadow-test against,
        # so go straight to Production.
        client.transition_model_version_stage(
            name=config.REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"No prior production model — promoted {best['name']} (version {mv.version}) "
              f"directly to Production.")
    else:
        # SHADOW DEPLOYMENT: a challenger that beats production on the offline
        # test set doesn't go straight to Production. It's staged instead, so
        # the serving layer can score it alongside the live model on real
        # traffic (without affecting what users see) before a human promotes
        # it. This catches gaps between offline eval and real-world behavior
        # that a single F1 number can hide.
        client.transition_model_version_stage(
            name=config.REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"Challenger {best['name']} (version {mv.version}, f1={best['f1']:.4f}) beat "
              f"current production (f1={current_prod_f1:.4f}) on the offline test set.")
        print("Staged as 'Staging' for shadow evaluation — it will be scored alongside "
              "Production on live traffic. Promote it manually once you're confident:")
        print(f"  python src/training/promote_staging.py --version {mv.version}")

    # 4. Save baseline feature stats for drift detection later
    baseline_stats = {
        col: {"mean": float(train_df[col].mean()), "std": float(train_df[col].std())}
        for col in config.NUMERIC_FEATURES
    }
    with open(config.BASELINE_STATS_PATH, "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"Baseline stats saved to {config.BASELINE_STATS_PATH}")


if __name__ == "__main__":
    main()
