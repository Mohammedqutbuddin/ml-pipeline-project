"""
Manually trains and stages a second candidate model as a "Staging" challenger
— useful for demoing/testing shadow deployment locally without waiting for a
real future retrain to naturally produce a better model.

In a real deployment, staging happens automatically inside train.py whenever
a retrain produces a model that beats the current Production model. This
script exists purely so you can see the shadow-deployment flow working
end-to-end right now, on demand.

Usage:
    python scripts/stage_demo_challenger.py
"""
import os
import sys

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.training.train import build_preprocessor, evaluate


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # confirm a Production model actually exists first
    prod_versions = client.get_latest_versions(config.REGISTERED_MODEL_NAME, stages=["Production"])
    if not prod_versions:
        print("No Production model found. Run `python src/training/train.py` first.")
        return

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    X_train, y_train = train_df[config.ALL_FEATURE_COLUMNS], train_df[config.TARGET_COLUMN]
    X_test, y_test = test_df[config.ALL_FEATURE_COLUMNS], test_df[config.TARGET_COLUMN]

    # A deliberately different Random Forest config — different random_state
    # and depth than the ones train.py tries, so it's a genuinely distinct
    # candidate rather than a duplicate.
    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=15, class_weight="balanced", random_state=7)),
    ])

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name="demo_challenger_rf") as run:
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_param("model_type", "demo_challenger_rf")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id

    print(f"Trained demo challenger: f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Staged as version {mv.version}.")
    print("\nNext steps:")
    print("  1. Restart (or hit /reload-model on) your serving app so it picks up the Staging model")
    print("  2. Send traffic: python scripts/generate_traffic.py --n 40")
    print("  3. Check agreement: python src/training/promote_staging.py")


if __name__ == "__main__":
    main()
