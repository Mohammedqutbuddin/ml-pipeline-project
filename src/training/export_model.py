"""
Exports the current Production (and Staging, if present) model from the live
MLflow registry to a self-contained local directory. This is the bridge step
between "local development with a live MLflow server" and "deployable
container with no runtime MLflow server dependency."

Run this locally, right before building a deployment image:

    python src/training/export_model.py

Then commit the output — model_artifacts/ is intentionally NOT gitignored,
because Render / Hugging Face Spaces build your Docker image from your git
repo, not your local disk. If model_artifacts/ isn't committed, the remote
build has nothing to COPY into the image.

    git add model_artifacts data/baseline_stats.json
    git commit -m "Export trained model for deployment"
    git push
"""
import json
import os
import shutil
import sys

import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

EXPORT_DIR = os.path.join(config.BASE_DIR, "model_artifacts")


def export_stage(client, stage: str, dest_dir: str):
    versions = client.get_latest_versions(config.REGISTERED_MODEL_NAME, stages=[stage])
    if not versions:
        return None

    version = versions[0]
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    mlflow.sklearn.save_model(model, dest_dir)

    run = client.get_run(version.run_id)
    return {
        "version": str(version.version),
        "stage": stage,
        "run_id": version.run_id,
        "metrics": dict(run.data.metrics),
    }


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    metadata = {}

    prod_meta = export_stage(client, "Production", os.path.join(EXPORT_DIR, "production"))
    if prod_meta is None:
        raise RuntimeError(
            "No Production model found in the registry. Run `python src/training/train.py` "
            "first, then re-run this export."
        )
    metadata["production"] = prod_meta
    print(f"Exported Production v{prod_meta['version']} (f1={prod_meta['metrics'].get('f1', '?'):.4f}) "
          f"-> model_artifacts/production")

    staging_meta = export_stage(client, "Staging", os.path.join(EXPORT_DIR, "staging"))
    if staging_meta:
        metadata["staging"] = staging_meta
        print(f"Exported Staging v{staging_meta['version']} (f1={staging_meta['metrics'].get('f1', '?'):.4f}) "
              f"-> model_artifacts/staging")
    else:
        print("No Staging model to export (none currently staged) — that's fine, "
              "the deployed app will just run without a shadow challenger.")

    with open(os.path.join(EXPORT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if not os.path.exists(config.BASELINE_STATS_PATH):
        print("\nWARNING: data/baseline_stats.json not found — drift detection will not "
              "work in the deployed app until you run train.py locally (it writes this "
              "file) and commit it alongside model_artifacts/.")

    print("\nExport complete. Before deploying, commit the exported artifacts:")
    print("  git add model_artifacts data/baseline_stats.json")
    print("  git commit -m \"Export trained model for deployment\"")
    print("  git push")
    print("\nThen build/test locally with:")
    print("  docker build -t churn-api .")
    print("  docker run -p 8000:8000 -e API_KEY=<a-real-key> -e ENVIRONMENT=production churn-api")


if __name__ == "__main__":
    main()
