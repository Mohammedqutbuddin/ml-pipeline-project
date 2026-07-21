"""
Reviews how a Staging (challenger) model has performed in shadow mode
against live traffic, then promotes it to Production if you decide it's
ready. This is the human-in-the-loop step after `train.py` stages a
challenger — shadow scoring de-risks the promotion by showing you real
agreement rates before the challenger ever serves a real user.

Usage:
    python src/training/promote_staging.py                  # just show the report
    python src/training/promote_staging.py --version 3       # promote version 3
"""
import argparse
import os
import sys

import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.monitoring.log_predictions import load_shadow_comparison


def print_shadow_report():
    df = load_shadow_comparison(limit=5000)
    if len(df) == 0:
        print("No shadow predictions logged yet. Make sure a Staging model exists and "
              "the serving app has received live traffic (shadow scoring happens "
              "automatically whenever a Staging model is present).")
        return None

    agreement_rate = df["agree"].mean()
    disagreements = df[df["agree"] == 0]

    print(f"Shadow comparison over {len(df)} requests:")
    print(f"  Agreement rate with Production: {agreement_rate:.2%}")
    print(f"  Disagreements: {len(disagreements)}")
    if len(disagreements):
        avg_prob_gap = (disagreements["shadow_probability"] - disagreements["production_probability"]).abs().mean()
        print(f"  Avg probability gap on disagreements: {avg_prob_gap:.4f}")
    return agreement_rate


def promote(version: str):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"\nPromoted version {version} to Production. Restart or hit the serving app's "
          f"/reload-model endpoint to pick it up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None,
                         help="Model version to promote from Staging to Production")
    args = parser.parse_args()

    print_shadow_report()

    if args.version:
        promote(args.version)
    else:
        print("\nNo --version passed, so nothing was promoted. Review the report above, "
              "then re-run with --version <n> when you're confident.")
