"""
Compares recent live prediction traffic against the training baseline to
detect feature drift. Uses Population Stability Index (PSI) as the core
metric (simple, interpretable, industry-standard) and optionally generates a
richer HTML report via Evidently if installed.

Run with: python src/monitoring/drift_check.py
Exits with code 1 if drift exceeds the threshold — this is what a CI job
checks to decide whether to trigger retraining.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.monitoring.log_predictions import load_recent_predictions


def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    PSI < 0.1  -> no significant shift
    0.1 - 0.2  -> moderate shift, worth watching
    > 0.2      -> significant shift, action recommended
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    expected_pct, _ = np.histogram(expected, bins=breakpoints)
    actual_pct, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct) / len(expected)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct) / len(actual)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def run_drift_check() -> dict:
    if not os.path.exists(config.BASELINE_STATS_PATH):
        raise FileNotFoundError(
            "No baseline stats found. Run `python src/training/train.py` first — it saves "
            "baseline stats after promoting a model."
        )

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    recent = load_recent_predictions(limit=1000)

    if len(recent) < 30:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(recent)} logged predictions; need >= 30 to assess drift.",
            "drift_detected": False,
        }

    results = {}
    max_psi = 0.0
    for col in config.NUMERIC_FEATURES:
        if col not in recent.columns:
            continue
        psi = population_stability_index(train_df[col].values, recent[col].values)
        results[col] = round(psi, 4)
        max_psi = max(max_psi, psi)

    drift_detected = max_psi >= config.DRIFT_PSI_THRESHOLD

    report = {
        "status": "ok",
        "n_recent_predictions": len(recent),
        "psi_per_feature": results,
        "max_psi": round(max_psi, 4),
        "threshold": config.DRIFT_PSI_THRESHOLD,
        "drift_detected": drift_detected,
    }

    # Optional richer HTML report if evidently is installed
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        common_cols = [c for c in config.NUMERIC_FEATURES if c in recent.columns]
        evidently_report = Report(metrics=[DataDriftPreset()])
        evidently_report.run(
            reference_data=train_df[common_cols], current_data=recent[common_cols]
        )
        evidently_report.save_html(os.path.join(config.BASE_DIR, "monitoring_report.html"))
        report["html_report"] = "monitoring_report.html"
    except ImportError:
        report["html_report"] = None
        report["note"] = "Install `evidently` for a richer HTML drift report."

    return report


if __name__ == "__main__":
    report = run_drift_check()
    print(json.dumps(report, indent=2))

    if report.get("drift_detected"):
        print(f"\nDRIFT DETECTED (max PSI {report['max_psi']} >= threshold {report['threshold']}). "
              "Retraining should be triggered.")
        sys.exit(1)
    else:
        print("\nNo significant drift detected.")
        sys.exit(0)
