import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.generate_data import generate_churn_data
from src.training.train import get_candidates, evaluate
from src import config


def test_candidates_can_fit_and_predict():
    df = generate_churn_data(n_rows=500)
    X = df[config.ALL_FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    candidates = get_candidates()
    for name, pipeline in candidates.items():
        pipeline.fit(X, y)
        metrics = evaluate(pipeline, X, y)
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_returns_all_expected_metrics():
    df = generate_churn_data(n_rows=300)
    X = df[config.ALL_FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    candidates = get_candidates()
    pipeline = candidates["logistic_regression"]
    pipeline.fit(X, y)
    metrics = evaluate(pipeline, X, y)

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
