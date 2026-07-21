import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.validate_data import validate
from src.data.generate_data import generate_churn_data


def test_generated_data_passes_validation():
    df = generate_churn_data(n_rows=500)
    report = validate(df)
    assert report["passed"], [c for c in report["checks"] if c["status"] == "FAIL"]


def test_validation_catches_missing_columns():
    df = generate_churn_data(n_rows=500).drop(columns=["monthly_charges"])
    report = validate(df)
    assert not report["passed"]


def test_validation_catches_out_of_range_values():
    df = generate_churn_data(n_rows=500)
    df.loc[0, "tenure_months"] = -5  # invalid
    report = validate(df)
    assert not report["passed"]


def test_validation_catches_bad_categorical_value():
    df = generate_churn_data(n_rows=500)
    df.loc[0, "contract_type"] = "lifetime-deal"  # not in allowed domain
    report = validate(df)
    assert not report["passed"]


def test_validation_catches_too_few_rows():
    df = generate_churn_data(n_rows=10)
    report = validate(df)
    assert not report["passed"]
