"""
Lightweight data quality gate — the pipeline refuses to train on data that
fails these checks. This is a hand-rolled version of what tools like Great
Expectations / Pandera do, kept dependency-light on purpose so it's easy to
explain line-by-line in an interview.
"""
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config


class DataValidationError(Exception):
    pass


def validate(df: pd.DataFrame) -> dict:
    """Runs all checks and returns a report. Raises DataValidationError on hard failure."""
    report = {"checks": [], "passed": True}

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        report["checks"].append({"check": name, "status": status, "detail": detail})
        if not condition:
            report["passed"] = False

    # 1. Row count
    check(
        "min_row_count",
        len(df) >= config.MIN_ROWS_REQUIRED,
        f"got {len(df)}, need >= {config.MIN_ROWS_REQUIRED}",
    )

    # 2. Required columns present
    missing_cols = set(config.ALL_FEATURE_COLUMNS + [config.TARGET_COLUMN]) - set(df.columns)
    check("required_columns_present", len(missing_cols) == 0, f"missing: {missing_cols}")

    # 3. Null fraction per column
    for col in config.ALL_FEATURE_COLUMNS:
        if col in df.columns:
            null_frac = df[col].isnull().mean()
            check(
                f"null_fraction[{col}]",
                null_frac <= config.MAX_NULL_FRACTION,
                f"{null_frac:.2%} nulls (max {config.MAX_NULL_FRACTION:.0%})",
            )

    # 4. Numeric ranges
    for col, bounds in config.NUMERIC_FEATURES.items():
        if col in df.columns:
            out_of_range = ((df[col] < bounds["min"]) | (df[col] > bounds["max"])).mean()
            check(
                f"range[{col}]",
                out_of_range == 0,
                f"{out_of_range:.2%} of values outside [{bounds['min']}, {bounds['max']}]",
            )

    # 5. Categorical values are within the known set
    for col, allowed in config.CATEGORICAL_FEATURES.items():
        if col in df.columns:
            unknown = ~df[col].isin(allowed)
            check(
                f"categorical_domain[{col}]",
                unknown.sum() == 0,
                f"{unknown.sum()} rows with values outside {allowed}",
            )

    # 6. Target column is binary
    if config.TARGET_COLUMN in df.columns:
        valid_targets = df[config.TARGET_COLUMN].isin([0, 1]).all()
        check("target_is_binary", valid_targets)

    # 7. No fully duplicated rows
    dup_frac = df.duplicated().mean()
    check("duplicate_rows", dup_frac < 0.05, f"{dup_frac:.2%} duplicated rows")

    return report


if __name__ == "__main__":
    df = pd.read_csv(config.RAW_DATA_PATH)
    report = validate(df)

    print(json.dumps(report, indent=2))

    if not report["passed"]:
        failed = [c["check"] for c in report["checks"] if c["status"] == "FAIL"]
        raise DataValidationError(f"Data validation failed on: {failed}")

    print("\nAll data quality checks passed.")
