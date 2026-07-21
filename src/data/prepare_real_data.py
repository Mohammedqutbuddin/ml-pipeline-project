"""
Transforms the real Kaggle/IBM "Telco Customer Churn" dataset into this
project's schema (src/config.py), so training, validation, serving, and
monitoring all work unchanged on real data.

WHY THIS SCRIPT EXISTS: the raw dataset doesn't match our schema out of the
box — column names differ, categorical values differ ("Month-to-month" vs
"month-to-month"), and it has real-world messiness (11 rows have a blank
string "" in TotalCharges instead of a number, for customers with zero
tenure — a classic real-data gotcha this script handles explicitly rather
than silently).

SETUP:
1. Download the dataset from Kaggle (free account required):
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   File: WA_Fn-UseC_-Telco-Customer-Churn.csv (7,043 rows, ~955 KB)
2. Place it at: data/raw_telco.csv
3. Run: python src/data/prepare_real_data.py

This produces data/churn_raw.csv, data/churn_train.csv, data/churn_test.csv
in the exact schema the rest of the pipeline expects — after this, every
other command in the README (validate_data.py, train.py, etc.) works
completely unchanged.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

RAW_KAGGLE_CSV_PATH = os.path.join(config.DATA_DIR, "raw_telco.csv")

_CONTRACT_MAP = {
    "Month-to-month": "month-to-month",
    "One year": "one-year",
    "Two year": "two-year",
}

# "No internet service" and "No phone service" are effectively "No" for our
# purposes — collapsing 3-value columns to boolean is a deliberate
# simplification, noted here rather than left implicit.
_YES_NO_MAP = {"Yes": True, "No": False, "No internet service": False, "No phone service": False}


def load_raw_kaggle_csv(path: str = RAW_KAGGLE_CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected the raw Kaggle CSV at {path}.\n"
            "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "(file: WA_Fn-UseC_-Telco-Customer-Churn.csv) and place it there, then re-run this script."
        )
    return pd.read_csv(path)


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # --- Real-data messiness #1: TotalCharges is a string column with 11
    # blank entries (customers with tenure == 0, i.e. brand new — they
    # haven't been charged yet, so there's no total). Coercing straight to
    # numeric would silently turn these into NaN; we make the fix explicit
    # and log how many rows it affected, which is the kind of thing you'd
    # actually want to know about a real dataset before training on it.
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    n_blank = df["TotalCharges"].isna().sum()
    if n_blank:
        print(f"Found {n_blank} blank TotalCharges values (new customers, tenure=0). "
              f"Imputing as 0.0 — a zero-tenure customer has no charges yet.")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # --- Real-data messiness #2: duplicate customerIDs, if any, would double
    # -count a customer in both train and test after splitting. Real Kaggle
    # exports occasionally have this; check explicitly rather than assume.
    n_dupes = df["customerID"].duplicated().sum()
    if n_dupes:
        print(f"Found {n_dupes} duplicate customerID rows — dropping duplicates, keeping first.")
        df = df.drop_duplicates(subset="customerID", keep="first")

    out = pd.DataFrame(
        {
            "tenure_months": df["tenure"].astype(int),
            "monthly_charges": df["MonthlyCharges"].astype(float),
            "total_charges": df["TotalCharges"].astype(float),
            "contract_type": df["Contract"].map(_CONTRACT_MAP),
            "internet_service": df["InternetService"],  # already "DSL"/"Fiber optic"/"No"
            "has_tech_support": df["TechSupport"].map(_YES_NO_MAP),
            "senior_citizen": df["SeniorCitizen"].astype(bool),  # already 0/1 in the raw data
            "churned": df["Churn"].map({"Yes": 1, "No": 0}),
        }
    )

    # --- Sanity check the mapping didn't silently introduce nulls (e.g. an
    # unexpected category value that wasn't in our map) before we hand this
    # off to the validation gate.
    problem_cols = [c for c in out.columns if out[c].isna().any()]
    if problem_cols:
        raise ValueError(
            f"Mapping produced unexpected nulls in columns: {problem_cols}. "
            "This usually means the raw dataset has a category value the mapping "
            "dictionaries above don't account for — inspect df['Contract'].unique() "
            "etc. and extend the maps."
        )

    return out


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)

    raw = load_raw_kaggle_csv()
    print(f"Loaded {len(raw)} raw rows from {RAW_KAGGLE_CSV_PATH}")

    df = transform(raw)
    print(f"Transformed to {len(df)} rows matching project schema.")
    print(f"Churn rate: {df['churned'].mean():.2%}")

    # same split strategy as generate_data.py, so downstream code is identical
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    df.to_csv(config.RAW_DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"\nWrote {config.RAW_DATA_PATH}, {config.TRAIN_DATA_PATH}, {config.TEST_DATA_PATH}")
    print("Next: python src/data/validate_data.py && python src/training/train.py")
