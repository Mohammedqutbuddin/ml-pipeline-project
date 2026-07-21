"""
Generates a synthetic but realistic customer churn dataset, schema-aligned
with the real Kaggle/IBM "Telco Customer Churn" dataset so that switching to
real data (via prepare_real_data.py) requires no other code changes.

This is the fallback/demo path — for the real dataset, see
src/data/prepare_real_data.py and SETUP_AND_IMPROVEMENTS.md.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config


def generate_churn_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 72, n_rows)
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"], n_rows, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.45, 0.20]
    )
    monthly_charges = np.round(rng.normal(70, 25, n_rows).clip(15, 250), 2)
    total_charges = np.round(monthly_charges * tenure_months + rng.normal(0, 50, n_rows), 2).clip(0)
    has_tech_support = rng.choice([True, False], n_rows, p=[0.4, 0.6])
    senior_citizen = rng.choice([True, False], n_rows, p=[0.16, 0.84])

    # churn probability driven by realistic signal: short tenure, month-to-month,
    # fiber optic (historically higher churn in this dataset), no tech support,
    # senior citizens skew slightly higher churn
    churn_logit = (
        -0.04 * tenure_months
        + np.where(contract_type == "month-to-month", 1.2, 0.0)
        + np.where(contract_type == "one-year", 0.3, 0.0)
        + np.where(internet_service == "Fiber optic", 0.6, 0.0)
        + np.where(internet_service == "No", -0.4, 0.0)
        + np.where(has_tech_support, -0.5, 0.3)
        + np.where(senior_citizen, 0.3, 0.0)
        + 0.01 * (monthly_charges - 70)
        - 1.0
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "has_tech_support": has_tech_support,
            "senior_citizen": senior_citizen,
            "churned": churned,
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df = generate_churn_data()

    # simple train/test split, stratified on target
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    df.to_csv(config.RAW_DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"Generated {len(df)} rows -> {config.RAW_DATA_PATH}")
    print(f"Train: {len(train_df)} rows -> {config.TRAIN_DATA_PATH}")
    print(f"Test:  {len(test_df)} rows -> {config.TEST_DATA_PATH}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
