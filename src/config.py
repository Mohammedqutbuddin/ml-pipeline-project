"""
Central configuration for the pipeline. Keeping this in one place means every
stage (validation, training, serving, monitoring) agrees on schema and paths.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "churn_raw.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "churn_train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "churn_test.csv")
BASELINE_STATS_PATH = os.path.join(DATA_DIR, "baseline_stats.json")
PREDICTIONS_DB_PATH = os.path.join(BASE_DIR, "logs", "predictions.db")

# --- MLflow ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "churn-prediction"
REGISTERED_MODEL_NAME = "churn-classifier"

# --- Feature schema ---
# Used by validation, training, and the FastAPI request model.
# NOTE: this schema is aligned with the real Kaggle/IBM "Telco Customer Churn"
# dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) so that swapping in real data
# via src/data/prepare_real_data.py requires no changes here. num_support_tickets
# was dropped (that dataset has no such field) in favor of internet_service and
# senior_citizen, which are real, available signals.
NUMERIC_FEATURES = {
    "tenure_months": {"min": 0, "max": 120},
    "monthly_charges": {"min": 0, "max": 500},
    "total_charges": {"min": 0, "max": 50000},
}
CATEGORICAL_FEATURES = {
    "contract_type": ["month-to-month", "one-year", "two-year"],
    "internet_service": ["DSL", "Fiber optic", "No"],
}
BOOLEAN_FEATURES = ["has_tech_support", "senior_citizen"]
TARGET_COLUMN = "churned"

ALL_FEATURE_COLUMNS = (
    list(NUMERIC_FEATURES.keys())
    + list(CATEGORICAL_FEATURES.keys())
    + BOOLEAN_FEATURES
)

# --- Data quality thresholds (used by validate_data.py) ---
MAX_NULL_FRACTION = 0.02          # fail if any column has >2% nulls
MIN_ROWS_REQUIRED = 200

# --- Model promotion / retraining thresholds ---
MIN_F1_TO_PROMOTE = 0.55          # candidate must beat this to ever be promoted
DRIFT_PSI_THRESHOLD = 0.2         # population stability index threshold to flag drift
