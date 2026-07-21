# Churn Prediction MLOps Pipeline

A production-grade machine learning pipeline demonstrating enterprise ML engineering practices: data validation, experiment tracking, model registry with promotion gates, shadow deployment, live monitoring, drift detection, and automated retraining—built entirely on free tooling and CPU-only infrastructure.

This is a **portfolio project showcasing ML engineering skills**, not just model training. It answers the questions interviewers actually ask about production ML systems.

---

## Why This Project Stands Out

Most churn-prediction portfolios stop at "trained a model, got 85% accuracy." This pipeline demonstrates what separates ML engineers from data scientists:

| Interviewer Question | How This Project Answers It |
|---|---|
| **"How do you validate data quality before training?"** | Schema validation + null rate + range + duplicate checks. Training refuses to run on bad data. |
| **"How do you know a new model is actually better?"** | MLflow tracks every candidate. Only models beating the current production model are considered. |
| **"What if a model looks good offline but fails in production?"** | Shadow deployment stages challengers and scores them silently on real traffic before promotion. |
| **"How do you explain predictions to stakeholders?"** | `/explain` endpoint returns per-customer SHAP contributions—actionable, not just a metric. |
| **"How do you detect model performance decay?"** | Population Stability Index (PSI) tracks feature drift in real-time against training baseline. |
| **"Is your serving layer secured?"** | API-key auth on every data-returning endpoint. App refuses to boot with default credentials. |
| **"What happens when drift is detected?"** | GitHub Actions workflow automatically triggers retraining—not just an alert, actual remediation. |

---

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────────┐
│  Data Layer  │────▶│  Validation  │────▶│   Training    │────▶│ Model Registry │
│ (CSV/gen/   │     │ (schema +    │     │ (scikit-learn │     │ (MLflow: Prod/ │
│  real Telco) │     │  quality     │     │  + MLflow)    │     │  Staging)      │
└─────────────┘     └──────────────┘     └───────────────┘     └────────┬────────┘
                                                                         │
                     new challenger beats prod?  ──────▶ stage, don't promote
                                                         (shadow deployment)
                                                                         │
                                          ┌──────────────────────────────┴──────────────────────────────┐
                                          ▼                                                              ▼
                        ┌──────────────────────────────┐                            ┌──────────────────────────────┐
                        │  LOCAL DEV: Serve live from  │                            │ DEPLOY: Export model, bake   │
                        │  registry (MODEL_SOURCE=     │                            │ into Docker (MODEL_SOURCE=   │
                        │  mlflow) · Full shadow flow  │                            │ local) · No live MLflow      │
                        └────────────┬─────────────────┘                            └──────────────┬───────────────┘
                                     └──────────────────────┬──────────────────────────────────────┘
                                                            ▼
                        ┌─────────────────────────────────────────────────┐
                        │         FastAPI Serving Layer                   │
                        │  • API-key auth on data endpoints               │
                        │  • Shadow scoring + SHAP explanations           │
                        │  • Fails fast with default credentials          │
                        └──────────────────────┬────────────────────────┘
                                               ▼
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│   Retraining │◀────│    Drift     │◀────│  Monitoring +   │◀────│   Predictions  │
│  Trigger (GA │     │  Detection   │     │  Logging        │     │   & Shadow     │
│  scheduled)  │     │  (PSI +      │     │  (SQLite)       │     │   Comparisons  │
└──────────────┘     │  Evidently)  │     └─────────────────┘     └────────────────┘
                     └──────────────┘
```

**End-to-end flow:**
1. New data arrives (synthetic or real Telco dataset)
2. Validation gate checks schema, nulls, ranges, duplicates
3. Training runs candidates, logs all metrics to MLflow
4. If a candidate beats production on test set → stage it (not promote yet)
5. Two serving paths:
   - **Local dev**: Serve live from MLflow registry + shadow deployment
   - **Deploy**: Export winning model, bake into Docker image
6. FastAPI serves production predictions while silently scoring staging challenger
7. Every prediction logged to SQLite + shown on operations dashboard
8. Drift detector (PSI) compares recent traffic against training baseline
9. If drift detected → GitHub Actions workflow auto-retrains

---

## Tech Stack

| Component | Tool | Rationale |
|---|---|---|
| **Experiment Tracking & Registry** | MLflow (Production + Staging stages) | Industry standard; Staging stage enables shadow deployment |
| **Model Training** | scikit-learn | Fast on CPU, no GPU required, proven in production |
| **Data Validation** | Custom validation module | Zero external dependencies, explicit contracts |
| **Data Source** | Synthetic generator + real Kaggle/IBM Telco dataset | Reproducible + realistic |
| **Serving** | FastAPI + Uvicorn | Async, lightweight, easy containerization |
| **Model Loading** | Dual-mode: MLflow registry (dev) or local artifacts (deploy) | `MODEL_SOURCE` env var switches modes |
| **Explainability** | SHAP (permutation explainer) | Per-prediction reasoning, model-agnostic |
| **Shadow Deployment** | Custom SQLite logging | Real traffic comparison, zero additional infrastructure |
| **Monitoring & Drift** | Population Stability Index (custom) + Evidently AI | PSI is dependency-free; Evidently adds rich HTML reports |
| **Dashboard** | Hand-built HTML/CSS/JS + FastAPI | Full design control, intentional UX |
| **Authentication** | API-key headers | Required on every data endpoint, enforced at startup |
| **Logging** | Structured JSON (stdlib) + SQLite | Production-grade format (Datadog, CloudWatch compatible) |
| **Containerization** | Docker + docker-compose | Two configs: dev (live registry) and deploy (baked model) |
| **CI/CD** | GitHub Actions | Trains, tests, exports model, smoke-tests deploy path on every push |
| **Hosting** | Render or Hugging Face Spaces (free tier) | Verified deployment targets |

---

## Verified Test Results

Every number below was measured while building this project—not placeholder estimates.

### Unit & Integration Tests

```
============================= test session starts ==============================
collecting ... collected 12 items

tests/test_data_validation.py::test_generated_data_passes_validation        PASSED
tests/test_data_validation.py::test_validation_catches_missing_columns      PASSED
tests/test_data_validation.py::test_validation_catches_out_of_range_values  PASSED
tests/test_data_validation.py::test_validation_catches_bad_categorical      PASSED
tests/test_data_validation.py::test_validation_catches_too_few_rows         PASSED
tests/test_training.py::test_candidates_can_fit_and_predict                 PASSED
tests/test_training.py::test_evaluate_returns_all_expected_metrics          PASSED
tests/test_serving.py::test_health_reports_loaded_model                     PASSED
tests/test_serving.py::test_predict_requires_api_key                        PASSED
tests/test_serving.py::test_predict_with_valid_key_returns_prediction       PASSED
tests/test_serving.py::test_predict_rejects_invalid_contract_type           PASSED
tests/test_serving.py::test_dashboard_serves_html                           PASSED

======================= 12 passed in 7.66s ========================
```

**Run tests locally:**
```bash
pytest tests/ -v
# Set RUN_SERVING_TESTS=1 for serving layer tests
RUN_SERVING_TESTS=1 pytest tests/ -v
```

### Model Training Results

Three candidate models trained and evaluated:

| Model | F1 Score | ROC-AUC |
|---|---|---|
| **Logistic Regression** (selected) | 0.5553 | 0.7977 |
| Random Forest | 0.5321 | 0.7766 |
| Gradient Boosting | 0.4035 | 0.7795 |

Logistic regression automatically promoted based on F1 score. The promotion gate compares every candidate against production before allowing advancement.

### Shadow Deployment Results

A below-threshold Random Forest (F1 = 0.4982) was staged and silently scored on real request traffic:

```
Shadow Comparison (60 real requests):
  Agreement Rate with Production:  66.7%
  Disagreements:                   20
  Avg Probability Gap:             ~0.35
```

This is the decision metric a human reviews before promotion—demonstrating that offline F1 alone misses behavioral differences that matter in production.

### Drift Detection Results

Artificially shifted feature distribution (`monthly_charges` from different range):

```json
{
  "psi_per_feature": {
    "tenure_months": 0.31,
    "monthly_charges": 14.37,
    "total_charges": 3.77
  },
  "max_psi": 14.37,
  "threshold": 0.2,
  "drift_detected": true
}
```

PSI interpretation: `< 0.1` = no shift, `0.1–0.2` = moderate shift, `> 0.2` = significant shift.
Detector correctly stayed silent below 30 logged predictions and triggered on drift.

### Serving Performance

| Metric | Measured Value |
|---|---|
| `/predict` latency (avg) | 8–15 ms (CPU inference only) |
| `/explain` latency (SHAP) | ~9.3 s (permutation explainer, 50-row background) |
| Auth enforcement | ✓ Unauthenticated requests return 401 |
| Input validation | ✓ Invalid `contract_type` returns 422 |

**Performance note:** SHAP permutation explainer is slow by design (model-agnostic). A model-specific explainer (e.g., `LinearExplainer` for logistic regression) would hit <100ms—documented as a known trade-off rather than hidden.

---

## Project Structure

```
ml-pipeline-project/
├── README.md                          # Overview, architecture, results
├── SETUP_GUIDE.md                     # Local setup + troubleshooting
├── GITHUB_AND_DEPLOYMENT.md           # GitHub, real data, deployment, security audit
├── SETUP_AND_IMPROVEMENTS.md          # Roadmap for extensions
│
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Deploy-ready: bakes model_artifacts/
├── .dockerignore
├── docker-compose.yml                 # Local dev with live MLflow registry
├── .gitignore
│
├── .github/workflows/
│   ├── ci.yml                         # Trains, tests, exports, smoke-tests on push
│   └── retrain.yml                    # Scheduled drift check + auto-retrain
│
├── scripts/
│   └── generate_traffic.py            # Simulates realistic request traffic
│
├── model_artifacts/                   # Committed: winning model (export_model.py)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── metadata.json
│
├── src/
│   ├── config.py                      # Centralized: schema, paths, thresholds
│   ├── logging_config.py              # Structured JSON logging
│   │
│   ├── data/
│   │   ├── generate_data.py           # Synthetic data generator
│   │   ├── prepare_real_data.py       # Telco dataset adapter (handles messiness)
│   │   └── validate_data.py           # Schema + quality contract enforcement
│   │
│   ├── training/
│   │   ├── train.py                   # Candidate training + MLflow logging
│   │   ├── promote_staging.py         # Reviews shadow results, promotes on command
│   │   └── export_model.py            # Bakes registry model into model_artifacts/
│   │
│   ├── serving/
│   │   ├── app.py                     # FastAPI: dual-mode loading, auth, shadow scoring
│   │   ├── models.py                  # Pydantic schemas
│   │   └── dashboard.html             # Custom operations console
│   │
│   └── monitoring/
│       ├── drift_check.py             # PSI calculation + Evidently integration
│       └── log_predictions.py         # SQLite: predictions + shadow comparisons
│
└── tests/
    ├── test_data_validation.py        # Schema + quality gate tests
    ├── test_training.py               # Candidate training + evaluation
    └── test_serving.py                # Auth, validation, model loading
```

---

## Setup Guide

### Windows

**1. Install Python 3.10+**
```cmd
# Verify installation
python --version
```

**2. Create virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**3. Clone and install dependencies**
```cmd
git clone https://github.com/Mohammedqutbuddin/ml-pipeline-project.git
cd ml-pipeline-project
pip install -r requirements.txt
```

**4. Start MLflow tracking server (optional, for local dev)**
```cmd
mlflow ui
# Runs at http://localhost:5000
```

**5. Generate data and train**
```cmd
python -m src.data.generate_data
python -m src.training.train
```

**6. Run tests**
```cmd
pytest tests/ -v
```

**7. Start serving (after training)**
```cmd
python -m src.serving.app
# Runs at http://localhost:8000
# Set API_KEY environment variable first (see SETUP_GUIDE.md)
```

### macOS

**1. Install Python 3.10+ (via Homebrew)**
```bash
brew install python@3.10
python3 --version
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Clone and install dependencies**
```bash
git clone https://github.com/Mohammedqutbuddin/ml-pipeline-project.git
cd ml-pipeline-project
pip install -r requirements.txt
```

**4. Start MLflow tracking server (optional, for local dev)**
```bash
mlflow ui
# Runs at http://localhost:5000
```

**5. Generate data and train**
```bash
python -m src.data.generate_data
python -m src.training.train
```

**6. Run tests**
```bash
pytest tests/ -v
```

**7. Start serving**
```bash
python -m src.serving.app
# Runs at http://localhost:8000
```

### Linux (Ubuntu/Debian)

**1. Install Python 3.10+**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
python3 --version
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Clone and install dependencies**
```bash
git clone https://github.com/Mohammedqutbuddin/ml-pipeline-project.git
cd ml-pipeline-project
pip install -r requirements.txt
```

**4. Start MLflow tracking server**
```bash
mlflow ui
# Runs at http://localhost:5000
```

**5. Generate data and train**
```bash
python -m src.data.generate_data
python -m src.training.train
```

**6. Run tests**
```bash
pytest tests/ -v
```

**7. Start serving**
```bash
python -m src.serving.app
```

### Docker (All Platforms)

**Local Development (with MLflow registry)**
```bash
docker-compose up --build
# MLflow UI: http://localhost:5000
# FastAPI: http://localhost:8000
# Dashboard: http://localhost:8000/dashboard
```

**Production Deployment (baked model, no MLflow server)**
```bash
docker build -t ml-pipeline:latest .
docker run -p 8000:8000 \
  -e API_KEY="your-secure-key" \
  -e MODEL_SOURCE="local" \
  ml-pipeline:latest
```

---

## Usage Examples

### Generate Data & Train

```bash
# Generate synthetic data
python -m src.data.generate_data --output data/synthetic.csv --rows 10000

# Or use real Telco dataset
python -m src.data.prepare_real_data

# Train candidates
python -m src.training.train --data data/synthetic.csv --mlflow-uri http://localhost:5000
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 12,
    "monthly_charges": 65.5,
    "total_charges": 786.0,
    "internet_service": "fiber_optic"
  }'
```

### Get SHAP Explanation

```bash
curl -X POST http://localhost:8000/explain \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 12,
    "monthly_charges": 65.5,
    "total_charges": 786.0,
    "internet_service": "fiber_optic"
  }'

# Returns: per-feature SHAP contributions
```

### Check Drift

```bash
curl http://localhost:8000/metrics/drift \
  -H "X-API-Key: your-api-key"

# Returns: PSI per feature, max PSI, drift_detected flag
```

### View Operations Dashboard

```
http://localhost:8000/dashboard
# Real-time predictions, shadow comparisons, drift status
```

---

## Key Features

- **Data Validation Gate** — Schema, null rates, ranges, and duplicates checked before training. Explicit data contracts prevent garbage-in-garbage-out.

- **Model Registry with Promotion Workflow** — MLflow tracks all candidates. Promotion requires beating the current production model on a held-out test set.

- **Shadow Deployment** — New models staged and scored silently on real traffic before any human decision to promote. Agreement rates demonstrate real-world behavioral differences.

- **Per-Prediction Explainability** — SHAP contributions for every prediction, enabling actionable insights for stakeholders.

- **Real-Time Drift Detection** — Population Stability Index (PSI) monitors feature distributions in production. Deviations trigger automated retraining via GitHub Actions.

- **Dual-Mode Model Loading** — Local development uses live MLflow registry (full workflow); deployed version bakes winning model into Docker image (zero runtime dependencies).

- **Custom Operations Dashboard** — Hand-built console showing live predictions, shadow comparisons, drift status, and historical trends.

- **Production-Grade Security** — API-key authentication on every data endpoint. App refuses to boot with default credentials.

---

## Documentation

- **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** — Detailed local setup + troubleshooting table
- **[GITHUB_AND_DEPLOYMENT.md](./GITHUB_AND_DEPLOYMENT.md)** — GitHub push, real data, free deployment, security audit
- **[SETUP_AND_IMPROVEMENTS.md](./SETUP_AND_IMPROVEMENTS.md)** — Roadmap (DVC versioning, alerting, load testing, rate limiting)

---

## License

MIT

---

## Author

**Mohammad Qutbuddin**  
B.Tech Computer Science Engineering (AI & ML)

For inquiries, collaborations, or to view more work: [GitHub Profile](https://github.com/Mohammedqutbuddin)
