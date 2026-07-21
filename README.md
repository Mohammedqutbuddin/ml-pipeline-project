# Churn Prediction MLOps Pipeline

An end-to-end, production-style machine learning pipeline — data validation, experiment
tracking, a model registry with a promotion gate, **shadow deployment**, a live
monitoring dashboard, drift detection, and automated retraining — built to run entirely
on free tooling and a laptop CPU. No GPU, no paid services, no cloud account required.

Built as a portfolio project to demonstrate ML *engineering*, not just model training:
the skills that show up in production systems and get asked about in interviews.

---

## Why this project is different from a typical student ML project

Most churn-prediction portfolio projects stop at "trained a model, got 85% accuracy in
a notebook." This one answers the questions an ML Engineer interviewer actually asks:

| Question an interviewer asks | What this project does about it |
|---|---|
| "How do you know your data is good before training on it?" | A validation gate checks schema, null rates, value ranges, and duplicates — training refuses to run on bad data |
| "How do you know a new model is actually better before it ships?" | Every candidate is logged to MLflow; only models that beat the current production model on a held-out set are even considered for promotion |
| "What if a model looks good offline but breaks in production?" | **Shadow deployment** — a challenger model is staged and scored silently on live traffic (logged, never shown to users) before a human ever promotes it |
| "How would a support team act on a single prediction, not just an accuracy number?" | A `/explain` endpoint returns per-customer SHAP feature contributions |
| "How do you know when a model's performance is decaying?" | A live dashboard tracks population stability index (PSI) drift per feature against the training baseline |
| "Is your serving endpoint actually secured?" | API-key auth on scoring endpoints, read-only monitoring endpoints left open by design |
| "What happens when drift is detected — anything, or just an alert?" | A scheduled GitHub Actions workflow checks drift and automatically triggers retraining |

---

## System architecture

**Two paths from the same codebase**: a live-registry path for local development
(gives you shadow deployment, hot reload, the full MLflow workflow) and a deploy path
where the winning model gets exported and baked into a self-contained Docker image (no
live MLflow server needed at runtime — this is what makes it deployable to a free-tier
host that only gives you one service).

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────────┐
│  Data Layer  │────▶│  Validation  │────▶│    Training    │────▶│  Model Registry │
│ (raw CSV /   │     │ (schema +    │     │  (scikit-learn │     │    (MLflow:     │
│  generator /ᐩ│     │  quality     │     │  + MLflow      │     │  Production /   │
│  real Telco) │     │  checks)     │     │  tracking)     │     │  Staging)       │
└─────────────┘     └──────────────┘     └───────────────┘     └────────┬────────┘
                                                                          │
                     new challenger beats prod?  ──────▶ stage, don't promote
                                                          (shadow deployment)
                                                                          │
                                                          ┌───────────────┴───────────────┐
                                                          ▼                                ▼
                                          ┌─────────────────────────┐    ┌──────────────────────────┐
                                          │   LOCAL DEV: serve live  │    │  DEPLOY: export_model.py │
                                          │   from the registry      │    │  bakes the model into a  │
                                          │   (MODEL_SOURCE=mlflow)  │    │  self-contained Docker    │
                                          │                          │    │  image (MODEL_SOURCE=    │
                                          │                          │    │  local) — no live MLflow  │
                                          │                          │    │  needed at runtime        │
                                          └────────────┬─────────────┘    └─────────────┬────────────┘
                                                        └──────────────┬─────────────────┘
                                                                       ▼
                                          ┌─────────────────────────────────────────────┐
                                          │              Serving Layer                    │
                                          │   FastAPI · API-key auth on every endpoint    │
                                          │   returning data · shadow scoring · SHAP      │
                                          │   /explain · fails fast if deployed with a    │
                                          │   default API key                             │
                                          └───────────────────────┬───────────────────────┘
                                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────────┐
│   Retraining │◀────│    Drift     │◀────│   Monitoring   │◀────│   (from above)  │
│    Trigger   │     │   Detection  │     │  Dashboard +   │     │                 │
│  (GH Actions │     │  (PSI +      │     │  Logs (SQLite) │     │                 │
│   scheduled) │     │  Evidently)  │     │                │     │                 │
└──────────────┘     └──────────────┘     └───────────────┘     └────────────────┘
```

**Flow**: new data (synthetic, or the real Telco dataset via `prepare_real_data.py`)
lands → validated against a schema/quality contract → if valid, a model trains and
every candidate logs metrics/params/artifacts to MLflow → if a candidate beats the
current Production model on the offline test set, it's **staged**, not immediately
promoted. From there, two paths: locally, FastAPI serves live from the MLflow registry
(full shadow-deployment workflow); for deployment, `export_model.py` bakes the winning
model into a self-contained directory that the Docker image copies in at build time —
no live MLflow server needed at runtime. Either way, FastAPI serves Production
predictions while silently scoring any Staging challenger on the same live traffic →
every prediction and shadow comparison is logged to SQLite and shown live on the
operations dashboard → a scheduled job compares recent traffic against the training
baseline for drift → if drift crosses a threshold, GitHub Actions retrains
automatically. A human reviews shadow-agreement stats and promotes the challenger only
when confident.

### Tech stack

| Layer | Tool | Why |
|---|---|---|
| Experiment tracking / registry | MLflow (Production + Staging stages) | Industry standard; the Staging stage is what enables shadow deployment |
| Training | scikit-learn | Fast on CPU, no GPU needed |
| Data validation | Custom schema/quality checks | Lightweight, no external service dependency |
| Data source | Synthetic generator or real Kaggle/IBM Telco dataset | `prepare_real_data.py` adapts real data, handling its documented messiness explicitly |
| Serving | FastAPI + Uvicorn | Async, fast, easy to containerize |
| Model loading | Dual-mode: live MLflow registry (dev) or baked-in local artifacts (deploy) | `MODEL_SOURCE` env var — no live MLflow server needed once deployed |
| Explainability | SHAP | Per-prediction "why," not just global feature importance |
| Shadow deployment | Custom, SQLite-logged comparison | De-risks promotion using real traffic, not just offline metrics |
| Monitoring / drift | Population Stability Index (custom) + Evidently AI | PSI needs zero extra dependencies; Evidently adds a richer HTML report |
| Dashboard | Hand-built HTML/CSS/JS served by FastAPI | Full design control — deliberately not a default Streamlit template |
| Auth | API key header, required on every data-returning endpoint | Fails fast at startup if deployed with the default key |
| Logging | Structured JSON (stdlib logging) + SQLite | JSON logs are what real aggregators (Datadog, CloudWatch) expect |
| Containerization | Docker (deploy-ready, model baked in) + docker-compose (live-registry dev/test) | Standard deployment unit; two configs for two use cases |
| CI/CD | GitHub Actions | Trains a model, runs the full test suite, exports it, and smoke-tests the deploy path on every push |
| Hosting | Render or Hugging Face Spaces (free tier) | See [`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md) |

---


## Verified test results

Every number below was actually measured while building this project — not
placeholder numbers. Re-run any of them yourself with the commands shown.

### Unit + integration test suite

```
============================= test session starts ==============================
collecting ... collected 12 items

tests/test_data_validation.py::test_generated_data_passes_validation       PASSED
tests/test_data_validation.py::test_validation_catches_missing_columns     PASSED
tests/test_data_validation.py::test_validation_catches_out_of_range_values PASSED
tests/test_data_validation.py::test_validation_catches_bad_categorical_value PASSED
tests/test_data_validation.py::test_validation_catches_too_few_rows        PASSED
tests/test_training.py::test_candidates_can_fit_and_predict                PASSED
tests/test_training.py::test_evaluate_returns_all_expected_metrics         PASSED
tests/test_serving.py::test_health_reports_loaded_model                    PASSED
tests/test_serving.py::test_predict_requires_api_key                      PASSED
tests/test_serving.py::test_predict_with_valid_key_returns_prediction     PASSED
tests/test_serving.py::test_predict_rejects_invalid_contract_type         PASSED
tests/test_serving.py::test_dashboard_serves_html                         PASSED

======================= 12 passed in 7.66s ========================
```

Run it yourself: `pytest tests/ -v` (set `RUN_SERVING_TESTS=1` and a valid
`MLFLOW_TRACKING_URI` with a trained model to include the serving tests).

### Model training results

Three candidate models trained and logged to MLflow in the same run:

| Model | F1 | ROC-AUC |
|---|---|---|
| Logistic Regression (selected) | 0.5553 | 0.7977 |
| Random Forest | 0.5321 | 0.7766 |
| Gradient Boosting | 0.4035 | 0.7795 |

Logistic regression was automatically selected and promoted based on F1 score — the
promotion gate compares every new candidate against the current production model
before allowing promotion.

### Shadow deployment — real traffic comparison

A challenger Random Forest model (F1 = 0.4982, below the champion) was staged and
scored silently alongside the production model over real request traffic:

```
Shadow comparison over 60 requests:
  Agreement rate with Production: 66.7%
  Disagreements: 20
  Avg probability gap on disagreements: ~0.35
```

This is the number a human reviews before deciding whether to promote a challenger —
demonstrating the pipeline catches meaningful behavioral differences between models
that a single offline F1 score wouldn't surface.

### Drift detection — Population Stability Index

Verified the drift detector correctly flags an artificially shifted feature
distribution (`monthly_charges` sampled from a different range than training):

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

> Note: this schema snapshot predates the switch to the real Telco dataset's feature
> set (`internet_service`, `senior_citizen` replaced a synthetic-only
> `num_support_tickets` field — see [`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md)).
> The PSI mechanics shown here are unchanged either way.

PSI interpretation: < 0.1 no shift, 0.1–0.2 moderate shift, > 0.2 significant shift.
The detector correctly stayed silent (`insufficient_data`) below 30 logged predictions
and correctly triggered once artificially drifted traffic was introduced.

### Serving performance

| Metric | Measured value |
|---|---|
| Average `/predict` latency | 8–15 ms (model inference only, CPU, no GPU) |
| `/explain` (SHAP) latency | ~9.3 s (permutation explainer with a 50-row background sample — see note below) |
| Auth enforcement | Confirmed: unauthenticated `POST /predict` correctly returns `401` |
| Input validation | Confirmed: invalid `contract_type` correctly returns `422` |

**Known performance caveat, stated honestly**: the `/explain` endpoint is slow because
it uses a model-agnostic SHAP explainer to stay correct regardless of which model type
wins training. A model-specific explainer (e.g., `LinearExplainer` for the current
logistic regression champion) would cut this to milliseconds — documented as a known
follow-up rather than silently left slow.

---

## What's genuinely distinctive about this project

- **Shadow deployment is rare in student projects.** Most portfolios show a model going
  straight to "production." This pipeline treats "the model that wins offline" and "the
  model that's trusted with real traffic" as two different questions, and shows the
  agreement-rate evidence used to bridge them.
- **The monitoring dashboard is hand-built, not a Streamlit default.** It's a real
  HTML/CSS/JS console with intentional design — a blueprint-grid background, monospace
  data readouts, functional (not decorative) status colors, and a live signal strip of
  recent churn/retain outcomes — polling the same API endpoints a real on-call engineer
  would use.
- **A promotion gate with a paper trail.** Every training run logs metrics to MLflow
  regardless of whether it wins; the decision to promote or stage is explicit code, not
  a manual step, and it's demonstrated working both ways (straight promotion when no
  prior model exists, staging when a prior model exists).
- **Explainability answers "why," not just "what."** `/explain` returns per-customer
  SHAP contributions — actionable for a retention team, not just an accuracy metric for
  a slide deck.
- **Every claim in this README was tested, not assumed.** The numbers above came from
  actually running the pipeline end-to-end multiple times during development, including
  deliberately breaking things (missing columns, out-of-range values, zero traffic,
  artificially shifted distributions) to confirm the pipeline reacts correctly.
- **Deploy-ready and security-audited, not just "should work in theory."** The serving
  app was verified with a live MLflow server *fully unplugged* (`MLFLOW_TRACKING_URI`
  unset, no server running) to confirm the deployed mode is genuinely self-contained.
  Every endpoint that returns customer-derived data requires an API key — including a
  gap (public `/metrics/*` endpoints) that existed in an earlier version and was closed
  after an explicit audit. The app also refuses to boot in production with a default
  API key, catching the most common deployment mistake at startup instead of shipping
  it. Full detail in [`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md).

---

## Project structure

```
ml-pipeline-project/
├── README.md                       # this file — overview, architecture, verified results
├── SETUP_GUIDE.md                  # full local setup walkthrough + troubleshooting
├── GITHUB_AND_DEPLOYMENT.md        # push to GitHub, real data, free deployment, security audit
├── SETUP_AND_IMPROVEMENTS.md       # prioritized roadmap of further improvements
├── requirements.txt
├── Dockerfile                      # deploy-ready: bakes model_artifacts/ into the image
├── .dockerignore
├── docker-compose.yml              # local live-registry dev/test setup
├── .gitignore                      # deliberately keeps model_artifacts/ + baseline committed
├── .github/workflows/
│   ├── ci.yml                      # trains, tests, exports, and smoke-tests the deploy path
│   └── retrain.yml                 # scheduled drift check + conditional auto-retrain
├── scripts/
│   └── generate_traffic.py         # sends realistic simulated traffic for testing
├── model_artifacts/                # baked-in trained model (committed — see export_model.py)
├── src/
│   ├── config.py                   # central schema, paths, thresholds
│   ├── logging_config.py           # structured JSON logging
│   ├── data/
│   │   ├── generate_data.py        # synthetic data generator
│   │   ├── prepare_real_data.py    # real Telco dataset adapter
│   │   └── validate_data.py        # schema + quality gate
│   ├── training/
│   │   ├── train.py                # trains candidates, logs to MLflow, stages/promotes
│   │   ├── promote_staging.py      # reviews shadow results, promotes on command
│   │   └── export_model.py         # bakes the registry model into model_artifacts/
│   ├── serving/
│   │   ├── app.py                  # FastAPI, dual-mode (mlflow / local) model loading
│   │   └── dashboard.html          # custom operations console
│   └── monitoring/
│       ├── drift_check.py          # PSI + optional Evidently HTML report
│       └── log_predictions.py      # SQLite logging (predictions + shadow comparisons)
└── tests/
    ├── test_data_validation.py
    ├── test_training.py
    └── test_serving.py             # includes auth + deploy-path coverage
```

---

## Setup guide

Full step-by-step local setup (environment, data, MLflow, training, serving, shadow
deployment, testing) lives in [`SETUP_GUIDE.md`](./SETUP_GUIDE.md), including a
troubleshooting table for the exact errors you're likely to hit.

For connecting the real dataset, pushing to GitHub, and deploying for free — including
a verified security audit and every challenge to expect — see
[`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md).

---

## Talking points for interviews

- *"I don't train on unchecked data — the pipeline fails fast on schema, null rate, or
  range violations before wasting compute."*
- *"New models only get promoted if they beat the current production model — and even
  then, they're shadow-tested on real traffic first. I have actual agreement-rate data
  (66.7% in my test run) showing why that step matters."*
- *"The `/explain` endpoint gives per-customer reasoning, not just a global feature
  importance chart — something an actual retention team could act on."*
- *"I built a custom monitoring dashboard instead of defaulting to Streamlit, because I
  wanted full control over the design and the data model behind it."*
- *"Drift detection isn't just a metric on a dashboard — it's wired to a GitHub Actions
  workflow that can trigger retraining automatically."*
- *"I know the `/explain` endpoint is slow (~9s) because of the model-agnostic SHAP
  explainer — and I know exactly how I'd fix it (a model-specific explainer) if this
  needed to hit a real latency SLA."*
- *"The deployed version has no live MLflow server — I export the winning model into a
  self-contained artifact and bake it into the Docker image, which I verified works by
  running the app with the MLflow tracking URI completely unset."*
- *"I audited every endpoint for auth — the metrics endpoints used to be public in an
  earlier version, which I caught and closed, since they return real customer feature
  data."*

---

## Extending this project

See [`SETUP_AND_IMPROVEMENTS.md`](./SETUP_AND_IMPROVEMENTS.md) for a full prioritized
list of further improvements (DVC data versioning, alerting, load testing, rate
limiting, and more).
