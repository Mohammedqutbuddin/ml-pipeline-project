# Setup Guide — Churn Prediction MLOps Pipeline (Current Version)

This is the up-to-date, from-zero walkthrough for the project as it stands today,
including the real-dataset adapter, shadow deployment, the custom monitoring dashboard,
API auth, and the traffic generator. If you followed an earlier version of this guide,
note the schema changed: `num_support_tickets` was replaced with `internet_service` and
`senior_citizen` (see "What's changed" at the bottom).

---

## Prerequisites

- Python 3.10–3.11 (`python --version`)
- Git + a GitHub account
- Docker Desktop — optional, only needed for the containerized run
- ~2 GB free disk space
- A free Kaggle account — optional, only needed if you're connecting real data

---

## Step 1 — Get the project and create an environment

```bash
cd ml-pipeline-project

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows cmd
venv\Scripts\Activate.ps1       # Windows PowerShell
```

You'll know it worked when your prompt shows `(venv)` at the start.

## Step 2 — Install dependencies

```bash
pip install --upgrade pip
pip install "setuptools<82"
pip install -r requirements.txt
```

**Why the setuptools pin**: as of February 2026, `setuptools` v82 removed the
`pkg_resources` module that MLflow depends on internally. Without this pin you'll hit
`ModuleNotFoundError: No module named 'pkg_resources'` the moment you try to start
MLflow — this is a known, widely-reported issue, not something specific to your setup.

If `evidently` fails to install (it has heavier dependencies), skip it — the pipeline
degrades gracefully and still computes drift via PSI without it.

## Step 3 — Get data: synthetic (fast) or real (more impressive)

**Option A — synthetic data, ready in seconds:**

```bash
python src/data/generate_data.py
```

**Option B — the real Kaggle/IBM Telco Customer Churn dataset:**

1. Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   (file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`)
2. Place it at `data/raw_telco.csv`
3. Run:
   ```bash
   python src/data/prepare_real_data.py
   ```
   This handles real messiness explicitly — e.g. 11 rows with a blank `TotalCharges`
   for brand-new (zero-tenure) customers — and prints what it fixed. See
   [`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md) for full detail.

Either path produces the same output files, so everything below works unchanged:
`data/churn_raw.csv`, `data/churn_train.csv`, `data/churn_test.csv`.

## Step 4 — Validate the data

```bash
python src/data/validate_data.py
```

Expect `All data quality checks passed.` at the end. If it fails, the printed JSON
report tells you exactly which check failed and why.

## Step 5 — Start MLflow (own terminal — leave it running)

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

**Windows note**: put the whole command on one line. `cmd.exe` doesn't understand the
Unix `\` line-continuation character — if you split the command across lines with `\`,
each fragment gets run as its own broken command. Use `^` (cmd) or `` ` `` (PowerShell)
if you want to split it, or just keep it on one line.

Visit http://127.0.0.1:5000 to confirm the UI loads. Leave this terminal running for
the rest of the session.

## Step 6 — Train the model (back in your first terminal)

```bash
python src/training/train.py
```

Trains 3 candidate models (logistic regression, random forest, gradient boosting),
logs every run to MLflow, and promotes the best one to Production — or stages it if a
production model already exists (see Step 9, shadow deployment).

## Step 7 — Serve the model (own terminal — leave it running)

```bash
export API_KEY=dev-key-change-me      # macOS/Linux
set API_KEY=dev-key-change-me         # Windows cmd

uvicorn src.serving.app:app --reload --port 8000
```

By default this runs in **live-registry mode** (`MODEL_SOURCE=mlflow`), talking to the
MLflow server you started in Step 5 — this is what you want for local development,
since it gives you the full shadow-deployment workflow. There's a second mode,
`MODEL_SOURCE=local`, used only for deployment (no live MLflow server needed) — see
[`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md) for that.

- Interactive API docs: http://127.0.0.1:8000/docs
- **Live operations dashboard**: http://127.0.0.1:8000/dashboard (will prompt you for
  your API key on load — the metrics endpoints require it, same as `/predict`)

Test a prediction (note the current schema — `internet_service` and `senior_citizen`,
not `num_support_tickets`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-me" \
  -d '{"tenure_months": 5, "monthly_charges": 85.5, "total_charges": 427.5, "contract_type": "month-to-month", "internet_service": "Fiber optic", "has_tech_support": false, "senior_citizen": false}'
```

**Windows cmd note**: quotes inside `-d` need backslash-escaping:
```cmd
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -H "X-API-Key: dev-key-change-me" -d "{\"tenure_months\": 5, \"monthly_charges\": 85.5, \"total_charges\": 427.5, \"contract_type\": \"month-to-month\", \"internet_service\": \"Fiber optic\", \"has_tech_support\": false, \"senior_citizen\": false}"
```

## Step 8 — Generate traffic and check the dashboard/drift

Typing curl 30+ times to get past the drift detector's minimum sample size is painful —
use the traffic generator instead (separate terminal, same venv activated):

```bash
pip install requests    # if not already installed
python scripts/generate_traffic.py --n 40
```

Then refresh http://127.0.0.1:8000/dashboard — you'll see real numbers: prediction
count, churn rate, latency, per-feature drift, and a recent-predictions table.

Or check drift from the command line:

```bash
python src/monitoring/drift_check.py
```

## Step 9 — Try shadow deployment

The training script automatically **stages** (doesn't immediately promote) any
challenger model that beats the current production model — this lets you validate it
against real traffic before trusting it. To see this in action, you'd re-run training
after production already has a model (Step 6 again, or manually register a second
candidate). Once a Staging model exists:

```bash
python src/training/promote_staging.py
```

This prints the challenger's agreement rate with Production, measured over real traffic
the serving app has silently scored. Promote once confident:

```bash
python src/training/promote_staging.py --version 2
curl -X POST http://127.0.0.1:8000/reload-model -H "X-API-Key: dev-key-change-me"
```

## Step 10 — Get a per-prediction explanation

```bash
curl -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-me" \
  -d '{"tenure_months": 5, "monthly_charges": 85.5, "total_charges": 427.5, "contract_type": "month-to-month", "internet_service": "Fiber optic", "has_tech_support": false, "senior_citizen": false}'
```

Returns SHAP contributions per feature. Note: this takes ~9 seconds (model-agnostic
explainer) — that's a known, documented performance characteristic, not a bug.

## Step 11 — Run the test suite

```bash
pytest tests/test_data_validation.py tests/test_training.py -v

# to also run serving/auth/shadow tests, a trained model must exist:
export RUN_SERVING_TESTS=1
export API_KEY=dev-key-change-me
pytest tests/ -v
```

All 12 tests should pass.

## Step 12 — Docker (optional)

```bash
docker-compose up --build
```

Runs MLflow and the FastAPI server together as containers.

## Step 13 — Push to GitHub / CI

```bash
git add .
git commit -m "..."
git push
```

The Actions tab on GitHub will run `ci.yml` automatically (trains a model and runs the
full test suite) and `retrain.yml` on a schedule (or trigger it manually via
"Run workflow").

## Step 14 — Deploy for free (optional, high resume value)

See [`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md) for the full guide —
covers Render and Hugging Face Spaces (both genuinely free), plus honest challenges to
expect (ephemeral filesystems, cold starts, memory limits) and concrete fixes for each.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install "setuptools<82"` |
| `\` line continuation fails on Windows | Put the command on one line, or use `^` (cmd) / `` ` `` (PowerShell) instead of `\` |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root, not from inside `src/` |
| "No Production model found" when starting the API | Run `python src/training/train.py` first |
| `generate_traffic.py`: "Could not connect" | The `uvicorn` server isn't running (or crashed) in its own terminal — check that terminal, restart it, and run the traffic script from a *different* terminal |
| `pip install` fails on `evidently` | Drop it from `requirements.txt` — the pipeline works without it |
| `prepare_real_data.py`: `FileNotFoundError` | Make sure the Kaggle CSV is at exactly `data/raw_telco.csv` |
| `drift_check.py` shows `insufficient_data` | Expected below 30 logged predictions — send more traffic first |
| Port already in use | Change `--port` in the relevant command, or stop whatever's already using it |
| `curl` quoting errors on Windows | Escape inner quotes with `\"` as shown in Step 7, or use the `/docs` Swagger UI instead of curl |

---

## What's changed from earlier versions of this project

- **Schema**: `num_support_tickets` (synthetic-only) replaced with `internet_service`
  and `senior_citizen` (both real, available fields in the actual Telco dataset)
- **New**: `src/data/prepare_real_data.py` — real dataset adapter
- **New**: `scripts/generate_traffic.py` — simulated traffic generator for testing
  drift and the dashboard without manual curl loops
- **New**: shadow deployment (`Staging` stage in MLflow, `promote_staging.py`, dual
  scoring in the serving app)
- **New**: `/dashboard` — custom-built operations console
- **New**: `/explain` — SHAP-based per-prediction explanations
- **New**: API key auth on scoring endpoints, extended to `/metrics/*` as well
- **New**: structured JSON logging (`src/logging_config.py`)
- **New**: `src/training/export_model.py` and a dual-mode serving app
  (`MODEL_SOURCE=mlflow` for local dev, `MODEL_SOURCE=local` for deployment — no live
  MLflow server needed in the deployed container). See
  [`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md).
- **New**: the app refuses to start if `ENVIRONMENT=production` and `API_KEY` is left
  at its default value — catches the most common deployment mistake at boot time.
