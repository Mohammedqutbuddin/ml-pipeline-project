# Setup Guide & Improvement Roadmap — Churn Prediction MLOps Pipeline

This is a from-zero walkthrough (assumes nothing is running yet) plus a prioritized list
of improvements you can add to make the project even stronger for interviews.

---

## Part 1: End-to-End Setup Guide

### Prerequisites

- Python 3.10–3.11 installed (`python3 --version`)
- Docker Desktop installed if you want to run the containerized version (optional but
  recommended — shows deployment skill)
- Git + a GitHub account (for the CI/CD portion)
- ~2GB free disk space

### Step 1 — Unzip and enter the project

```bash
unzip ml-pipeline-project.zip
cd ml-pipeline-project
```

### Step 2 — Create an isolated environment

```bash
python3 -m venv venv

# Activate it:
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows (cmd)
venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

You'll know it worked when your terminal prompt shows `(venv)` at the start.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes 2–5 minutes depending on your connection. If `evidently` fails to install
(it has heavier dependencies), you can skip it — the pipeline degrades gracefully and
still computes PSI-based drift without it:

```bash
pip install -r requirements.txt --no-deps evidently  # or just remove it from requirements.txt
```

### Step 4 — Generate and validate data

```bash
python src/data/generate_data.py
python src/data/validate_data.py
```

Expected output: a JSON report showing all checks as `PASS`, ending with
`All data quality checks passed.` This creates `data/churn_raw.csv`,
`data/churn_train.csv`, and `data/churn_test.csv`.

**If validation fails**: read the JSON report — it tells you exactly which check failed
and why (e.g., which column has too many nulls, which values are out of range).

### Step 5 — Start the MLflow tracking server

Open a **new terminal window** (keep it running throughout), activate the venv again, then:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Visit **http://127.0.0.1:5000** — you should see the empty MLflow UI. Leave this
terminal running.

> Note: newer MLflow versions (3.x) print deprecation warnings about "stages" —
> these are harmless; registry promotion still works. If you hit an error mentioning
> "filesystem tracking backend is in maintenance mode," make sure you used the
> `sqlite:///mlflow.db` backend as shown above, not a bare folder path.

### Step 6 — Train and register the model

Back in your **original terminal** (venv activated, project root):

```bash
python src/training/train.py
```

Watch it: trains 3 models, logs each to MLflow, picks the best by F1, and registers +
promotes it to "Production". Go back to the MLflow UI and refresh — you'll see 3 runs
under the `churn-prediction` experiment, and under **Models** you'll see
`churn-classifier` with a version in the Production stage.

### Step 7 — Serve the model

```bash
uvicorn src.serving.app:app --reload --port 8000
```

Visit **http://127.0.0.1:8000/docs** for interactive Swagger docs, or test via curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 5, "monthly_charges": 85.5, "total_charges": 427.5, "contract_type": "month-to-month", "internet_service": "Fiber optic", "has_tech_support": false, "senior_citizen": false}'
```

Every request gets logged to `logs/predictions.db`.

### Step 8 — Generate some traffic, then check for drift

Send a handful of varied requests (or write a quick loop hitting `/predict` with random
values), then:

```bash
python src/monitoring/drift_check.py
```

With fewer than 30 logged predictions you'll see `insufficient_data` — that's correct
behavior, not a bug. Send more requests and re-run.

### Step 9 — Run the test suite

```bash
pytest tests/ -v
```

All 7 tests should pass. This is what the CI workflow runs automatically on every push.

### Step 10 — Try the Dockerized version (optional but impressive in interviews)

```bash
docker-compose up --build
```

This runs MLflow and the FastAPI server together as containers, closer to how it'd
actually be deployed.

### Step 11 — Push to GitHub and watch CI run

```bash
git init
git add .
git commit -m "Initial commit: end-to-end ML pipeline"
git remote add origin <your-repo-url>
git push -u origin main
```

Go to the **Actions** tab on GitHub — `ci.yml` should run automatically. The
`retrain.yml` workflow runs on a schedule but you can also trigger it manually from the
Actions tab via "Run workflow" (it uses `workflow_dispatch`).

### Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Make sure you're running commands from the project root, not from inside `src/` |
| MLflow "No Production model found" when starting the API | Run `python src/training/train.py` before starting the server |
| `pip install` fails on `evidently` | Drop it from requirements.txt — the pipeline works without it (see Step 3) |
| Port already in use | Change `--port` in the relevant command, or kill the process using that port |
| Docker build is slow | First build only — subsequent builds are cached |

---

## Part 2: Improvements to Add (Prioritized)

These are ranked by **impact on interview conversations vs. effort to implement**, so
you can decide where to invest your time.

### High impact, moderate effort

**1. Swap in a real dataset**
Replace `generate_data.py`'s synthetic output with the Kaggle "Telco Customer Churn"
dataset (or any real churn/tabular dataset). Update `config.py`'s schema to match. This
matters because interviewers will ask "is this real data?" — having a real answer with
real messiness (actual missing values, actual class imbalance) is more credible than a
clean synthetic set.

**2. Add a monitoring dashboard**
Right now drift checks print JSON. Add a small Streamlit or Gradio dashboard that reads
from `logs/predictions.db` and shows: prediction volume over time, prediction
distribution, feature drift trends, model version history. This is the single most
"demo-able" addition — screenshots of a dashboard are what recruiters actually look at.

**3. Alerting on drift**
When `drift_check.py` detects drift, send a Slack webhook or email instead of just
exiting with code 1. Ties directly into the GitHub Actions retrain workflow — add a
`curl` call to a Slack incoming webhook in `retrain.yml` when drift is detected. Shows
you think about the human-in-the-loop side of MLOps, not just automation.

**4. Data versioning with DVC**
Add DVC (Data Version Control) to track dataset versions alongside your Git commits, so
every trained model can be traced back to the exact data version that produced it. This
is a very commonly asked-about tool in ML Engineer interviews.

**5. Model explainability**
Add a SHAP-based explanation endpoint (`/explain`) to the FastAPI app that returns
feature importance for a given prediction. Directly relevant if you're ever asked "how
would you explain a model's decision to a non-technical stakeholder?"

### Medium impact, lower effort

**6. Config via environment variables / `.env` file**
Currently thresholds like `DRIFT_PSI_THRESHOLD` are hardcoded in `config.py`. Move them
to environment variables (with `python-dotenv`) so the same code can run differently in
dev/staging/prod without code changes — a basic but expected production practice.

**7. ~~API authentication~~ — done**
`/predict`, `/explain`, `/reload-model`, and all `/metrics/*` endpoints now require an
`X-API-Key` header, and the app refuses to start in production with a default key. See
[`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md) for the full audit.

**8. Rate limiting**
Still not implemented. Add `slowapi` (a FastAPI-compatible rate limiter) to `/predict`.
A valid API key currently has no limit on request volume — a real gap for a public
deployment, documented honestly rather than hidden.

**9. ~~Structured logging~~ — done**
`src/logging_config.py` — JSON-formatted logs via the stdlib `logging` module.

**10. ~~Expand test coverage~~ — done**
14 tests total now, including auth enforcement, metrics-endpoint auth, and a CI
smoke test of the deploy path itself (`MODEL_SOURCE=local` with no MLflow server).

### High impact, higher effort

**11. ~~Shadow deployment / canary release logic~~ — done**
Implemented and verified with real traffic (66.7% agreement rate measured in testing).
See the README's "verified test results" section.

**12. Feature store pattern**
Still open. A lightweight module that computes and caches features consistently for
both training and serving, preventing train/serve skew — one of the most common
real-world ML bugs.

**13. ~~Cloud deployment~~ — done (without Terraform)**
The model is now baked into a deploy-ready Docker image (`MODEL_SOURCE=local`, verified
to work with zero live MLflow server) and documented for Render / Hugging Face Spaces
in [`GITHUB_AND_DEPLOYMENT.md`](./GITHUB_AND_DEPLOYMENT.md). Terraform/infra-as-code for
a full cloud provider (AWS/GCP/Azure) instead of a PaaS remains a good next step if you
want to go further — it's a different skill (infra-as-code) than what's demonstrated
here (containerization + secrets + fail-fast config).

**14. Load testing**
Still open. Use Locust or k6 to load-test `/predict` and document results (requests/sec,
p50/p95/p99 latency) — real numbers beat vague "scalability" claims.

### Suggested next move

With shadow deployment, the dashboard, real data, and deployment now done, the highest-
leverage remaining items are **#8 (rate limiting)** — a quick, concrete security
addition — and **#14 (load testing)**, which would give you real throughput numbers to
quote alongside the latency numbers already in the README.

Want me to implement any of these directly into the project (e.g., the Streamlit
dashboard or the real-dataset swap)?
