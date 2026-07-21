# GitHub & Deployment Guide

Covers pushing this project to GitHub, connecting the real Telco dataset, and
deploying for free — using the deploy-ready architecture where the trained model is
baked into the Docker image, so no live MLflow server is needed at runtime. Every claim
below was tested (see the verification notes) rather than assumed.

---

## Part 1: Push to GitHub

```bash
cd ml-pipeline-project
git init
git add .
git commit -m "Initial commit: end-to-end ML pipeline with shadow deployment, monitoring, and deploy-ready serving"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

**Before your first commit**, double check `.gitignore` — it's been set up deliberately
so that files needed for *deployment* are NOT ignored, even though most generated data
usually would be:

| File / folder | Committed? | Why |
|---|---|---|
| `data/churn_raw.csv`, `data/churn_test.csv` | ❌ ignored | Not needed at runtime, regenerate locally anytime |
| `data/churn_train.csv` | ✅ **committed** | `/explain`'s SHAP background sample reads this at runtime |
| `data/baseline_stats.json` | ✅ **committed** | Drift detection's baseline — needed at runtime |
| `model_artifacts/` | ✅ **committed** | The actual trained model the deployed container serves |
| `mlruns/`, `mlflow.db` | ❌ ignored | Local-only tracking state, not needed for deployment |
| `venv/`, `__pycache__/` | ❌ ignored | Standard |

This matters because **Render and Hugging Face Spaces build your Docker image from
your GitHub repo**, not your local disk — if `model_artifacts/` isn't committed, the
remote build will fail with a missing-file error on the `COPY model_artifacts/` line in
the Dockerfile.

---

## Part 2: Connecting the real dataset

### Why this matters for your resume

Interviewers will ask "is this real data?" Having a real answer — with real messiness
handled explicitly — is more credible than a clean synthetic set.

### Steps

1. Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   (file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`, free Kaggle account required)
2. Place it at `data/raw_telco.csv`
3. Run:
   ```bash
   python src/data/prepare_real_data.py
   ```

**Verified**: I tested this adapter against a fixture replicating the real dataset's
exact structure, including its documented quirk — 11 rows with a blank string in
`TotalCharges` for customers with zero tenure. The script detects this, prints how many
rows were affected, and imputes `0.0` with an explicit justification rather than
silently producing broken data.

### Why the schema changed

The real dataset has no "support tickets" field, so the pipeline's schema was updated
to use two fields the real dataset actually has instead: `internet_service` (DSL /
Fiber optic / No) and `senior_citizen` (boolean) — both genuine, informative churn
predictors. This is itself a good interview talking point: *"I adjusted the feature set
to match what was actually available and predictive in real data, rather than forcing a
synthetic feature onto it."*

### After connecting real data

```bash
python src/data/validate_data.py
python src/training/train.py
python src/training/export_model.py   # re-export before redeploying
```

Everything downstream works unchanged. Update your README's test-results section with
the real numbers — more impressive to cite than synthetic ones.

---

## Part 3: Deploy-ready architecture — how the model gets baked in

### The problem this solves

The original architecture required a **live MLflow server** running alongside the
FastAPI app. Free hosting tiers typically give you one web service, not two — so this
had to change before deploying anywhere.

### The fix, verified

`src/training/export_model.py` pulls the current Production (and Staging, if any)
model out of the MLflow registry and saves it as a self-contained directory
(`model_artifacts/`) that needs no live server to load — just the `mlflow` Python
*library* (for its model-loading utility), not a running `mlflow server` process.

**I verified this actually works** by running the full app with `MLFLOW_TRACKING_URI`
completely unset and no MLflow server running anywhere, using only
`MODEL_SOURCE=local`:

```
HEALTH: {'status': 'ok', 'model_source': 'local', 'environment': 'production',
         'production_model_loaded': True, 'production_version': '1', ...}
PREDICT: 200 {'churn_prediction': 1, 'churn_probability': 0.8927, 'model_version': '1'}
EXPLAIN: 200 ok
METRICS: 200 {...}
```

All four endpoints worked with zero MLflow server present — confirming the deployed
container really is self-contained.

### How the two modes work

| | Local development | Deployment |
|---|---|---|
| Env var | `MODEL_SOURCE=mlflow` (default) | `MODEL_SOURCE=local` (Dockerfile default) |
| Model source | Live MLflow registry over HTTP | `model_artifacts/` baked into the image |
| Needs a running `mlflow server`? | Yes | No |
| Gives you shadow deployment / `promote_staging.py` / hot `/reload-model`? | Yes, fully | `/reload-model` only re-reads local disk (changes only on redeploy) |

You don't need to choose one forever — develop and iterate in `mlflow` mode locally,
then run `export_model.py` and switch to `local` mode only for the deployed container.

### Before every deploy

```bash
python src/training/export_model.py
git add model_artifacts data/baseline_stats.json data/churn_train.csv
git commit -m "Export trained model for deployment"
git push
```

---

## Part 4: Security cross-check (verified, not just claimed)

I audited every endpoint and closed one real gap that existed in an earlier version of
this project. Here's the current state:

| Endpoint | Auth required? | Why |
|---|---|---|
| `POST /predict`, `POST /explain`, `POST /reload-model` | Yes | Scoring endpoints |
| `GET /metrics/summary`, `/metrics/predictions`, `/metrics/shadow`, `/metrics/drift` | Yes (this changed) | These return real customer feature data — an earlier version left them open, which was a real gap for a public deployment. Now behind the same `X-API-Key` check. |
| `GET /dashboard` | No | Serves only the static HTML shell, no data — the page's own JavaScript prompts for an API key before it can call any `/metrics/*` endpoint |
| `GET /health` | No | No sensitive data, standard practice to leave health checks open |

**Verified with actual requests, not just code review**:
```
METRICS no auth: 401
METRICS with auth: 200 {...}
```

### Fail-fast production safety check

The app now **refuses to start** if `ENVIRONMENT=production` and `API_KEY` is left at
its default dev value — this catches the single most common deployment mistake
(forgetting to set a real key) at boot time instead of silently shipping an open API.

**Verified**:
```
CORRECTLY REFUSED TO START: Refusing to start: ENVIRONMENT=production but API_KEY
is still the default dev value. Set a real API_KEY as an environment variable...
```

### What's still a known limitation (stated honestly, not hidden)

- **No rate limiting** — someone with a valid key could still hammer the endpoint. A
  documented next step (`slowapi` is a natural fit), not implemented here.
- **Single shared API key**, not per-client keys or OAuth — fine for a portfolio demo,
  not fine for multi-tenant production use. Say this explicitly if asked in an
  interview; it shows you know the difference.
- **Dashboard auth via browser prompt + sessionStorage** is a lightweight pattern
  appropriate for a demo, not a real auth system (no expiry, no revocation). Good
  enough here; would need a real session/JWT flow for anything beyond a portfolio piece.
- **Ephemeral prediction logs** — see Part 6, Challenge 2.

---

## Part 5: Free deployment options (checked mid-2026)

| Platform | Free tier status | Notes |
|---|---|---|
| Render | Genuinely free, no credit card | 512 MB RAM / 0.1 CPU; spins down after ~15 min idle (cold start on next request) |
| Hugging Face Spaces (Docker SDK) | Genuinely free | Works well for FastAPI + Docker; no forced sleep like Render |
| Railway | Not really free anymore | Only ~$1/month credit post-trial |
| Fly.io | No free tier for new users | Requires a credit card now |

**Recommendation**: Render or Hugging Face Spaces.

### Deploying to Render

1. Push to GitHub (Part 1), including `model_artifacts/` (Part 3).
2. https://render.com → sign up (no card) → **New +** → **Web Service** → connect your repo.
3. Environment: **Docker**. Dockerfile path: `Dockerfile` (already in your repo).
4. Add environment variables:
   - `API_KEY` = a real value you generate (not `dev-key-change-me`)
   - `ENVIRONMENT` = `production` (this is also the Dockerfile's default, but setting
     it explicitly here is good practice)
5. Create the service. Your API will be live at `https://your-app-name.onrender.com`.

### Deploying to Hugging Face Spaces

1. https://huggingface.co/new-space → SDK: **Docker**.
2. Clone the Space repo, copy your project files (including `Dockerfile` and
   `model_artifacts/`) into it, commit, push.
3. Settings → **Repository secrets** → add `API_KEY`.

---

## Part 6: Challenges to expect (and the fix for each)

### 1. MLflow needing its own running server — solved

This was the original blocker; Part 3 above is the fix. Not a concern anymore with
`MODEL_SOURCE=local`.

### 2. The filesystem is still ephemeral for prediction logs

Free container platforms wipe local disk on redeploy/restart. The **model itself** is
safe (baked into the image), but `logs/predictions.db` (your prediction history) resets
on every restart. For a demo, this is fine — just know traffic you send will disappear
if the container restarts. For a persistent version, you'd swap SQLite for a managed
Postgres (Render, Neon, and Supabase all have small free tiers) — a good "next step" to
mention if asked.

### 3. Cold starts (Render specifically)

Free tier sleeps after ~15 min of no traffic; first request after that takes 30–60s to
wake up. Expected behavior — mention it if demoing live.

### 4. Memory limits

512 MB (Render free tier) is tight with scikit-learn + pandas + SHAP + MLflow all
loaded. The app already lazy-loads the SHAP explainer (only built on first `/explain`
call, not at startup) to help with this. If you hit OOM errors, consider trimming
`requirements.txt` for the deployed image (e.g., `pytest` doesn't need to ship in
production).

### 5. Secrets management

Never commit a real `API_KEY` to git or hardcode it in the Dockerfile — use the
platform's secret store. The Dockerfile deliberately does NOT set a default `API_KEY`,
and the app refuses to boot without a real one in production (Part 4).

### 6. HTTPS / CORS

Both Render and HF Spaces give you HTTPS automatically. CORS middleware is only needed
if you add a separate frontend calling this API from a browser on a different origin —
not needed for API-only demos via curl/Postman/`/docs`.

---

## Suggested order of operations

1. Connect real data locally (Part 2), confirm training/serving still work.
2. Update your README's test-results section with real numbers.
3. Export the model (Part 3), commit `model_artifacts/`.
4. Push to GitHub (Part 1) — CI will run the full suite including the deploy-mode smoke
   test automatically.
5. Deploy to Render or Hugging Face Spaces (Part 5).
6. Add the live URL to your resume/GitHub README — the single highest-impact addition
   you can make. A working link an interviewer can click beats any amount of description.
