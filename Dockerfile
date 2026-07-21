# Deploy-ready image: the trained model is baked in at build time
# (model_artifacts/, produced by `python src/training/export_model.py`), so
# this container has NO runtime dependency on a live MLflow server. That's
# what makes it deployable to a free-tier platform that only gives you one
# service (Render, Hugging Face Spaces, etc).
#
# Build (run the pipeline locally first — see SETUP_GUIDE.md):
#   docker build -t churn-api .
#
# Run:
#   docker run -p 8000:8000 -e API_KEY=<a-real-key> -e ENVIRONMENT=production churn-api
#
# For local development against a LIVE MLflow registry instead (shadow
# deployment workflow, promote_staging.py, etc.), use docker-compose.yml,
# which overrides MODEL_SOURCE back to "mlflow" for that use case.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<82" && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# The trained model, baked in at build time. If these are missing, run
# `python src/training/export_model.py` locally and commit the output —
# see GITHUB_AND_DEPLOYMENT.md.
COPY model_artifacts/ model_artifacts/

# Only the drift baseline is needed at runtime, not the raw/train/test CSVs
# (no training happens inside the deployed container).
COPY data/baseline_stats.json data/baseline_stats.json
COPY data/churn_train.csv data/churn_train.csv

ENV MODEL_SOURCE=local
ENV ENVIRONMENT=production
# API_KEY is intentionally NOT set here — it must come from a real secret
# (platform environment variable), never baked into the image or committed
# to git. The app refuses to start in production without one being set.

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
