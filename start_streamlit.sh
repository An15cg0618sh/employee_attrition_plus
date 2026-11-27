#!/usr/bin/env sh
set -eu

export APP_ENV="${APP_ENV:-development}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export DATA_PATH="${DATA_PATH:-data/employee_data.csv}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"
export MONITORING_DIR="${MONITORING_DIR:-artifacts/monitoring}"
export ALLOW_ARTIFACT_WRITES="${ALLOW_ARTIFACT_WRITES:-true}"

exec python -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
