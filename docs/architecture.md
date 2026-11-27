# System Architecture

## Runtime

The application is a Streamlit monolith with reusable Python modules:

- `app.py`: dashboard orchestration and session-state UI.
- `src/attrition.py`: validation, cleaning, splitting, training, optimization, metrics, and artifact persistence.
- `src/explainability.py`: SHAP and permutation-based explanation calculations.
- `src/workforce_analytics.py`: filtering and aggregate workforce analytics.
- `src/model_monitoring.py`: drift, quality, prediction, performance, export, and persistence.
- `src/deployment.py`: environment/configuration validation.
- `src/ui_theme.py`: presentation components, CSS, status cards, and chart styling.
- `train.py`: baseline model training entry point.

## Data Flow

1. Load or upload a validated CSV.
2. Validate the required schema and binary `quit` target.
3. Remove exact duplicates.
4. Create a deterministic 80/20 stratified split.
5. Fit preprocessing inside pipelines and cross-validation folds.
6. Train/evaluate baseline or optimized models.
7. Render aggregate metrics and SHAP explanations.
8. For monitoring, compare a batch with aggregate training-reference summaries.
9. Persist aggregate reports under `artifacts/monitoring/`.

## Tech Stack

- Python and pandas/NumPy
- scikit-learn
- SHAP
- SciPy
- Streamlit
- Plotly and Matplotlib
- Joblib
- pytest and Ruff

## Folder Structure

```text
app.py
train.py
src/
tests/
data/
artifacts/
reports/
docs/
.streamlit/
Dockerfile
render.yaml
Procfile
```

## Deployment Topology

The supported local/container topology is a single Streamlit process using `/ _stcore/health` for health checks. Docker, Render, and Procfile examples are provided. Monitoring persistence is local-filesystem based; hosted deployments requiring durability need persistent storage or object-backed storage.

## External Integrations

No credentials or external APIs are required. Hosting platforms may provide HTTPS, access controls, process supervision, and persistent volumes.

## Deployment Configuration

`APP_ENV`, `LOG_LEVEL`, `DATA_PATH`, `ARTIFACT_DIR`, `MONITORING_DIR`, and `ALLOW_ARTIFACT_WRITES` are read by `src/deployment.py`.

## Deployment Commands

Local:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.headless true --server.port 8501
curl.exe -I http://127.0.0.1:8501/_stcore/health
```

Docker:

```powershell
docker build -t workforce-insights:m2.7 .
docker run --rm -p 8501:8501 workforce-insights:m2.7
```

Render and Procfile examples are included at the repository root. Docker availability and a public provider deployment must be verified independently of local Streamlit startup.
