# Milestone 2.7 Deployment Handoff

## Supported startup

The application is a Python/Streamlit process. It is not a static Vercel or serverless Streamlit deployment.

Local Windows verification:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:APP_ENV="production"
$env:LOG_LEVEL="INFO"
$env:DATA_PATH="data/employee_data.csv"
$env:ARTIFACT_DIR="artifacts"
$env:MONITORING_DIR="artifacts/monitoring"
$env:ALLOW_ARTIFACT_WRITES="true"

.\start_streamlit.ps1
```

Health check:

```powershell
curl.exe -I http://127.0.0.1:8501/_stcore/health
```

## Environment

Copy `.env.example` to `.env` locally. Do not commit `.env` or `.streamlit/secrets.toml`.

| Variable | Default | Use |
|---|---|---|
| `APP_ENV` | `development` | development, staging, or production |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `DATA_PATH` | `data/employee_data.csv` | default CSV |
| `ARTIFACT_DIR` | `artifacts` | optimized/baseline artifact directory |
| `MONITORING_DIR` | `artifacts/monitoring` | aggregate monitoring output |
| `ALLOW_ARTIFACT_WRITES` | `true` | set false for read-only deployments |

## Docker

```powershell
docker build -t workforce-insights:m2.7 .
docker run --rm -p 8501:8501 workforce-insights:m2.7
curl.exe -I http://127.0.0.1:8501/_stcore/health
```

The Docker image excludes untracked model artifacts and starts the app from the repository dataset. Mounted storage is required for durable monitoring output.

## GitHub and external hosts

- GitHub is the source/release layer.
- Streamlit Community Cloud can run `app.py` and install `requirements.txt`.
- Render or another container host can use `Dockerfile`, `render.yaml`, or `Procfile`.
- Vercel is not a direct Streamlit runtime; use a Streamlit-capable host or redesign the app as separate APIs/frontends.
- Configure HTTPS, access controls, timeouts, and persistent storage at the provider.
- Do not live-deploy in this repository task.

## Security and privacy

- No credentials are required by the application.
- Never expose uploaded employee records or individual predictions.
- Monitoring exports contain aggregate values only.
- Do not use model outputs for employment decisions.
- Secrets must be supplied through the host secret manager.

## Troubleshooting

- Invalid `APP_ENV` or log level: correct the environment variables.
- Missing data: verify `DATA_PATH`.
- Write denied: set `ALLOW_ARTIFACT_WRITES=true` or mount a writable volume.
- Corrupt monitoring history: inspect or remove only `artifacts/monitoring/monitoring_history.json`, preserving model artifacts.
