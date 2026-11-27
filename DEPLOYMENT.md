# Milestone 2.7 - Deployment Guide

## Supported targets

The reference production target is a Linux container running Streamlit. The repository also provides a PowerShell local startup script and a `Procfile`/Render example.

This deployment runs the application only. It does not perform employee decisions and does not claim a fairness, compliance, or production-SLO certification.

## Local setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
.
\start_streamlit.ps1
```

On Linux/macOS:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
./start_streamlit.sh
```

Open `http://localhost:8501`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `development`, `staging`, or `production` |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `DATA_PATH` | `data/employee_data.csv` | Default dataset path |
| `ARTIFACT_DIR` | `artifacts` | Baseline/optimized artifact directory |
| `MONITORING_DIR` | `artifacts/monitoring` | Aggregate monitoring persistence directory |
| `ALLOW_ARTIFACT_WRITES` | `true` | Set `false` for read-only deployments |

Never commit `.env` or `.streamlit/secrets.toml`.

## Docker

```powershell
docker build -t workforce-insights:m2.7 .
docker run --rm -p 8501:8501 workforce-insights:m2.7
```

Verify:

```powershell
curl.exe -I http://localhost:8501/_stcore/health
```

Baseline and optimized model files are not required to be embedded in the container. The app trains and compares models from `data/employee_data.csv` at startup. Container monitoring writes require a writable volume or external object storage. This repository's persistence is local filesystem-only.

## Render

`render.yaml` defines a Docker web service and `/\_stcore/health` health check.

1. Create a new Render Blueprint or Docker Web Service from this repository.
2. Set `APP_ENV=production`.
3. Set `ALLOW_ARTIFACT_WRITES=true` only if the target filesystem must retain monitoring output; most hosted ephemeral filesystems require external persistence.
4. Deploy and verify the service health URL.

For production monitoring persistence, mount a persistent disk or implement object-backed storage before claiming durable monitoring history.

## Streamlit Community Cloud

Set the repository entry point to `app.py`, install `requirements.txt`, and configure optional environment values. Do not place credentials in repository files. The supplied app should use the repository's CSV and baseline artifacts.

## Health and smoke checks

```powershell
python -m pytest -q
python -m ruff check src/ app.py tests/
python -m py_compile app.py src/attrition.py src/model_monitoring.py
python -m streamlit run app.py --server.headless true --server.port 8501
curl.exe -I http://127.0.0.1:8501/_stcore/health
```

## Troubleshooting

- Missing dataset: verify `DATA_PATH` points to a valid CSV.
- Missing package: run `python -m pip install -r requirements.txt`.
- File upload rejected: use CSV or TXT for monitoring and CSV for the sidebar dataset.
- Write denied: set `ALLOW_ARTIFACT_WRITES=true` or provide a writable artifact volume.
- Corrupt monitoring history: remove only the named `artifacts/monitoring/monitoring_history.json` file and rerun; preserve baseline artifacts.

## Security and privacy

- No credentials are required by the educational prototype.
- Uploaded and monitoring data are processed in memory and aggregate monitoring reports are stored locally.
- Monitoring exports exclude individual records and identifiers.
- Use HTTPS and access controls at the hosting layer.
- Do not use predictions for employment decisions.
