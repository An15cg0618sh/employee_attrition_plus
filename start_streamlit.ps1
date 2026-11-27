$ErrorActionPreference = "Stop"

$env:APP_ENV = if ($env:APP_ENV) { $env:APP_ENV } else { "development" }
$env:LOG_LEVEL = if ($env:LOG_LEVEL) { $env:LOG_LEVEL } else { "INFO" }
$env:DATA_PATH = if ($env:DATA_PATH) { $env:DATA_PATH } else { "data/employee_data.csv" }
$env:ARTIFACT_DIR = if ($env:ARTIFACT_DIR) { $env:ARTIFACT_DIR } else { "artifacts" }
$env:MONITORING_DIR = if ($env:MONITORING_DIR) { $env:MONITORING_DIR } else { "artifacts/monitoring" }
$env:ALLOW_ARTIFACT_WRITES = if ($env:ALLOW_ARTIFACT_WRITES) { $env:ALLOW_ARTIFACT_WRITES } else { "true" }

python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
