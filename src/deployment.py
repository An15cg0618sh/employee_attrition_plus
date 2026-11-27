"""Deployment configuration validation and environment helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ALLOWED_ENVIRONMENTS = {"development", "staging", "production"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def get_deployment_config() -> dict[str, Any]:
    """Read deployment settings without exposing or logging secret values."""
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if app_env not in ALLOWED_ENVIRONMENTS:
        raise ValueError(
            f"APP_ENV must be one of {sorted(ALLOWED_ENVIRONMENTS)}."
        )
    if log_level not in ALLOWED_LOG_LEVELS:
        raise ValueError(
            f"LOG_LEVEL must be one of {sorted(ALLOWED_LOG_LEVELS)}."
        )

    allow_writes = os.getenv("ALLOW_ARTIFACT_WRITES", "true").strip().lower()
    if allow_writes not in {"true", "false"}:
        raise ValueError("ALLOW_ARTIFACT_WRITES must be true or false.")

    data_path = Path(os.getenv("DATA_PATH", "data/employee_data.csv"))
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    monitoring_dir = Path(os.getenv("MONITORING_DIR", "artifacts/monitoring"))
    return {
        "app_env": app_env,
        "log_level": log_level,
        "data_path": data_path,
        "artifact_dir": artifact_dir,
        "monitoring_dir": monitoring_dir,
        "allow_artifact_writes": allow_writes == "true",
    }
