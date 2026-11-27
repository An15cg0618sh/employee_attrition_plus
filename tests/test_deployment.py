from pathlib import Path

import pytest

from src.deployment import get_deployment_config


ROOT = Path(__file__).parents[1]


def test_deployment_config_defaults(monkeypatch):
    for key in (
        "APP_ENV",
        "LOG_LEVEL",
        "DATA_PATH",
        "ARTIFACT_DIR",
        "MONITORING_DIR",
        "ALLOW_ARTIFACT_WRITES",
    ):
        monkeypatch.delenv(key, raising=False)
    config = get_deployment_config()
    assert config["app_env"] == "development"
    assert config["log_level"] == "INFO"
    assert config["allow_artifact_writes"] is True


def test_deployment_config_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "unknown")
    with pytest.raises(ValueError, match="APP_ENV"):
        get_deployment_config()


def test_deployment_config_rejects_invalid_log_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "TRACE")
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        get_deployment_config()


def test_deployment_config_rejects_invalid_write_flag(monkeypatch):
    monkeypatch.setenv("ALLOW_ARTIFACT_WRITES", "sometimes")
    with pytest.raises(ValueError, match="ALLOW_ARTIFACT_WRITES"):
        get_deployment_config()


def test_env_example_contains_no_real_credentials():
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "APP_ENV=development" in env_text
    assert "ALLOW_ARTIFACT_WRITES=true" in env_text
    lowered = env_text.lower()
    assert "password=" not in lowered
    assert "api_key=" not in lowered
    assert "token=" not in lowered


def test_production_startup_configs_exist():
    assert (ROOT / ".streamlit" / "config.toml").exists()
    assert (ROOT / "Dockerfile").exists()
    assert (ROOT / "render.yaml").exists()
    assert (ROOT / "Procfile").exists()


def test_app_uses_deployment_config_and_structured_logging():
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "get_deployment_config" in app_text
    assert "logging.basicConfig" in app_text
    assert "ALLOW_ARTIFACT_WRITES=false" in app_text


def test_gitignore_protects_secrets_and_generated_monitoring():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert ".streamlit/secrets.toml" in gitignore
    assert "artifacts/" in gitignore


def test_requirements_include_runtime_drift_dependencies():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scipy>=1.10" in requirements
    assert "requests>=2.32.2" in requirements


def test_dockerfile_does_not_require_untracked_artifacts():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY artifacts" not in dockerfile
    assert "streamlit" in dockerfile
    assert "run" in dockerfile
    assert "app.py" in dockerfile
