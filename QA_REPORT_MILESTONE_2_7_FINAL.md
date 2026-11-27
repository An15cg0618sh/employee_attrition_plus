# Milestone 2.7 Final QA Report

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI
**Milestone:** 2.7 - Production Deployment & Documentation
**Audit Date:** 2026-11-27
**Verdict:** PASS with documented local/container limitations

## Scope

This report records local production-readiness verification. Public hosted deployment is a separate operational action unless a hosting provider is configured and verified.

## Test Inventory

-- Full pytest suite: **93 passed** in 59.72 seconds.
- Deployment tests: **10 passed**.
- Ruff: **all checks passed**.
- Python compilation: passed.
- Dependency installation: `python -m pip install -r requirements.txt` completed successfully.
- `python -m pip check`: `No broken requirements found`.

## Deployment Verification

- Production-style Streamlit startup succeeded on port 8518.
- `/_stcore/health` returned HTTP 200.
- Browser page loaded the Workforce Insights dashboard and monitoring/workforce sections.
- Docker executable was unavailable in the local environment, so Docker image build/run could not be executed locally.
- No public hosted deployment was performed.

## Artifact Integrity

| Artifact | SHA-256 |
|---|---|
| `artifacts/best_model.joblib` | `90FB9226F8CECC95C11781ECA7119BC2405E38EB1FBA7312A36C81FAF11E6816` |
| `artifacts/training_report.json` | `E07230A78B26C44099A4D17EF19C9B9B4A9D82E42C92E1CC6CD72F7AA9B720FF` |
| `artifacts/optimized_best_model.joblib` | `BDFCD479620348287918CAEECDE9D6E8C2811528EF7AA8789311698C7350848B` |
| `artifacts/optimized_training_report.json` | `478794E1697CB572641426D11D935141C23F8A56927512B2CB530EEA67696508` |

Deployment setup did not retrain or overwrite model artifacts.

## Security

- `.env` and Streamlit secrets are ignored.
- `.env.example` contains no credentials.
- Monitoring exports remain aggregate-only.
- Baseline and optimized model paths are not overwritten by deployment setup.
- Source/config/secret scans found no credential assignments.

## Remaining Limitations

- Docker build/run was not executable in this environment.
- Public hosting/provider deployment was not performed.
- Monitoring persistence uses local filesystem storage and requires external storage for durable hosted deployments.
