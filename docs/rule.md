# Repository Rules

## Coding Standards

- Preserve the existing Python, pandas, scikit-learn, Streamlit, and Ruff patterns.
- Keep ML, drift, and presentation logic modular.
- Add focused tests that fail when behavior changes.
- Do not weaken, skip, or delete existing tests.
- Do not suppress Ruff findings as a substitute for fixing implementation quality.
- Keep line and import formatting compatible with current Ruff configuration.

## Security and Environment

- Never commit real credentials or `.env` files.
- Commit `.env.example` only with neutral placeholders.
- Keep `.streamlit/secrets.toml` ignored.
- Do not log uploaded records or individual predictions.
- Keep monitoring exports aggregate-only.
- Preserve HTTPS/access-control responsibilities at the hosting layer.

## Artifact Guardrails

- Never overwrite `artifacts/best_model.joblib`.
- Never overwrite `artifacts/training_report.json`.
- Never overwrite optimized baseline/optimized artifacts during UI cleanup.
- Write monitoring output only under `artifacts/monitoring/` or the configured monitoring directory.

## Git and Release Guardrails

- Start feature work from the approved M2.6/M2.7 checkpoint.
- Use a separate branch for milestone work.
- Review staged diffs before commit.
- Verify baseline artifact hashes.
- Run pytest, Ruff, compilation, pip/dependency checks, and Streamlit health before release.

## AI/Agent Operating Rules

- Inspect before modifying.
- Preserve unrelated work.
- Prefer complete surgical changes.
- Use exact cleanup paths and inspect generated directories before deletion.
- Update documentation and tests alongside implementation changes.
- Never fabricate test, browser, commit, or deployment evidence.
