# Milestone 2.6 Implementation Plan — Model Monitoring and Data Drift Detection

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI  
**Milestone:** 2.6 — Model Monitoring and Data Drift Detection  
**Date:** 2026-11-27  
**Status:** CHECKPOINTS 2-6 IMPLEMENTED — Final QA complete

---

## 1. Checkpoint 1 — Repository Inspection & Baseline Verification

### 1.1 Baseline Execution Results

| Check | Command | Result |
|---|---|---|
| Test suite | `.venv\Scripts\python.exe -m pytest -q` | **62 passed in 85.16s**, 0 failed, 0 skipped, 0 warnings |
| Compilation | `.venv\Scripts\python.exe -m py_compile app.py src/attrition.py` | **Exit code 0** |
| Ruff lint | `.venv\Scripts\python.exe -m ruff check src/ app.py tests/` | **49 pre-existing findings** (exit code 1) |

**Test count breakdown:**
- Historical Milestone 2.4 tests: **53**
- Existing additional pre-M2.5 test (`test_tenure_grouping_missing_and_boundary_values` in `tests/test_analytics.py`): **1**
- Accepted Milestone 2.5 optimization tests (`tests/test_optimization.py`): **8**
- **Verified Milestone 2.5 baseline total:** **62**

**Ruff baseline note:** No `pyproject.toml` or `ruff.toml` exists in the repository. All 49 Ruff findings are pre-existing (`I001` import sorting, `BLE001` blind `Exception` in `app.py`, `C408` `dict()` calls in `src/analytics.py`, `RUF059`/`PERF102` in tests). All new Milestone 2.6 files (`src/model_monitoring.py`, `tests/test_monitoring.py`, and new `app.py` additions) will be kept Ruff-clean.

### 1.2 Git & Artifact Inventory

- **Current branch:** `main` (`1850ec0 Update Milestone 2.5 files`)
- **Uncommitted local changes (preserved, untouched):**
  - `M QA_REPORT_MILESTONE_2_5_FINAL.md`
  - `M README.md`
  - `M app.py`
  - `M tests/test_optimization.py`
- **Artifacts directory (`artifacts/`):**
  - `artifacts/best_model.joblib` (16,769,531 bytes)
  - `artifacts/training_report.json` (569 bytes)
  - `artifacts/optimized_best_model.joblib` and `artifacts/optimized_training_report.json`: not present on disk yet (created only when user triggers optimization in the UI).
  - `artifacts/monitoring/`: does not exist yet.

### 1.3 Existing Architecture & Gaps Identified

- **`app.py`:** Uses `@st.cache_data` for `cached_training` and `cached_global_shap`, `joblib.hash(cleaned_data)` as `data_fingerprint`, `st.session_state` (`m25_*`) invalidated on dataset fingerprint changes, user-triggered `st.form("m25_optimization_form")`, baseline/optimized model selection, and privacy-safe aggregate CSV export.
- **`src/attrition.py`:** Provides `EXPECTED_COLUMNS`, `validate_dataset_frame`, `clean_dataset`, `split_dataset`, `build_preprocessor`, `calculate_metrics`, `train_and_compare`, `optimize_models`, `compare_baseline_and_optimized`, `save_optimized_model`, and `load_optimized_model`.
- **Environment dependencies:** `scipy 1.15.2` is already installed (providing `scipy.stats.ks_2samp` and `scipy.stats.chi2_contingency`), so no new dependencies are required in `requirements.txt`. `openpyxl` is not installed; ingestion will natively support `.csv` and delimited `.txt` via `pd.read_csv`, and clearly reject unsupported binary formats (such as `.xlsx` when `openpyxl` is absent) with actionable explanations.
- **Implementation gap:** No monitoring module, drift detection, prediction/performance monitoring, monitoring persistence, or monitoring dashboard section exists yet.

---

## 2. Proposed Architecture for Milestone 2.6

| Module / Path | Responsibility |
|---|---|
| `src/model_monitoring.py` | Reference dataset metadata & aggregate stats, monitoring data ingestion & validation, numerical drift (PSI + KS + summary stats), categorical drift (Chi-square + category shift/unseen/missing), data-quality monitoring, prediction distribution monitoring, delayed ground-truth performance monitoring, explainable status engine, aggregate JSON/CSV export builders, and isolated monitoring persistence |
| `tests/test_monitoring.py` | Comprehensive unit and integration tests for all monitoring, drift, quality, prediction, performance, status, and persistence paths (using `tmp_path`) |
| `app.py` | New user-triggered `Model Monitoring & Data Drift` section with `st.session_state` (`m26_*`) keyed by dataset fingerprint, model identity, and threshold config |
| `artifacts/monitoring/` | Isolated persistence (`reference_metadata.json`, `monitoring_history.json`, `reports/`) — never modifies `best_model.joblib`, `training_report.json`, `optimized_best_model.joblib`, or `optimized_training_report.json` |
| `README.md` | Updated documentation for Milestone 2.6 architecture, statistical methods, thresholds, privacy safeguards, and usage |
| `MILESTONE_2_6_IMPLEMENTATION_REPORT.md` | Final implementation and verification report |

---

## 3. Detailed Design by Feature

1. **Reference Dataset Management (`src/model_monitoring.py`):**
   - Uses the cleaned training split (`X_train` from `split_dataset(data, random_state=42)`) or cleaned reference data as the reference source, separating `quit` from feature columns.
   - Computes deterministic `joblib.hash` fingerprint, UTC ISO-8601 `created_at` timestamp, `row_count`, `feature_count`, `feature_schema` (column -> dtype/role), missing-value statistics, and aggregate reference summaries (numeric quantiles/bins/mean/median/std, categorical value counts/proportions) without storing duplicate employee-level records in `reference_metadata.json`.

2. **Monitoring Data Ingestion & Validation:**
   - Supports `.csv` and delimited `.txt` (`pd.read_csv` / `sep=None, engine="python"` fallback for `.txt`), rejecting empty files, invalid encodings, or missing `openpyxl` for `.xlsx` with clear messages.
   - Validates against the 9 expected feature columns (`EXPECTED_COLUMNS - {"quit"}`), reports missing columns, unexpected columns, incompatible numeric types, invalid numeric values, duplicate rows, and whether `quit` is present, missing, or contains invalid/partial labels.
   - Supports feature drift and prediction monitoring **without requiring `quit`**.

3. **Numerical Drift Detection:**
   - **PSI:** Reference quantile/uniform bin edges (excluding non-finite values, handling constant features safely) with small smoothing (`epsilon = 1e-6`) to prevent `log(0)`. Default thresholds: `< 0.10` Low, `0.10–0.25` Moderate, `> 0.25` Significant.
   - **Two-Sample KS Test (`scipy.stats.ks_2samp`):** Statistic, p-value, and drift flag at configurable `alpha = 0.05`, accompanied by sample-size sensitivity notice.
   - **Summary Comparison:** Reference vs. current mean, median, std, and missing percentage.

4. **Categorical Drift Detection:**
   - Compares reference vs. current category proportions, detects unseen/new categories and missing/disappeared categories, and tracks missing-value changes.
   - Runs `scipy.stats.chi2_contingency` on aligned category counts only when expected-frequency assumptions are met (e.g., >= 2 categories, total count > 0, no expected cell < 1 and >= 80% of cells >= 5); otherwise flags Chi-square as unsuitable/skipped and falls back to descriptive proportion comparison.

5. **Data Quality Monitoring:**
   - Tracks missing-value count/rate and delta vs. reference, invalid numeric coercions, unexpected categories, duplicate count, schema mismatches, out-of-range values (e.g. `satisfaction_level` / `last_evaluation` outside `[0, 1]`, negative hours/tenure), and empty columns.
   - Produces independent `Healthy` / `Warning` / `Critical` data-quality status with configurable thresholds.

6. **Prediction Distribution & Delayed Performance Monitoring:**
   - Uses `model.predict_proba(X)[:, 1]` (`quit = 1`) on the currently selected model (Baseline LR, Baseline RF, Optimized LR, or Optimized RF).
   - Monitors mean, median, std, histogram/quantile distribution, and predicted-positive rate at the active threshold, clearly labeled: *"These values are model predictions, not confirmed employee outcomes."*
   - Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and confusion matrix **only when valid binary `quit` labels are present** (handling single-class batches where ROC-AUC/PR-AUC are undefined). Displays *"Model performance cannot be verified until ground-truth outcomes are available."* when labels are absent.

7. **Monitoring Status Engine, Persistence & Caching:**
   - Evaluates `Healthy`, `Warning`, `Critical`, and `Insufficient Data` using configurable thresholds across quality, numerical drift, categorical drift, prediction shift, and (if available) performance delta, with explicit human-readable reasons. Never marks a model unusable from a single p-value alone.
   - Stores aggregate-only `reference_metadata.json`, `monitoring_history.json` (deduplicated by run signature: dataset fingerprint + model identity + threshold config), and `reports/*.json` under `artifacts/monitoring/`.
   - Uses `st.session_state["m26_*"]` keyed by `(reference_fingerprint, monitoring_fingerprint, model_identity, threshold_config)` so ordinary Streamlit reruns never re-execute monitoring automatically, and failed runs preserve the last successful report.

8. **Responsible AI & Privacy Safeguards:**
   - All exports (aggregate JSON, feature drift CSV, summary) contain aggregate statistics only—zero employee-level records, IDs, or individual predictions.
   - Clear UI notices explain that drift != bias/causation/inaccuracy and this is an educational prototype.

---

## 4. Checkpoint Rollout Plan

- **Checkpoint 1 (Current):** Repository inspection, 62-test baseline verification, and implementation plan. **STOP for approval.**
- **Checkpoint 2:** Implement core monitoring engine (`src/model_monitoring.py`: reference management, ingestion/validation, numerical drift, categorical drift, data quality, status engine) + `tests/test_monitoring.py`. **STOP for approval.**
- **Checkpoint 3:** Implement prediction distribution & delayed performance monitoring + focused tests. **STOP for approval.**
- **Checkpoint 4:** Implement isolated persistence, monitoring history, cache-key validation, and export builders + `tmp_path` tests. **STOP for approval.**
- **Checkpoint 5:** Integrate user-triggered `Model Monitoring & Data Drift` section into `app.py` with `st.session_state` (`m26_*`). **STOP for approval.**
- **Checkpoint 6:** Update `README.md`, produce `MILESTONE_2_6_IMPLEMENTATION_REPORT.md`, run full regression, Ruff, compilation, and Streamlit health checks. **STOP for final QA.**
