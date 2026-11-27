# QA Report - Milestone 2.5: Complete Regression, Model Comparison, Optimization and Deployment Readiness Audit

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI  
**Audit Date:** 2026-11-27  
**Final Milestone Verdict:** **FAIL**

## 1. Executive Summary

An independent audit was performed against the actual repository. No source code or test code was changed during the audit.

- The repository does not implement Milestone 2.5 hyperparameter optimization or baseline-versus-optimized functionality.
- The expected `MILESTONE_2_5_IMPLEMENTATION_PLAN.md` and `MILESTONE_2_5_IMPLEMENTATION_REPORT.md` are absent.
- The prior Milestone 2.4 baseline is **not reproducible** in the fresh validation environment: 52 tests pass and 1 test fails.
- `python -m ruff check src/ app.py tests/` reports 42 findings.
- `python -m py_compile app.py` passes.
- Streamlit starts successfully and returned HTTP 200 at `/_stcore/health`; live browser verification confirmed dashboard rendering and existing baseline interactions.
- The browser-verified implementation contains no model-optimization controls, optimized metrics, or baseline-versus-optimized display.

## 2. Previous Milestone 2.4 Regression Baseline

The prior report documented 53 passing tests, 0 failures, 0 skips, 0 warnings, Ruff passing, compilation passing, and Streamlit startup passing.

The fresh run performed for this audit found:

| Check | Milestone 2.4 Reported Baseline | Fresh Observed Result | Status |
|---|---:|---:|---|
| pytest test count | 53 passed | 52 passed, 1 failed | FAIL |
| Failures | 0 | 1 | FAIL |
| Skips | 0 | 0 | PASS |
| Warnings | 0 | 0 | PASS |
| Ruff | Passing | 42 findings | FAIL |
| Python compilation | Passing | Exit code 0 | PASS |
| Streamlit startup | Passing | HTTP 200 on port 8511 | PASS |

The previous 53-test result is therefore not retained in the fresh environment.

The first environment attempt lacked declared runtime packages; `requirements.txt` and Ruff were restored into `.venv-new2`. The restored environment used Python 3.13.12, pandas 3.1.1, scikit-learn 1.9.1, Streamlit 1.67.0, Plotly 7.2.0, SHAP 0.54.1, Ruff 0.17.6, and pytest 9.2.0. Dependency restoration is environment-only; it did not modify repository source.

## 3. Files Inspected

- `app.py`
- `train.py`
- `src/attrition.py`
- `src/explainability.py`
- `src/workforce_analytics.py`
- `src/analytics.py`
- `tests/test_pipeline.py`
- `tests/test_explainability.py`
- `tests/test_workforce_analytics.py`
- `tests/test_analytics.py`
- `artifacts/best_model.joblib`
- `artifacts/training_report.json`
- `README.md`
- `requirements.txt`
- `QA_REPORT_MILESTONE_2_4_FINAL.md`
- `QA_REPORT_MILESTONE_2_5_FINAL.md`
- `data/employee_data.csv`

Missing expected files:

- `MILESTONE_2_5_IMPLEMENTATION_PLAN.md`
- `MILESTONE_2_5_IMPLEMENTATION_REPORT.md`

## 4. Files Modified During Audit

No source, test, dataset, artifact, or configuration files were modified.

The only audit deliverable modified was this QA report. Dependency packages were restored in `.venv-new2` only.

## 5. Model Comparison Verification

Evidence from `src/attrition.py`:

- `get_candidate_models()` defines Logistic Regression and Random Forest.
- `train_and_compare()` uses one shared 80/20 `train_test_split(..., test_size=0.20, random_state=42, stratify=y)` for both candidates.
- Both candidates use `Pipeline` objects with the same `build_preprocessor(X_train)` logic.
- `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` is used for 3-fold cross-validation.
- The positive class is obtained with `predict_proba(X_test)[:, 1]`, consistent with `quit = 1`.
- Accuracy, precision, recall, F1, ROC-AUC, and PR-AUC are calculated in `calculate_metrics()`.
- CV F1 mean, CV F1 standard deviation, and CV ROC-AUC mean are calculated.
- `train.py` prints the comparison and saves the highest CV-F1 row.

Defects/gaps:

- Training and prediction durations are not measured or stored.
- The baseline loop has no per-model exception handling; one failed model aborts the comparison.
- No baseline metric charts are specifically implemented for Milestone 2.5; the app renders a table and separate threshold/curve charts.
- There is no invalid-metric exclusion or robustness against non-finite results.

Status: **PARTIAL**

## 6. Hyperparameter Optimization Verification

A full targeted search found no `GridSearchCV`, `RandomizedSearchCV`, bounded search-space definition, optimization function, optimization duration, best parameters, or best CV score in `src/`, `app.py`, `train.py`, or `tests/`.

No lightweight optimization run was possible because no optimization implementation exists.

Status: **FAIL**

## 7. Data Leakage Audit

The complete baseline flow is traceable:

`load_dataset/validate_dataset_frame` -> `clean_dataset` -> `train_test_split` -> `build_preprocessor(X_train)` inside `Pipeline` -> `cross_validate` with `StratifiedKFold` -> `pipeline.fit(X_train, y_train)` -> held-out `predict_proba(X_test)[:, 1]` -> test metrics.

Supporting evidence:

- `clean_dataset()` removes exact duplicates before splitting.
- `train_test_split()` occurs once before candidate-model fitting.
- `build_preprocessor(X_train)` is used to construct each pipeline; no held-out test frame is supplied to preprocessing fitting.
- Numeric median imputation and scaling, categorical most-frequent imputation and one-hot encoding, are pipeline steps.
- `cross_validate()` receives `X_train` and `y_train`; the preprocessing and model are refitted inside each CV fold.
- `X_test` is used only after fitting and cross-validation for each candidate.
- The target has binary validation and is converted to integer.

No hyperparameter selection step exists, so test-set exposure from optimization cannot occur. However, the absence of optimization is a missing-feature failure, not evidence that an optimization implementation is leak-free.

Baseline leakage status: **PASS**  
Hyperparameter leakage status: **FAIL / NOT IMPLEMENTED**

## 8. Baseline vs Optimized Performance

Only baseline model metrics are present.

- `artifacts/training_report.json` retains baseline selected-model metrics and CV metrics.
- It contains no baseline-versus-optimized delta table.
- It contains no independent optimized test metrics.
- No best-model criterion beyond comparison sorting by `cv_f1_mean` is implemented.
- The `comparison` DataFrame is deterministically sorted in descending CV F1.
- Invalid-metric exclusion is absent.

Status: **FAIL**

## 9. Model Persistence Verification

Baseline artifact persistence is implemented with `joblib.dump` and `joblib.load` in `src/attrition.py`.

Automated save/load evidence:

- `test_saved_model_round_trip` passes in the fresh suite.
- It uses `tmp_path`, so existing artifacts are not overwritten.
- Reloaded and original Random Forest probabilities match to tight tolerances.
- `artifacts/best_model.joblib` and `artifacts/training_report.json` were not modified during the audit.

Missing verification:

- No optimized model artifact.
- No optimization metadata artifact.
- No corrupted-artifact handling.
- Missing-artifact handling is not explicit in `load_model()`, beyond raw `joblib.load` exceptions.

Status: **PARTIAL**

## 10. SHAP Compatibility

Baseline SHAP support is implemented in `src/explainability.py`:

- Random Forest uses `TreeExplainer` and positive-class slicing.
- Logistic Regression uses `LinearExplainer` with an independent background masker and raw log-odds output.
- Transformed feature names align with the explanation array.
- Global importance, summary, impact direction, and local waterfall explanations are available.
- Tests cover mathematical reconstruction for both baseline models.

No optimized model exists, so optimized-model SHAP support is necessarily absent. The UI does not expose unsupported optimized models or an unsupported-model limitation because no optimization model is exposed.

Status:

- Existing RF/LR compatibility: **PASS**
- Optimized-model compatibility: **FAIL / NOT IMPLEMENTED**

## 11. Browser Verification

Live browser verification was performed.

Verified at `http://127.0.0.1:8511`:

- Streamlit health endpoint returned HTTP 200.
- Dashboard title loaded.
- Dataset Overview metrics rendered.
- `11,582` uploaded records, `11,573` deduplicated records, `17.2%` observed attrition, and `9` duplicates rendered.
- Sidebar Department, Salary, tenure range, work-accident, promotion, and minimum reliable group size controls rendered.
- Department, Salary, Tenure, and Satisfaction & Performance tabs rendered.
- Salary, tenure, and satisfaction tab visualizations rendered.
- The aggregate CSV data URL rendered and was privacy-safe with aggregate headers and no individual records.
- Threshold explorer rendered and changed from `0.50` to `0.55` live.
- Logistic Regression was selected from the model dropdown; metrics rerendered to the LR values.
- Three global SHAP tabs remained accessible.
- Individual SHAP and exact reconstruction UI remained present.
- Responsible-use and data-quality notices remained present.

Not verified or absent:

- No Milestone 2.5 model-comparison metric chart.
- No user-triggered optimization control.
- No optimization progress/completion message.
- No best parameters display.
- No best optimized-model summary.
- No optimization-only download/persistence interaction exists.
- Browser-native aggregate download completion was not separately recaptured in this audit.

## 12. Responsible AI and Privacy Audit

Verified from `app.py`, `src/attrition.py`, `src/workforce_analytics.py`, `src/explainability.py`, and browser rendering:

- No individual employee ranking UI or logic was found.
- No termination, promotion, hire, or penalty logic was found.
- No employee-level model-comparison export was found.
- No individual SHAP export was found.
- Comparison and workforce exports remain aggregate.
- Minimum reliable group size and small-sample safeguards remain in the UI.
- Non-causation notices remain visible.
- Educational-prototype disclaimers remain visible.
- Human-review and privacy cautions remain visible.

Status: **PASS**

## 13. Automated Test Results

### pytest

Command:

```text
.venv-new2\Scripts\python.exe -m pytest -v
```

Fresh result:

- Collected: 53 items
- Passed: 52
- Failed: 1
- Skipped: 0
- Warnings: 0
- Duration: 62.35 seconds

Failure:

- `tests/test_analytics.py::test_tenure_grouping`
- `assign_tenure_group()` returns `NaN` for one of the invalid/nullable tenure inputs instead of the assertion's required `"Unknown"` value.
- Location: `src/analytics.py::assign_tenure_group()`
- Root cause: pandas 3 tenure categorization/`astype(str)` behavior does not convert the final missing-category value to `"Unknown"` in this environment. The test expects explicit `"Unknown"` for `None`, `np.nan`, and `"invalid"`.

### Ruff

Command:

```text
.venv-new2\Scripts\python.exe -m ruff check src/ app.py tests/
```

Fresh result:

- 42 findings
- Ruff exit code 1
- Findings include import ordering and broad `Exception` catches in `app.py`, plus additional existing lint violations across the audited source and tests.

This differs from the prior report's "All checks passed."

### Python compilation

Command:

```text
.venv-new2\Scripts\python.exe -m py_compile app.py
```

- Exit code 0

### Streamlit startup

Command:

```text
.venv-new2\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8511
```

- Health endpoint: HTTP 200
- Live browser: title and dashboard loaded
- Server stopped after browser audit

## 14. Issues and Root-Cause Analysis

### Critical - Milestone 2.5 is not implemented

- **Evidence:** No optimization implementation or documentation exists.
- **Impact:** Hyperparameter optimization, deployment metadata, optimized persistence, and baseline-versus-optimized evaluation are unavailable.
- **Root cause:** The repository remains at the Milestone 2.4 baseline implementation.
- **Required correction:** Implement bounded search and independent evaluation in `src/attrition.py`, persist optimization metadata and the best fitted pipeline, add isolated tests, and integrate user-triggered controls in `app.py`.

### Critical - Milestone 2.4 automated baseline regressed

- **Evidence:** 52 passed / 1 failed.
- **Impact:** The required previous 53-pass baseline is not retained.
- **Root cause:** The tenure grouping edge case fails in the fresh declared-environment validation; exact source behavior is in `src/analytics.py::assign_tenure_group()`.
- **Required correction:** Preserve existing assertions and fix tenure missing/invalid normalization.

### Moderate - Ruff baseline is not passing

- **Evidence:** 42 findings in a fresh Ruff run.
- **Impact:** The documented clean Ruff state is not reproducible.
- **Root cause:** Existing source has lint violations, including unsorted imports and broad exception catches.
- **Required correction:** Clean only relevant, repository-supported violations without weakening lint rules.

### Moderate - Artifact error handling is incomplete

- **Evidence:** `load_model()` directly calls `joblib.load(path)` without contextual missing/corrupted artifact handling.
- **Impact:** UI error behavior depends on raw library exceptions.
- **Required correction:** Return or surface contextual errors and test missing and corrupted temporary artifacts.

### Moderate - Baseline comparison lacks timing and per-model error handling

- **Evidence:** `train_and_compare()` has no timing fields and one candidate exception aborts the loop.
- **Impact:** Milestone 2.5 timing and graceful-failure requirements are unsatisfied.
- **Required correction:** Add bounded timing data and isolate candidate failures.

## 15. Outstanding Limitations

- Milestone 2.5 implementation and documentation do not exist.
- No optimization run or optimized artifact can be evaluated.
- Browser verification does not include a completed aggregate CSV download event.
- Some browser interactions, including all sidebar filter combinations and full SHAP tab switching, are only partially verified and should be manually retested after implementation.
- A dependency restoration changed the local environment but not repository files.

## 16. Requirement-by-Requirement Acceptance Table

| Criterion | Status | Supporting Evidence |
|---|---|---|
| 53 prior tests remain passing | FAIL | Fresh run: 52 passed, 1 failed |
| No skips/warnings | PASS | Fresh pytest: 0 skips, 0 warnings |
| Ruff passes | FAIL | 42 findings |
| Python compilation passes | PASS | `py_compile app.py`, exit 0 |
| Streamlit startup passes | PASS | HTTP 200, live browser loaded |
| M2.5 plan/report documents exist | FAIL | Both files absent |
| All baseline classifiers evaluated | PASS | Logistic Regression and Random Forest in `get_candidate_models()` |
| Equivalent baseline splits/preprocessing | PASS | Shared stratified split and pipeline preprocessor |
| Positive class quit=1 | PASS | `predict_proba(... )[:, 1]` and binary target validation |
| All baseline metrics/CV metrics calculated | PASS | Six test metrics plus CV F1, CV F1 SD, CV ROC-AUC |
| Training/prediction times recorded | FAIL | No timing fields |
| Failed models handled gracefully | FAIL | No per-model exception handling |
| Hyperparameter optimization implemented | FAIL | No search implementation |
| Search uses stratified CV/training-only data | FAIL | No search exists |
| Fixed seeds reproducible | PARTIAL | Baseline seeds fixed; optimization absent |
| Search spaces bounded | FAIL | No search spaces |
| Optimization duration/best metadata saved | FAIL | No metadata |
| Imputation/encoding are pipeline-local | PASS | Preprocessing steps inside fitted pipelines |
| Test data excluded from model selection | PASS | Baseline selection is CV-only before held-out test evaluation |
| No target leakage | PASS | Target is isolated and test only used after fit/CV |
| Baseline metrics retained | PASS | Artifact and report contain baseline metrics |
| Optimized metrics independently calculated | FAIL | No optimized model |
| Metric delta math implemented | FAIL | No delta table |
| Deterministic best-model criterion | PASS | Comparison sorted descending by CV F1 mean |
| Optimized model save/reload | FAIL | No optimized artifact |
| Reload predict-proba consistency | PARTIAL | Baseline Random Forest round trip passes only |
| Missing/corrupted artifact handling | FAIL | No explicit handling/test |
| Baseline RF/LR SHAP | PASS | Tests and browser UI verify existing support |
| Optimized-model SHAP support/limitation | FAIL | No optimized models/UI limitation |
| Dataset Overview | PASS | Live browser verified |
| Workforce Analytics tabs | PASS | All four tabs live-verified |
| Existing sidebar filters | PASS | Controls rendered; full combination matrix NOT TESTED |
| Threshold explorer | PASS | Live threshold changed 0.50 to 0.55 |
| Global SHAP | PASS | Live global section and three tabs rendered |
| Individual SHAP | PASS | Live section and reconstruction control rendered |
| M2.5 comparison UI | FAIL | No model-optimization section |
| Optimization user action/progress/result | FAIL | Not implemented |
| Existing model selection remains functional | PASS | RF/LR live switching verified |
| No individual ranking/actions/export | PASS | No implementation/export found |
| PII and individual SHAP privacy | PASS | Aggregate exports and privacy notices verified |
| Small-group safeguards | PASS | Group-size control and reliability flags verified |
| Non-causation notice | PASS | Visible in browser |
| Educational disclaimer | PASS | Visible in browser |

## 17. Final Milestone Verdict

**FAIL**

Milestone 2.5 functionality is not implemented. The repository also fails the fresh Milestone 2.4 regression gate with one test failure and 42 Ruff findings. No optimization implementation, optimized artifact, or baseline-versus-optimized dashboard can be accepted.
