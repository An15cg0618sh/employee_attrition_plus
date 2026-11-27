# Milestone 2.5 – Checkpoint 3 Implementation Plan

**Date:** 2026-11-27  
**Status:** Inspection complete; implementing dashboard integration

## 1. Verified baseline

- The user-provided Checkpoint 2 reconciliation records 53 historical M2.4 tests, one additional pre-optimization test, and seven M2.5 optimization tests.
- Current collection: **61 tests**. Fresh baseline run: **61 passed** in 58.23s.
- Current branch is `main` at `5bd879c` (`first commit`), tracking `origin/main`.
- `MILESTONE_2_5_CHECKPOINT_2_RECONCILIATION.md` is already staged in Git. Treat it as a user change; do not overwrite or unstage it.
- Baseline artifacts are `artifacts/best_model.joblib` and `artifacts/training_report.json`. No optimized production artifacts currently exist.
- No dependencies need to be added: Streamlit, Plotly, pandas, NumPy, scikit-learn, SHAP, and joblib are already declared and installed.

## 2. Existing dashboard architecture

`app.py` is a single-page Streamlit dashboard:

1. Loads an uploaded or default CSV, validates the schema, and creates a deduplicated analytics frame.
2. Uses `@st.cache_data` for baseline `train_and_compare()` and global SHAP calculations.
3. Shows baseline comparison metrics near the top.
4. Provides workforce filters, four aggregate analytics tabs, and aggregate-only CSV export.
5. Provides a baseline model selector, threshold metrics, ROC/PR/confusion charts, and button-triggered permutation importance.
6. Provides three global SHAP tabs and local SHAP explanations.

No `st.session_state` usage exists yet. The dashboard currently relies on cached baseline training and standard widget state. There is no model-optimization or model-version selector in `app.py`.

The app's active workforce analytics use `src/workforce_analytics.py`; `src/analytics.py` remains a separate tested helper module. There is no explicit fairness audit UI or fairness-metric implementation in the current app/source; existing bias, non-causation, privacy, and educational-use notices must remain intact.

## 3. Existing model integration points

- `src/attrition.py::TrainingResult` and `OptimizationResult` have the fields needed by downstream metrics and explainability.
- `train_and_compare(data)` returns baseline results, a comparison table, and cleaning metadata.
- `optimize_models(data, random_state=42, n_iter=6, n_jobs=1, cv_splits=3, candidate_models=None, search_spaces=None)` returns successful optimized results, a comparison table, cleaning metadata, and candidate failures.
- `get_candidate_models()` and `get_hyperparameter_search_spaces()` provide the existing supported candidates and bounded spaces.
- `compare_baseline_and_optimized()` calculates baseline/optimized deltas.
- `save_optimized_model()` / `load_optimized_model()` use filenames separate from baseline artifacts.
- `threshold_metrics()` and `get_permutation_importance()` accept baseline or optimized result objects.
- `src/explainability.py` accepts sklearn pipelines for Random Forest and Logistic Regression; existing optimization tests verify optimized local SHAP reconstruction.

### Concrete integration gaps to address

1. Optimized results currently contain CV F1 mean/std but not CV ROC-AUC mean, which the dashboard must display. Extend the existing search to collect ROC-AUC in the same CV run while continuing to select parameters by CV F1.
2. `compare_baseline_and_optimized()` checks test-label indices but not exact `X_test` equality. Add that guard so dashboard comparisons reject mismatched held-out feature records.
3. `cached_global_shap()` names its pipeline parameter `_pipeline`, which excludes the model from Streamlit's cache key. Baseline and optimized pipelines on the same test frame could reuse stale explanations. Add an explicit model cache key and include it in the cache key.
4. The current estimator comparison helper returns `0.0` relative change when the baseline is zero and uses a signed denominator. Implement the requested `abs(baseline)` denominator and expose zero-baseline relative changes as unavailable, not fabricated zero improvement.
5. Optimized search time includes CV fitting and refit; display it as “Optimization + refit time,” distinct from baseline training time.

## 4. Planned UI changes

### Model Comparison & Performance Evaluation

- Add a clearly labelled section with baseline and optimized rows for each successfully available model.
- Show Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, CV F1 mean/std, CV ROC-AUC mean, baseline training duration, optimized tuning/refit duration, and prediction duration.
- Display a formatted comparison table, grouped held-out metric chart, separate CV chart, and explicit interpretation of metric trade-offs.
- Before optimization, show an informative empty state; never fill optimized columns with fabricated values.
- Render missing/non-finite values safely and label relative deltas unavailable when the baseline metric is zero.

### Hyperparameter Optimization

- Use a Streamlit form with LR/RF/Both selection and an iteration control bounded to 1–6.
- Run only from the explicit submit button; changing filters or other widgets must not retrigger optimization.
- Show a spinner/status while the synchronous, single-job search is running.
- Store successful results and comparison metadata in session state. Build new values locally and replace the previous successful state only if the new run yields at least one successful candidate.
- Show each failed candidate's error while retaining successful results from the same run.
- Persist the best successful candidate selected by training-only CV F1 through `save_optimized_model()` only after the user-triggered run. Never write baseline artifact paths.

### Model/version selection and downstream behavior

- Preserve the existing baseline family selector and add Baseline/Optimized version choices only for models available in the current session.
- Keep failed candidates out of selectors.
- Use the selected result consistently for threshold metrics, permutation importance, global SHAP, and local SHAP.
- Fix SHAP cache identity so switching model/version cannot display another pipeline's cached explanation.
- Preserve existing analytics filters/tabs and all Responsible AI notices; add the required educational model-comparison notice.
- Do not add employee-level ranking, decision recommendations, or exports.

## 5. Session-state strategy

Use namespaced keys such as:

- `m25_optimization_results`
- `m25_optimization_comparison`
- `m25_optimization_failures`
- `m25_optimization_cleaning_report`
- `m25_optimization_run_id`
- `m25_optimization_status`

The submitted form is the only optimization call site. A failed/all-failed run updates status and errors but does not replace the previous successful result. Filter/widget reruns only read session state. Model selectors are derived from successful baseline and session-held optimized results.

## 6. Implementation sequence

1. Extend the existing search result's CV metadata with ROC-AUC using multi-metric scoring and `refit="f1"`; retain the bounded seed/fold/parallelism settings.
2. Harden test-split comparison (`X_test` and `y_test`) and zero-baseline delta handling.
3. Add the user-triggered optimization form and transactional session-state update in `app.py`.
4. Add comparison table/charts, model/version selection, and SHAP cache-key correction.
5. Update README setup, optimization behavior, bounded search, CV/test isolation, artifact locations, SHAP, and limitations.
6. Add focused tests for the new CV metric, exact test-frame guard, zero-baseline deltas, and UI state behavior without weakening existing tests.
7. Run targeted tests, all 61 existing tests plus additions, collection, compilation, Ruff, Streamlit health, and browser interactions. Compare Ruff counts before/after; do not suppress rules.

## 7. Risks and stop gates

- Full-data Random Forest optimization is expensive even with six candidates, three folds, and one job. Keep it explicitly submitted and allow choosing one model/one iteration for interactive validation.
- `OptimizationResult` contains fitted pipelines and held-out data; keep it in the active Streamlit session only and never export row-level data.
- Cached SHAP results must be keyed by model/version as well as input data.
- Do not imply a performance improvement, fairness, or causation from a metric delta.
- Keep the staged reconciliation report untouched.
- Stop if any original test fails, the total count drops, an optimization runs on ordinary rerun, baseline artifacts change, or held-out test data influences selection.
- Do not claim fairness auditing exists; preserve the current governance notices and report the absence of a dedicated fairness audit feature.
