# Milestone 2.5 Implementation Plan – Model Comparison, Hyperparameter Optimization & Persistence

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI  
**Date:** 2026-11-27  
**Status:** Checkpoint 1 Approved; Checkpoint 2 Blocked Pending Baseline Reconciliation

---

## 1. Initial Baseline Findings

The initial baseline was rechecked before proceeding. Historical reports state that Milestone 2.4 contains 53 tests, but the currently collected pre-optimization modules do not match that inventory. This discrepancy is a Checkpoint 2 stop-gate; no tests have been removed or skipped in this task.

- **Current pre-optimization test modules:** `pytest -q tests/test_analytics.py tests/test_explainability.py tests/test_pipeline.py tests/test_workforce_analytics.py` → **54 passed**, **0 failed**, in `64.26s`.
- **Current per-file collection:** analytics 12, explainability 18, pipeline 11, workforce analytics 13 (**54 total**). The Milestone 2.4 report instead records analytics 11, explainability 18, pipeline 13, workforce analytics 11 (**53 total**). The historical 53-test inventory cannot be reconciled by count alone.
- **Optimization tests:** `pytest -q tests/test_optimization.py` → **7 passed**, **0 failed**, in `39.23s`, with no warnings.
- **Complete suite:** `pytest -q` → **61 passed**, **0 failed**, in `99.59s`, with no warnings. `pytest --collect-only -q tests` also reports **61 tests** (54 in the pre-optimization modules and 7 optimization tests).
- **Bounded-computation finding (resolved in Checkpoint 2 reconciliation):** The baseline Random Forest remains `n_jobs=-1` for backward compatibility. `optimize_models()` now clones candidates, sets any non-`None` nested `n_jobs` parameter to `1`, fixes `RandomizedSearchCV(n_jobs=1)`, and rejects outer `n_jobs` values other than `1`. The optimization tests verify the baseline and optimized estimator settings and the rejection guard.
- **Dependency check:** The project already declares the required packages in `requirements.txt`. Pylance's selected `.venv` resolves `sklearn`, `joblib`, pandas, NumPy, Streamlit, and the other existing project modules; installed scikit-learn is 1.9.0 and joblib is 1.4.2. No dependency addition is indicated.
- **Quality checks:** Ruff currently reports **48 findings** (exit code 1; see the reconciliation report for rule counts); `python -m py_compile src/attrition.py tests/test_optimization.py app.py` exits `0`. Streamlit startup was not rerun in this checkpoint.
- **Artifacts:** No artifact was written or modified during this baseline check.

**Checkpoint decision:** Stop before Checkpoint 2 sign-off. The bounded-parallelism issue is corrected and validated, but the documented 53-test baseline still cannot be fully reconciled against the 54 current pre-optimization tests: the workspace has no Git history, the supplied GitHub repository is empty, and the historical QA report does not enumerate all test names or assertions. Optimization code and tests are present in the workspace; their presence is not a completion or correctness claim.

---

## 2. Mandatory Architectural Conditions Incorporated

1. **Preserve All 53 Original Tests:** Do not delete, weaken, or skip any Milestone 2.4 test. Report the historical original count (53), newly added tests, and current total separately at every checkpoint. Do not claim preservation or sign off until the current inventory is reconciled with the 53-test historical baseline.
2. **Shared Deterministic Train/Test Split:** A single deterministic `split_dataset(data, random_state=42, test_size=0.20)` function produces `(X_train, X_test, y_train, y_test, cleaning_report)`. Both baseline evaluation and hyperparameter optimization evaluate on the **exact same held-out `X_test` / `y_test` indices**.
3. **Zero Data Leakage & Safe Deduplication Rationale:**
   - All learned preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) stays inside `sklearn.pipeline.Pipeline` and is fitted strictly inside CV training folds and `X_train`.
   - Hyperparameter search (`RandomizedSearchCV`) runs strictly on `(X_train, y_train)` using `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`. `X_test` is never accessed during CV, tuning, or model selection.
   - **Why deterministic duplicate removal before splitting is safe:** `clean_dataset()` calls `data.drop_duplicates()` deterministically without computing any learned population statistics (no means, medians, variances, or category frequencies). Removing exact duplicates *before* `train_test_split` is required to prevent identical records from appearing simultaneously in both `X_train` and `X_test`, which would otherwise cause train-test contamination.
4. **Bounded Computation & User-Triggered UI:**
   - Uses `RandomizedSearchCV` with bounded parameter spaces, `n_iter` capped (default `6`, configurable low limit for tests), and `n_jobs=1` initially.
   - Streamlit optimization is strictly user-triggered (button/control + `st.session_state`), never executed automatically on reruns.
5. **Separate Optimized Artifacts:**
   - Baseline `artifacts/best_model.joblib` and `artifacts/training_report.json` are **never** overwritten by the optimized workflow.
   - Optimized artifacts are stored separately (`artifacts/optimized_best_model.joblib` and `artifacts/optimized_training_report.json`), and tests use `tmp_path`.
6. **Backward Compatibility:** Existing `train_and_compare`, `save_model`, `load_model`, `app.py` baseline/SHAP/analytics sections, and `src/explainability.py` continue to work identically.
7. **No Unnecessary Dependencies:** Existing packages satisfy all requirements.
8. **Checkpoint Reporting:** Each checkpoint reports changed files, tests executed (original, new, total), actual outputs, and any unresolved issues.

---

## 3. Implementation Sequence

- **Checkpoint 1 (Approved; baseline reconciliation pending):** Repository inspection, dependency check, and approved plan. The current test inventory and nested parallelism findings are recorded above.
- **Checkpoint 2 (Blocked; not signed off):** Core ML & Persistence in `src/attrition.py` + unit tests in `tests/test_optimization.py`:
  - `split_dataset()` for shared deterministic splitting.
  - Timing (`train_time_sec`, `predict_time_sec`) and per-model error handling in `train_and_compare()`.
  - `optimize_models()` using bounded `RandomizedSearchCV` (`n_jobs=1`, `StratifiedKFold`, `X_train` only).
  - `compare_baseline_and_optimized()` with independent metrics, absolute + relative deltas, NaN exclusion, and deterministic CV-based selection.
  - `save_optimized_model()` / `load_optimized_model()` and contextual error handling for missing/corrupted artifacts.
  - Before sign-off: obtain the historical test source/name inventory or another auditable baseline snapshot, then compare every original assertion. Separate original, new, and combined test runs are recorded in the Checkpoint 2 reconciliation report.
- **Checkpoint 3 (Pending Checkpoint 2 Review):** Integrate Model Comparison charts, user-triggered Hyperparameter Optimization UI, Best Parameters/Summary, and SHAP compatibility into `app.py` (preserving all existing sections) + `README.md` + final `QA_REPORT_MILESTONE_2_5_FINAL.md`.
