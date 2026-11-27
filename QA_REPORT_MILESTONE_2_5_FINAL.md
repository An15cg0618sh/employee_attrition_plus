# Milestone 2.5 - Final QA Report

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI  
**Audit Date:** 2026-11-27  
**Checkpoint:** 3 - Dashboard integration and final validation  
**Overall Status:** PASS

## 1. Scope Completed

Milestone 2.5 now includes:

- Bounded, user-triggered hyperparameter optimization for Logistic Regression and Random Forest.
- A shared deterministic train/test split and pipeline-local preprocessing.
- Training-fold CV F1 parameter selection and CV ROC-AUC reporting.
- Separate baseline and optimized artifacts.
- Baseline-versus-optimized held-out test metrics and cross-validation summaries.
- Baseline/optimized version selection for evaluation, permutation importance, and SHAP views.
- Session-state retention for optimization results and failures.
- README and final QA documentation.

The implementation remains an educational workforce-analytics prototype. It does not add fairness auditing, employee-level ranking, or adverse-employment decisioning.

## 2. Core ML Controls

Verified implementation controls:

- Deduplication occurs before the shared split.
- Preprocessing is fitted inside pipeline training/CV folds.
- `RandomizedSearchCV` uses training records only.
- Search scoring includes F1 and ROC-AUC; F1 is the `refit` criterion.
- Outer and inner estimator parallelism is restricted to one job.
- Random state 42 and three stratified CV folds are used.
- Candidate iterations are bounded to 1–6.
- Baseline and optimized comparisons require exact `X_test` equality and exact `y_test` equality.
- Relative metric changes use `abs(baseline)`; a zero baseline yields `NaN` rather than fabricated improvement.
- Optimized artifacts use filenames separate from baseline artifacts.

## 3. Dashboard Integration

Optimization is available through an explicit Streamlit form. Ordinary reruns and workforce filter changes do not start optimization.

Successful session state is replaced only after at least one optimized candidate succeeds. Partial failures remain visible while successful candidates remain available. All-failed runs retain prior successful state.

Changing the dataset fingerprint resets optimization state. The global SHAP cache key includes baseline/optimized version, dataset fingerprint, model name, and optimization run identity.

The dashboard preserves the existing workforce filters, threshold explorer, permutation importance, global SHAP, local SHAP, aggregate CSV export, and responsible-use notices.

## 4. Test Coverage

The focused optimization suite covers:

- Shared deterministic split and no train/test overlap.
- Outer and inner parallelism controls.
- Reproducibility.
- Failed candidate handling.
- Metric delta arithmetic and zero-baseline handling.
- Exact held-out feature and label mismatch guards.
- Optimized artifact isolation and corruption/missing-file handling.
- Optimized pipeline SHAP reconstruction.
- Dashboard form, session-state, and SHAP cache contracts.

Historical tests remain in the suite and were not deleted, skipped, renamed, or weakened.

The test suite is now 62 tests: the preserved/pre-existing suite plus the optimization and dashboard contract coverage.

## 5. Validation Commands

```powershell
python -m pytest -q
python -m ruff check src/ app.py tests/
python -m py_compile app.py src/attrition.py tests/test_optimization.py
python -m streamlit run app.py
```

## 6. Actual Validation Results

- `python -m pytest -q`: **62 passed** in 64.35 seconds.
- `python -m py_compile app.py src/attrition.py tests/test_optimization.py`: passed.
- `python -m ruff check src/ app.py tests/`: **all checks passed**.
- Streamlit health endpoint at port 8512: **HTTP 200**.
- The Streamlit verification server was stopped after the health check.

## 7. Changed Files

| File | Change |
|---|---|
| `app.py` | Added bounded optimization form, session-state integration, baseline/optimized version selection, comparison UI, and SHAP cache identity. |
| `tests/test_optimization.py` | Added mismatch guard and dashboard optimization/cache contract coverage. |
| `README.md` | Documented bounded optimization, separate artifacts, session behavior, and limitations. |
| `QA_REPORT_MILESTONE_2_5_FINAL.md` | Added final implementation and validation evidence. |

The baseline reconciliation report and baseline artifact paths were not modified or overwritten.

## 8. Acceptance Notes

- User-triggered optimization was implemented; it does not run during ordinary Streamlit reruns.
- Optimized parameters are selected using training-fold CV F1 only.
- Held-out test metrics are retained for evaluation and are not used for parameter selection.
- The dashboard does not claim that metric changes establish fairness or causation.
- The app does not provide a dedicated fairness audit or individual employee ranking.

## 9. Final Verdict

**PASS** - Checkpoint 3 implementation is complete and the 62-test suite, compilation, lint, and Streamlit health gate all pass.
