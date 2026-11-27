# QA Report – Milestone 2.3: Verification & Deployment Readiness Audit

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI  
**Date:** 2026-11-27  
**Overall Verdict:** **BLOCKED** *(Formal Milestone 2.3 acceptance remains blocked because no official Milestone 2.3 specification exists in the repository. On the preliminary project-health scope, all 48 automated tests, lint/compile checks, deep numerical/SHAP audits, and the user-supplied live UI output pass, with interactive browser clicks/downloads still `PARTIAL`.)*

---

## A. Executive Summary

- **Overall Verdict:** **BLOCKED** (No official Milestone 2.3 checklist/specification exists in the repository.)
- **Preliminary Project-Health Baseline (with User-Supplied Live UI Output):**
  - **Total acceptance criteria (Preliminary Health Scope + M23-00):** 14
  - **Requirements `PASS`:** 12
  - **Requirements `FAIL`:** 0
  - **Requirements `PARTIAL`:** 1 (`H-13`: Live UI default view verified from user-supplied app output; interactive clicks/CSV upload/download not yet captured)
  - **Requirements `NOT VERIFIED`:** 0
  - **Requirements `BLOCKED`:** 1 (`M23-00`: Official Milestone 2.3 specification not present in repository)

---

## B. Milestone 2.3 Acceptance Checklist

| ID | Requirement | Implementation Status | Test Status | Evidence | Issues |
|---|---|---|---|---|---|
| M23-00 | Official Milestone 2.3 feature specification & acceptance criteria | **BLOCKED** | **BLOCKED** | Inspected `README.md`, `SOURCE_ANALYSIS.md`, `LICENSE_NOTE.txt`, `src/`, `tests/` — no Milestone 2.3 spec exists. | Official checklist must be provided/confirmed by project owner. |
| H-01 | Application startup & health endpoint | **PASS** | **PASS** | `py_compile app.py` (exit 0); `streamlit run app.py` live UI output supplied by user. | None |
| H-02 | CSV upload & schema validation | **PASS** | **PASS** | `validate_dataset_frame()` in `src/attrition.py` & `app.py`; `test_validate_dataset_frame_rejects_invalid_schemas` & `test_training_rejects_tiny_dataset` passed. | None |
| H-03 | Dataset integrity & deduplication | **PASS** | **PASS** | Live UI `Data quality details`: `rows_before: 11582`, `duplicates_removed: 9`, `rows_after: 11573`, `missing_feature_cells: 3`. | None |
| H-04 | Dashboard data consistency | **PASS** | **PASS** | Live UI `Dataset overview` and `Workforce Analytics` both show `11,573` records and `17.2%` observed attrition rate. | None |
| H-05 | ML training, CV, metrics & thresholds | **PASS** | **PASS** | Live UI `Random Forest` metrics: `Accuracy 0.984`, `Precision 0.987`, `Recall 0.920`, `F1 0.952`, `ROC-AUC 0.987`, `PR-AUC 0.973`; all ML tests passed. | None |
| H-06 | Model persistence (`save_model` / `load_model`) | **PASS** | **PASS** | Joblib roundtrip verified identical `predict_proba` outputs on `Random Forest`. | None |
| H-07 | Global SHAP explainability & alignment | **PASS** | **PASS** | Live UI renders `📊 Feature Importance`, `🌈 SHAP Summary Plot`, `🧭 Feature Impact Direction`; 18/18 SHAP tests passed. | Minor: `\text{mean}(|\text{SHAP}|)` rendered as raw LaTeX text in `app.py`. |
| H-08 | Individual SHAP reconstruction | **PASS** | **PASS** | Live UI Record #0: `Left (1)`, `99.8%`, Baseline `0.500`, top 4 pos (`+0.2421`, `+0.1788`, `+0.0783`, `+0.0061`), top 4 neg (`-0.0103`, `-0.0025`, `-0.0014`, `-0.0006`); math diff `9.44e-15` (RF) / `1.11e-16` (LR). | None |
| H-09 | Workforce analytics & sidebar filters | **PASS** | **PASS** | Live UI KPIs: `11,573` filtered, `17.2%` rate, `hr (601)` `18.8%`, `low (5575)` `21.1%`, `5 years (1045)`. 21 analytics tests passed. | `src/analytics.py` is an unused parallel module alongside `src/workforce_analytics.py`. |
| H-10 | Small-group reliability safeguards | **PASS** | **PASS** | Live UI caption: `Rankings require at least 30 records in a group`; boundary tests (`29/30/31`) passed. | None |
| H-11 | Privacy-safe aggregate CSV export | **PASS** | **PASS** | `build_aggregate_export()` exports only aggregate columns; live UI states `The export excludes individual employee records and personal identifiers.` | None |
| H-12 | Responsible AI & privacy governance | **PASS** | **PASS** | Live UI displays top banner, Workforce Analytics notice, Global/Local SHAP reminders, and full `Responsible-use notes, SHAP governance & limitations` expander. | None |
| H-13 | Live browser UI rendering & interactions | **PARTIAL** | **PARTIAL** | Default live UI output confirmed all metrics, labels, sections, and expanders. Interactive clicks (tabs 2–4, sidebar filter changes, CSV upload/download, LR toggle) not yet captured. | Automated Chrome DevTools blocked (`DevToolsActivePort` missing). |

---

## C. Files Inspected

- `app.py`, `train.py`, `requirements.txt`, `README.md`, `SOURCE_ANALYSIS.md`, `data/employee_data.csv`
- `src/attrition.py`, `src/explainability.py`, `src/workforce_analytics.py`, `src/analytics.py`
- `tests/test_pipeline.py`, `tests/test_explainability.py`, `tests/test_workforce_analytics.py`, `tests/test_analytics.py`

---

## D. Files Modified

No source, test, dataset, or configuration files were modified (`QA_REPORT_MILESTONE_2_3.md` updated with user-supplied live UI evidence).

---

## E. Automated Test Results

- **Compilation:** `python -m py_compile app.py` → Exit code `0`
- **Linting:** `python -m ruff check src/ app.py tests/` → `All checks passed!`
- **Test Suite:** `.venv\Scripts\python.exe -m pytest -v` → **48 passed in 31.69s, 0 failed, 0 skipped, 0 warnings**
  - `tests/test_analytics.py`: 11 passed
  - `tests/test_explainability.py`: 18 passed
  - `tests/test_pipeline.py`: 9 passed
  - `tests/test_workforce_analytics.py`: 10 passed

---

## F. Functional Verification (Live UI Output vs. Backend Audit)

Every numerical value in the user's live UI output matches the verified backend calculations:
- **Uploaded / Deduplicated / Duplicates / Missing:** `11,582` / `11,573` / `9` / `3`
- **Observed Attrition Rate (Overview & Analytics):** `17.2%`
- **Highest Reliable Department:** `hr` (`601 employees`, `18.8%`)
- **Highest Reliable Salary:** `low` (`5575 employees`, `21.1%`)
- **Highest Reliable Tenure:** `5 years` (`1045 employees`)
- **Random Forest Metrics:** Accuracy `0.984`, Precision `0.987`, Recall `0.920`, F1 `0.952`, ROC-AUC `0.987`, PR-AUC `0.973`
- **Local SHAP (Sample 0, RF):** `Left (1)`, `99.8%`, Baseline `0.500`, top 4 positive (`+0.2421`, `+0.1788`, `+0.0783`, `+0.0061`), top 4 negative (`-0.0103`, `-0.0025`, `-0.0014`, `-0.0006`)

---

## G. Browser Verification

- **Verified from User-Supplied Live UI Text:** Full default-load page structure, all KPI values, ML metrics, SHAP metrics/contributors, JSON data-quality report, and Responsible AI notices.
- **Not Yet Captured Interactively:** Switching to `Logistic Regression`, moving the threshold slider, filtering by sidebar `Department`/`Salary`/`Years at company`, clicking `Salary`/`Tenure`/`Satisfaction & Performance` tabs, uploading a test CSV, and downloading `workforce_analytics_aggregate.csv`.

---

## H. Regression Testing

All Milestone 2.1 and Milestone 2.2 automated tests passed (`48/48`, 0 warnings, 0 regressions vs. the 44-test baseline).

---

## I. Bugs and Outstanding Issues

1. **[BLOCKING FOR M2.3 SIGN-OFF] Missing Official Milestone 2.3 Checklist:** No Milestone 2.3 specification exists in the repository.
2. **[LOW / UI POLISH] Raw LaTeX Syntax in `app.py`:** In `app.py`, `(\text{mean}(|\text{SHAP}|))` uses `\(...\)` instead of `$...$`, causing Streamlit to render raw LaTeX text (`(\text{mean}(|\text{SHAP}|))`) in the Global SHAP heading description.
3. **[LOW / VERIFICATION GAP] Interactive Browser Actions:** Tab clicks, sidebar filter updates, LR toggle, and CSV download remain `PARTIAL` until manually exercised.
4. **[LOW / ARCHITECTURAL HYGIENE] Unused Parallel Module:** `src/analytics.py` (and `tests/test_analytics.py`) and `.venv-new2` remain unused alongside `src/workforce_analytics.py` and `.venv`.

---

## J. Security and Responsible AI Review

- Verified in live UI output: top educational warning, Workforce Analytics non-causation notice, Global/Local SHAP reminders, privacy-safe export caption, and all 6 Responsible AI & Explainability Principles.
- Zero PII, credentials, or individual employee rankings exposed.

---

## K. Remaining Manual Checks

1. Confirm or provide the official Milestone 2.3 acceptance checklist.
2. In your open browser tab, quickly exercise the interactive controls:
   - Select `sales` in the sidebar `Department` filter and verify `Filtered employees` updates from `11,573` to `3,086`.
   - Click `Salary`, `Tenure`, and `Satisfaction & Performance` tabs to confirm Plotly charts render.
   - Click `Download filtered aggregate analytics CSV` and inspect the CSV columns.
   - Switch `Choose a model` to `Logistic Regression` and adjust the threshold slider.

---

## L. Final Milestone Verdict

**FINAL MILESTONE VERDICT: BLOCKED**

Your live UI output confirms that the dashboard's numbers, ML metrics, SHAP values, and Responsible AI notices are 100% consistent with the backend (`48/48` tests passing). To reach a formal **PASS**, please confirm the official Milestone 2.3 scope and verify the remaining interactive browser controls in Section K (and optionally let me fix the minor `$...$` LaTeX formatting in `app.py`).
