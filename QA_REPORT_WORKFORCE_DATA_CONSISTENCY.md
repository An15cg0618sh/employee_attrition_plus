# QA Report – Workforce Analytics Data Consistency

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI  
**Date:** 2026-11-27  
**Status:** Verified & Passed (`45/45` tests passing, `0` warnings, `0` lint errors)

---

## 1. Root Cause

1. **Deduplicated vs. Raw Dataset Flow:**
   - The supplied CSV (`data/employee_data.csv`) contains **11,582** total rows, of which **9** are exact duplicates, yielding **11,573** unique records after `clean_dataset(data)`.
   - In `app.py`, `clean_dataset(data)` is executed at line 73 to produce `cleaned_data` (11,573 rows), and `apply_workforce_filters(cleaned_data, ...)` feeds `analytics_data` (11,573 rows when no filters are selected).
   - **Why 11,582 was previously observed:** Prior to `cleaned_data` being wired into `app.py`, `apply_workforce_filters` and the `Dataset overview` attrition rate/distribution chart referenced raw `data` (11,582 rows, `17.3%` attrition rate) rather than `cleaned_data` (11,573 rows, `17.2%` attrition rate).
   - **Current state in `app.py`:** `cleaned_data` is now consistently used across `Dataset overview` (observed attrition rate `17.2%` and target distribution), `Workforce Analytics` (`11,573` unfiltered records, `17.2%` attrition rate), all Plotly charts, and the aggregate CSV export (`build_aggregate_export`).
2. **Tenure & Salary Group Ordering:**
   - `tenure_attrition` and `salary_attrition` in `src/workforce_analytics.py` previously inherited `group_attrition`'s default sort (`["reliable", "attrition_rate"]`), causing tenure buckets to appear out of chronological order (`5 years`, `4 years`, `3 years`, `6-10 years`, `2 years`) in charts and tables.

---

## 2. Files Inspected and Changed

| File | Action | Details |
|---|---|---|
| `app.py` | Inspected & Verified | Confirmed `cleaned_data` (11,573 rows) is the shared source of truth for `Dataset overview` metrics/distribution, sidebar filter options, `analytics_data`, Plotly charts, and CSV export. |
| `src/attrition.py` | Inspected | Confirmed `clean_dataset` removes exact duplicates (`11,582 -> 11,573`) and `train_and_compare` trains/evaluates on `cleaned`. |
| `src/workforce_analytics.py` | Inspected & Modified | Verified `TENURE_BINS` (`[-inf, 1, 2, 3, 4, 5, 10, inf]`, `right=True, include_lowest=True`) are mutually exclusive and exhaustive. Added `_sort_by_order` so `tenure_attrition` sorts chronologically (`0-1 years` through `10+ years`, then `Unknown`) and `salary_attrition` sorts by tier (`low`, `medium`, `high`, `Unknown`). |
| `tests/test_workforce_analytics.py` | Modified | Added `test_deduplicated_source_of_truth_and_no_filter_counts`, updated `test_tenure_grouping_handles_ranges_and_missing_values` to assert chronological ordering, updated `test_filters_update_filtered_dataset` to test on `cleaned`, and verified aggregate CSV export sums to `11,573` per dimension with no employee identifiers. |
| `QA_REPORT_WORKFORCE_DATA_CONSISTENCY.md` | Created | This audit report. |

---

## 3. Fix Implemented

1. **Shared Source of Truth (`cleaned_data`):**
   - Verified `app.py` deduplicates `data` via `cleaned_data, cleaning_report = clean_dataset(data)` prior to building sidebar filter options and calling `apply_workforce_filters(cleaned_data, ...)`.
   - Verified `Dataset overview` (`attrition_rate` and `Target distribution`) and `Workforce Analytics` (`Filtered employees`, group summaries, Plotly charts, and `workforce_analytics_aggregate.csv`) all compute from `cleaned_data` (`11,573` rows).
2. **Explicit Chronological / Hierarchical Ordering (`src/workforce_analytics.py`):**
   - Added `_sort_by_order()` helper used by `tenure_attrition` (`TENURE_LABELS + ["Unknown"]`) and `salary_attrition` (`["low", "medium", "high", "Unknown"]`).
   - `highest_attrition_group()` continues to sort reliable groups by `attrition_rate` descending internally, so top-risk KPI insights remain unaffected.

---

## 4. Test Results

Executed after the final edit:

- **Lint check:** `python -m ruff check src/ app.py tests/` → `All checks passed!`
- **Compilation check:** `python -m py_compile app.py src/attrition.py src/explainability.py src/workforce_analytics.py src/analytics.py` → exit code `0`
- **Full pytest suite:** `.venv\Scripts\python.exe -m pytest -v` → **45 passed in 49.90s, 0 warnings**
  - `tests/test_analytics.py`: 11 passed
  - `tests/test_explainability.py`: 18 passed
  - `tests/test_pipeline.py`: 6 passed
  - `tests/test_workforce_analytics.py`: 10 passed

---

## 5. Warning Classifications

In the preceding test run (`44 passed, 4 warnings`), 4 `PytestUnraisableExceptionWarning` messages were observed in `tests/test_explainability.py::test_reconstruction_property_random_forest` and `tests/test_pipeline.py::test_shap_rejects_empty_and_invalid_input`:
- `<function Variable.__del__>` → `RuntimeError: main thread is not in main loop` (2 occurrences)
- `<function Image.__del__>` → `RuntimeError: main thread is not in main loop` (2 occurrences)

**Classification:** **Harmless**
- **Cause:** On Windows, `matplotlib.pyplot` defaults to the interactive `TkAgg` backend when `MPLBACKEND` is unset. Figures created in earlier plot tests leave Tkinter `Variable`/`Image` objects that are later finalized by Python's garbage collector on a background thread (e.g., `loky`/joblib worker thread), triggering Tkinter's thread-affinity check during object destruction.
- **Impact:** Zero impact on numerical correctness, model outputs, SHAP values, or Streamlit runtime (Streamlit sets the non-interactive `Agg` backend automatically). In the latest `45/45` run, 0 warnings were emitted.

---

## 6. Final Dataset Counts

| Metric | Value | Source |
|---|---|---|
| Uploaded records (`data`) | **11,582** | `data/employee_data.csv` |
| Exact duplicate rows removed | **9** | `clean_dataset(data)` |
| Records after deduplication (`cleaned_data`) | **11,573** | `clean_dataset(data)` |
| Missing feature cells after deduplication | **3** | `clean_dataset(data)` |
| Observed attrition rate (deduplicated) | **17.2%** (`0.172038`) | `cleaned_data["quit"].mean()` |
| Workforce Analytics – Filtered employees (no filters) | **11,573** | `apply_workforce_filters(cleaned_data, ...)` |
| Aggregate CSV export sum per dimension (`department`, `salary`, `tenure`) | **11,573** | `build_aggregate_export(...)` |

---

## 7. Remaining Limitations

1. **Deduplication Scope:** `clean_dataset` removes *exact* duplicate rows across all columns (`drop_duplicates()`). Distinct employees who happen to share identical feature values and `quit` status cannot be distinguished in an anonymized dataset without a unique employee ID, so exact-row deduplication treats them as duplicate records.
2. **Tenure Range Filter vs. Missing Tenure:** When the sidebar `Years at company` slider is active, `apply_workforce_filters` uses `tenure.between(min, max)`, which excludes rows with missing/NaN `time_spend_company` if any are present in an uploaded CSV (the supplied `data/employee_data.csv` has 0 missing values in `time_spend_company`).
3. **Browser Cache / Running Server:** If an older Streamlit server process was still running in a browser tab from before the `cleaned_data` update, restarting `streamlit run app.py` and refreshing the browser is required to see `11,573`.
