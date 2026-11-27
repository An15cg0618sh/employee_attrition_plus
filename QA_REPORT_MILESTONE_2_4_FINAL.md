# QA Report - Milestone 2.4 Final

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI

**Date:** 2026-11-27

**Overall Verdict:** **CONDITIONAL PASS**

Milestone 2.4 is implementation-complete and passes the complete automated regression suite. The aggregate CSV payload and filtered values were verified in the live browser, but Playwright did not emit a completed browser `download` event for the final link. Per the release rules, this prevents an unconditional PASS verdict.

## 1. Executive Summary

- Milestone 2.3 baseline: 49 automated tests passing.
- Milestone 2.4 baseline after focused additions: 53 automated tests passing.
- Ruff and Python compilation pass.
- Streamlit starts successfully and loads the full dashboard.
- Dataset, ML, SHAP, workforce analytics, and Responsible AI behavior remain intact.
- The live browser exposes a privacy-safe aggregate CSV payload with the required headers.
- The browser-native aggregate download link was added after the standard Streamlit download button failed to emit a Playwright download event.
- A completed Playwright download event remains NOT VERIFIED.

## 2. Initial Milestone 2.3 Baseline

The starting baseline was:

- 49 tests
- 0 failures
- 0 skips
- 0 warnings
- Ruff passed
- `py_compile app.py` passed
- Streamlit startup passed
- Raw rows: 11,582
- Duplicate rows removed: 9
- Deduplicated rows: 11,573
- Missing feature cells: 3
- Deduplicated attrition rate: approximately 17.2%
- One unresolved item: completed aggregate CSV download event

## 3. Files Inspected

- `app.py`
- `train.py`
- `src/attrition.py`
- `src/explainability.py`
- `src/workforce_analytics.py`
- `src/analytics.py`
- `requirements.txt`
- `README.md`
- `tests/test_pipeline.py`
- `tests/test_explainability.py`
- `tests/test_workforce_analytics.py`
- `tests/test_analytics.py`
- `data/employee_data.csv`
- `artifacts/best_model.joblib`
- `artifacts/training_report.json`
- Existing Milestone 2.1, 2.2, and 2.3 QA reports

## 4. Files Modified

- `app.py`
  - Replaced the standard aggregate CSV download control with a browser-native download link for the aggregate CSV payload.
  - Preserved the existing aggregate-only data and empty-result behavior.
- `src/workforce_analytics.py`
  - Added `aggregate_export_to_csv()` for explicit UTF-8 CSV serialization.
- `tests/test_pipeline.py`
  - Added a joblib save/load predict-probability round-trip test.
- `tests/test_workforce_analytics.py`
  - Added combined-filter export consistency coverage.
  - Added empty export header coverage.
  - Added aggregate CSV payload round-trip coverage.
- `README.md`
  - Expanded into release documentation covering setup, architecture, usage, analytics, export, and Responsible AI limitations.
- `QA_REPORT_MILESTONE_2_4_FINAL.md`
  - Added this final report.

No model was retrained or replaced. No existing dashboard section was removed.

## 5. Issues Discovered

1. The Milestone 2.3 report could not verify a completed CSV download event.
2. The standard Streamlit `st.download_button` emitted no Playwright download event in the provided browser environment.
3. The browser-native link also did not produce a Playwright `download` event, so Playwright/browser automation remains the limiting verification layer rather than evidence of a CSV payload defect.
4. The README lacked a complete release-oriented architecture, setup, and limitation description.

No individual employee export, individual prediction export, individual SHAP export, or consequential decision logic was discovered.

## 6. Root-Cause Analysis

The aggregate download data is calculated correctly in `src/workforce_analytics.py` and rendered from `build_aggregate_export()`. The unresolved browser issue was specific to browser automation observation of Streamlit's download control behavior:

- The standard button rendered and could be activated.
- Playwright did not emit or complete a browser download event.
- Direct browser inspection of the replacement browser-native link confirmed that the link's base64 payload contains the correct CSV.

This establishes payload correctness, but not the explicit Playwright completed-download event required for PASS.

## 7. Fixes Implemented

- Added `aggregate_export_to_csv()` as an explicit aggregate CSV serializer.
- Added a browser-native aggregate CSV data-URL link while preserving the aggregate-only payload and privacy caption.
- Added model-persistence and export round-trip tests.
- Expanded README release documentation.

## 8. Automated Test Results

### Final `python -m pytest -v`

- **53 passed**
- **0 failed**
- **0 skipped**
- **0 warnings**

The increase from 49 to 53 tests is explained by four added release tests:

1. Random Forest save/load predict-probability round trip.
2. Combined-filter aggregate export consistency.
3. Empty filtered export required-header behavior.
4. Aggregate CSV byte payload round trip.

### Final `python -m ruff check src/ app.py tests/`

- **All checks passed**

### Final `python -m py_compile app.py`

- **Exit code 0**

### Streamlit Startup

- Streamlit started successfully on port 8501.
- The app title was `Workforce Insights | Attrition Lab`.
- The dashboard rendered with the dataset overview, workforce analytics, model controls, and SHAP sections.

## 9. Dataset Consistency Verification

| Metric | Verified Value |
|---|---:|
| Raw records | 11,582 |
| Duplicate rows removed | 9 |
| Deduplicated records | 11,573 |
| Missing feature cells | 3 |
| Deduplicated attrition rate | 17.2038% |
| No-filter workforce records | 11,573 |
| Department export sum | 11,573 |
| Salary export sum | 11,573 |
| Tenure export sum | 11,573 |

The cleaned frame remains the shared source for Dataset Overview and Workforce Analytics. Exact duplicate removal and missing-feature reporting match the established baseline.

## 10. ML and SHAP Verification

Verified through source and automated tests:

- Random Forest and Logistic Regression train and produce test predictions.
- Model comparison uses cross-validation metrics.
- Threshold adjustment changes threshold-dependent metrics.
- Saved model round-trip reproduces identical predict-probability output.
- Global SHAP feature importance aligns with transformed feature names.
- Global SHAP reconstruction matches native model output.
- Individual SHAP probability-space reconstruction remains correct for Random Forest.
- Individual SHAP log-odds/margin reconstruction remains correct for Logistic Regression.
- Positive-class SHAP extraction is preserved.

Browser rendering additionally verified:

- Both model selectors are available.
- A Logistic Regression selection rerendered metrics.
- Decision threshold updates were visible.
- All three global SHAP tabs were available.
- Local waterfall heading rendered.
- Exact Mathematical Reconstruction opened and displayed the baseline, sum, reconstructed output, and formatted probability.

## 11. Workforce Analytics Verification

Verified:

- Department tab
- Salary tab
- Tenure tab
- Satisfaction & Performance tab
- Department filter
- Salary filter
- Years-at-company range handles
- Work-accident filter
- Promotion filter
- Minimum reliable group size
- Combined-filter behavior
- Empty-data export headers
- Small-group safeguards

Browser-exercised combined filters produced a filtered CSV payload of 952 department/salary records for the sales/low/work-accident/promotion/tenure-filtered state. The payload's department, salary, and tenure counts were consistent with the filtered export state.

## 12. Browser Verification

The live Streamlit app loaded in the integrated browser.

Browser checks that passed:

- Dashboard load
- Department, salary, accident, promotion, tenure, and group-size controls
- All four Workforce Analytics tabs
- Random Forest and Logistic Regression selection
- Threshold slider
- Three global SHAP tabs
- Local SHAP waterfall section
- Exact Mathematical Reconstruction expander
- Privacy-safe aggregate link

Browser limitation:

- The standard Streamlit download button and replacement browser-native link did not emit a completed Playwright `download` event.
- Therefore an actual browser completion event is NOT VERIFIED.

The browser link payload itself was decoded and inspected successfully.

## 13. Actual CSV Download Verification

The browser-native link exposed this exact header row:

```text
dimension,category,employee_count,left_count,stayed_count,attrition_rate,reliable
```

The decoded filtered payload contained:

- Department aggregate: `sales`, 952 employees
- Salary aggregate: `low`, 952 employees
- Tenure aggregates for 3, 4, 5, and 6-10 years
- Required counts and attrition rates
- No individual employee names
- No employee IDs
- No individual predictions
- No individual SHAP values
- No row-level employee records

Status:

- **Payload: PASS**
- **Headers: PASS**
- **Privacy: PASS**
- **Filtered consistency: PASS**
- **Completed browser download event: NOT VERIFIED**

## 14. Privacy and Responsible AI Audit

Preserved:

- Aggregate-only export
- No individual ranking
- No automated employment decisions
- Small-group reliability safeguards
- Correlation/non-causation notices
- SHAP model-behaviour notices
- Historical-bias and privacy notices
- Human-review/educational-use limitations

No individual-level export or consequence-action implementation was found.

## 15. Code Quality and Architecture

- `src/attrition.py` is the model, validation, cleaning, metric, and persistence layer.
- `src/explainability.py` is the SHAP layer.
- `src/workforce_analytics.py` is the dashboard's active aggregate/filter/export implementation.
- `src/analytics.py` remains a separate modular analytics implementation with independent tests. It was not deleted because it serves as tested Milestone 2.1/2.2 functionality and is not part of a confirmed defect or circular dependency.
- No circular dependency was identified.
- No unused dependency was added.
- Empty filtered datasets are handled without export and no crash.

## 16. Remaining Limitations

1. Completed Playwright browser download event remains unverified.
2. The browser environment may require manual confirmation of the downloaded filename and saved file.
3. Browser data-URL download behavior can vary across browsers and enterprise security settings.
4. The app remains an educational prototype and is not production HR decision software.

## 17. Final Milestone Verdict

**CONDITIONAL PASS**

Milestone 2.4 passes all 53 automated tests, lint, compile, application startup, live browser UI, filtered payload, and Responsible AI safeguards. The only required item that remains incomplete is automatic capture of a completed aggregate CSV download event. A manual browser download is still recommended as the final sign-off step.
