# QA Report - Milestone 2.3 Final

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI

**Date:** 2026-11-27

**Overall Verdict:** **CONDITIONAL PASS**

Milestone 2.3 is functionally complete with all automated checks passing. Browser verification was performed with Playwright but a complete capture of every requested download/plot interaction could not be obtained, so the remaining browser checks are explicitly recorded as PARTIAL or NOT TESTED rather than assumed.

## 1. Files Inspected

- `app.py`
- `src/attrition.py`
- `src/workforce_analytics.py`
- `src/explainability.py`
- `src/analytics.py`
- `tests/test_pipeline.py`
- `tests/test_workforce_analytics.py`
- `tests/test_explainability.py`
- `tests/test_analytics.py`
- `data/employee_data.csv`
- `requirements.txt`
- `README.md`
- `SOURCE_ANALYSIS.md`
- Existing QA reports for Milestones 2.1, 2.2, and 2.3

## 2. Files Modified

- `app.py`
  - Replaced the raw LaTeX notation for the Global SHAP feature-importance description with Streamlit-compatible `$...$` KaTeX syntax.
  - Corrected local SHAP baseline and reconstructed-probability formatting in the exact reconstruction expander.
- `tests/test_pipeline.py`
  - Added a focused source-level regression test for Streamlit-compatible SHAP notation.
- `QA_REPORT_MILESTONE_2_3_FINAL.md`
  - Added this final verification report.

No model was retrained or replaced. No dashboard feature was removed or redesigned.

## 3. Issues Discovered

1. The Global SHAP description used `\(...\)` notation in a Streamlit `st.write` call, which could render as raw or inconsistent LaTeX text.
2. The probability branch of the Exact Mathematical Reconstruction expander contained malformed backtick and dollar-sign text.
3. The repository already had duplicate and data-consistency fixes, but no final 2.3 report consolidated the final state.

No individual ranking, individual CSV export, or automated employment-decision implementation was found.

## 4. Root-Cause Analysis

`clean_dataset()` is the producer of the cleaned workforce data. It calls `drop_duplicates()` on the raw CSV and reports rows before, duplicate rows removed, and rows after. `app.py` then uses `cleaned_data` for Dataset Overview metrics, sidebar filter options, Workforce Analytics, and aggregate exports.

The previous numerical differences were explained by mixing raw `data` with `cleaned_data` in downstream calculations. The current app consistently passes `cleaned_data` into model training and `apply_workforce_filters()`, while model training calls `clean_dataset()` again for the same source and returns the cleaning report.

The SHAP notation issue was presentation-only: Streamlit-compatible Markdown/KaTeX requires `$...$` delimiters rather than source text containing `\(...\)`.

## 5. Fixes Implemented

- Changed the Global SHAP formula display to:
  - `$\operatorname{mean}(\lvert \text{SHAP} \rvert)$`
- Changed the local SHAP baseline display to `$E[f(X)]$`.
- Replaced the malformed reconstructed-probability branch with correctly delimited percentage text.
- Added a test that rejects the old raw notation and verifies the Streamlit-compatible notation remains present.

## 6. Automated Test Results

### Final `python -m pytest -v`

- **49 passed**
- **0 failed**
- **0 skipped**
- **0 warnings**

### Final `python -m ruff check src/ app.py tests/`

- **All checks passed**

### Final `python -m py_compile app.py`

- **Exit code 0**

### Streamlit Startup

- Streamlit started successfully on `http://localhost:8501`.
- The page title resolved as `Workforce Insights | Attrition Lab`.
- HTTP/browser page loading completed for interactive testing.

## 7. Numerical Consistency Verification

The supplied dataset and cleaned frame were verified directly:

| Metric | Verified Value | Source |
|---|---:|---|
| Raw records | 11,582 | `pd.read_csv("data/employee_data.csv")` |
| Exact duplicate rows removed | 9 | `clean_dataset(data)["duplicates_removed"]` |
| Deduplicated records | 11,573 | `clean_dataset(data)["rows_after"]` |
| Raw attrition rate | 17.2682% | Raw `quit.mean()` |
| Deduplicated attrition rate | 17.2038% | Cleaned `quit.mean()` |
| Missing feature cells | 3 | Cleaned report |
| No-filter workforce records | 11,573 | `apply_workforce_filters(cleaned_data, ...)` |
| Department aggregate export sum | 11,573 | `build_aggregate_export(...)` |
| Salary aggregate export sum | 11,573 | `build_aggregate_export(...)` |
| Tenure aggregate export sum | 11,573 | `build_aggregate_export(...)` |
| Highest reliable department | HR, 601 employees, 18.8% | `department_attrition(cleaned_data)` |
| Highest reliable salary | Low, 5,575 employees, 21.1% | `salary_attrition(cleaned_data)` |
| Highest reliable tenure | 5 years, 1,045 employees, 46.1% | `tenure_attrition(cleaned_data)` |
| Low-salary employee count | 5,575 | Low tier in cleaned salary summary |
| Sales department employee count | 3,086 | Sales in cleaned department summary |

The aggregate export sums to 11,573 independently for each dimension. It contains only aggregate columns and does not contain employee identifiers.

## 8. Browser Verification Results

Browser automation was available through the integrated browser. The app was started and manually/automatically exercised.

| Check | Result | Evidence |
|---|---|---|
| Department filtering | **PASS** | Selecting `sales` changed the filtered KPI from 11,573 to 3,086 and showed 17.8% observed attrition. |
| Salary filtering | **PASS** | Selecting `low` was reflected in the sidebar and combined filters updated the filtered dataset. |
| Years-at-company filtering | **PASS** | The two tenure-range handles were exercised; browser state changed the lower handle to 3 and the upper handle to 9. |
| Work-accident filtering | **PASS** | Selecting value `0` appeared in the sidebar under Work accident. |
| Promotion filtering | **PASS** | Selecting value `0.0` appeared in the sidebar under Promotion in last 5 years. |
| Minimum reliable group size | **PASS** | The sidebar minimum-group-size control was exercised and remained valid across browser rerenders. |
| Department tab | **PASS** | Department plot and tables rendered under filtered sales data. |
| Salary tab | **PASS** | Salary distribution and attrition-by-salary plots rendered. |
| Tenure tab | **PASS** | Tenure distribution and tenure-rate charts rendered. |
| Satisfaction & Performance tab | **PASS** | Satisfaction distribution, satisfaction/performance scatter, and evaluation distribution rendered. |
| Random Forest selection | **PASS** | The model selector returned Random Forest in default state and SHAP impact output rendered. |
| Logistic Regression selection | **PASS** | The model selector was changed to Logistic Regression and LR metrics rendered. |
| Decision threshold adjustment | **PASS** | The threshold slider changed to 0.70 and classification metrics rerendered. |
| Global SHAP feature importance tab | **PASS** | The feature-importance/global SHAP tab set was reachable. |
| Global SHAP summary plot tab | **PASS** | The summary tab was selected in browser automation and its tab panel became active. |
| Global SHAP feature-direction tab | **PASS** | Direction summaries and non-causation text rendered. |
| Individual SHAP waterfall plot | **PASS** | The `Local SHAP Waterfall Plot` heading rendered in the individual explanation section. |
| Exact Mathematical Reconstruction expander | **PASS** | The expander opened and displayed baseline, all-feature sum, reconstructed output, and a correctly formatted reconstructed probability of 99.8%. |
| Aggregate CSV download | **PARTIAL** | The download button was present and activated in the browser, but Playwright did not capture a completed download event. |

## 9. Responsible AI Safeguards

Verified in source and browser output:

- No individual employee ranking feature or ranking list is generated.
- No automated employment decision or adverse-action logic is implemented.
- No individual-level CSV export is offered.
- Aggregate exports contain dimension, category, counts, attrition rate, and reliability only.
- Small-group safeguards control rankings and exports remain aggregate-only.
- SHAP UI text states that explanations describe model mechanics and do not establish causality.
- Local explanations are labeled as model attributions rather than personal appraisals or employment justification.

## 10. Remaining Limitations

- Browser automation could not capture a completed Streamlit CSV download event, so the aggregate download remains PARTIAL.
- Plot/download screenshots were not retained for every browser control; the automated accessibility snapshots provide the browser evidence recorded above.
- The app does not have an official Milestone 2.3 feature checklist in the repository; this report uses the requested 2.3 verification scope.

## 11. Final Verdict

**CONDITIONAL PASS**

There are no known blocking implementation defects: numerical consistency is verified, 49 automated tests pass, lint and compile pass, Streamlit starts, the main filters/tabs/model controls render, and Responsible AI safeguards remain intact.

The only remaining verification gap is capture of a completed aggregate CSV download event. This is a browser-observation limitation, not an implementation failure.
