# QA Report – Milestone 2.2: Advanced Workforce Analytics

**Project:** Employee Attrition Prediction & Responsible Workforce Analytics  
**Date:** 2026-11-27  
**Verdict:** **PASS** (100% of requirements met, 44/44 tests passing)  

---

## 1. Files Inspected

- `app.py`: Streamlit application incorporating sidebar filters, dataset overview, interactive analytics section with Plotly charts, model threshold explorer, permutation importance, and SHAP explainability.
- `src/attrition.py`: Preprocessing and machine learning training pipeline (`RandomForestClassifier`, `LogisticRegression`).
- `src/explainability.py`: SHAP explainability module supporting global and local explanations.
- `src/workforce_analytics.py`: Aggregate workforce analytics logic, filtering, small-sample safeguards, and aggregate CSV export.
- `src/analytics.py`: Dedicated analytical module providing group aggregations, statistical summaries, tenure groupings, and Plotly figure generators.
- `requirements.txt`: Python package specifications.
- `train.py`: Model training entrypoint and artifact generation.
- `tests/test_pipeline.py`, `tests/test_explainability.py`, `tests/test_workforce_analytics.py`, `tests/test_analytics.py`: Automated test suites.

---

## 2. Files Created or Modified

| File | Status | Description |
|---|---|---|
| `requirements.txt` | Modified | Added `plotly>=5.18` ensuring compatibility with Python 3.13 and existing dependencies. |
| `src/workforce_analytics.py` | Verified / Enhanced | Provides `apply_workforce_filters`, `department_attrition`, `salary_attrition`, `tenure_attrition`, `highest_attrition_group`, and `build_aggregate_export`. |
| `src/analytics.py` | Created | Modular analytics and Plotly chart generator functions for standalone and testable analytics computation. |
| `tests/test_analytics.py` | Created | 11 unit and integration tests covering department, salary, and tenure aggregations, tenure grouping, small-sample safeguards, dynamic filters, empty dataset handling, missing values, aggregate CSV export, and Plotly figure generation. |
| `tests/test_workforce_analytics.py` | Verified | 9 unit tests covering workforce analytics filtering, grouping, small-group safeguards, and privacy-safe exports. |
| `src/attrition.py` | Cleaned | Removed unused imports (`confusion_matrix`). |
| `tests/test_pipeline.py` | Cleaned | Removed unused imports (`calculate_metrics`). |
| `app.py` | Verified / Enhanced | Integrates the complete Interactive Workforce Analytics dashboard section with dynamic Plotly charts, sidebar filters, KPI cards, small-sample warnings, and CSV export. |
| `README.md` | Verified | Contains comprehensive documentation of analytics features, dependencies, and responsible use disclaimers. |
| `QA_REPORT_MILESTONE_2_2.md` | Created | This QA Report. |

---

## 3. Features Implemented

### A. Department Analysis
- Employee distribution and headcount share by department.
- Observed attrition count and attrition rate by department.
- Interactive dual Plotly bar and histogram charts.
- Subgroup sample-size indicator flagging departments with $N < \text{min\_group\_size}$ (default 30).

### B. Salary Analysis
- Distribution across `low`, `medium`, and `high` salary tiers.
- Stacked and grouped bar charts comparing stayed vs. left employees.
- Observed attrition rate comparison across compensation tiers.

### C. Tenure Analysis
- Grouped tenure into meaningful discrete ranges (`0-1 years`, `2 years`, `3 years`, `4 years`, `5 years`, `6-10 years`, `10+ years`, and `1-2 years`, `3-4 years`, `5-6 years`, `7+ years`).
- Combined headcount and attrition rate line/bar charts.
- Identification of tenure bands exhibiting higher historical turnover.

### D. Satisfaction and Performance
- Distribution histograms of employee job satisfaction scores.
- Interactive Plotly scatter plot examining satisfaction vs. last evaluation score, color-coded by quit status.
- Box plot of performance evaluation scores across retention outcomes.
- Quadrant breakdown (e.g., At-Risk High Performers vs. Engaged Contributors).
- Prominent disclaimers reinforcing that correlations do not imply causation.

### E. Interactive Filters
- Sidebar multiselects for Department, Salary category, Work accident history, and Recent promotion.
- Range slider for Years at company (tenure).
- Adjustable slider for Minimum Reliable Group Size ($N=5$ to $N=100$, default $30$).
- All analytics metrics, KPI badges, and Plotly charts update synchronously upon filter modification.
- Graceful empty-dataset handling displaying informative warnings without crashes.

### F. Dynamic Key Workforce Insights
- Overall filtered attrition rate and employee count.
- Department with highest reliable attrition rate.
- Salary category with highest reliable attrition rate.
- Tenure group with highest reliable attrition rate.
- Safeguard enforcement: Groups below the minimum sample threshold are labeled as "Not enough data" or flagged with sample size warnings.

### G. Privacy-Safe Aggregate Data Export
- Download button generating `workforce_analytics_aggregate.csv`.
- Exports category-level aggregate statistics only (dimension, category, headcount, left count, stayed count, attrition rate, reliability flag).
- Strictly excludes row-level data and personal identifiers.

### H. Responsible AI Notice
- Visible notice displayed at the top of the analytics section:
  > "This dashboard is an educational workforce analytics prototype. Observed historical patterns and model predictions do not establish causation or determine an individual's future behavior. Do not use these outputs to rank, penalize, hire, promote, or terminate employees. Any organizational interpretation requires appropriate privacy safeguards and human review."

---

## 4. Tests Executed & Actual Results

Executed the complete automated test suite using `pytest -v`:

```
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.2.0, pluggy-1.7.0 -- .venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\anany\OneDrive\Desktop\Project\employee_attrition_plus
plugins: anyio-4.9.0
collected 44 items

tests/test_analytics.py::test_tenure_grouping PASSED                     [  2%]
tests/test_analytics.py::test_department_attrition_calculations PASSED   [  4%]
tests/test_analytics.py::test_salary_attrition_calculations PASSED       [  6%]
tests/test_analytics.py::test_tenure_attrition_calculations PASSED       [  9%]
tests/test_analytics.py::test_small_sample_safeguards PASSED             [ 11%]
tests/test_analytics.py::test_filter_workforce_data PASSED               [ 13%]
tests/test_analytics.py::test_empty_dataset_handling PASSED              [ 15%]
tests/test_analytics.py::test_missing_values_handling PASSED             [ 18%]
tests/test_analytics.py::test_satisfaction_performance_summary PASSED    [ 20%]
tests/test_analytics.py::test_aggregate_export_and_csv_generation PASSED [ 22%]
tests/test_analytics.py::test_plotly_charts_generate_valid_figures PASSED [ 25%]
tests/test_explainability.py::test_clean_feature_names PASSED            [ 27%]
tests/test_explainability.py::test_build_explainer_supported_models PASSED [ 29%]
tests/test_explainability.py::test_explainer_unsupported_model_raises PASSED [ 31%]
tests/test_explainability.py::test_pipeline_validation_errors PASSED     [ 34%]
tests/test_explainability.py::test_compute_global_shap_both_models PASSED [ 36%]
tests/test_explainability.py::test_shap_feature_importance_and_impact PASSED [ 38%]
tests/test_explainability.py::test_single_prediction_explanation PASSED  [ 40%]
tests/test_explainability.py::test_positive_class_reconstruction_uses_all_features[Random Forest] PASSED [ 43%]
tests/test_explainability.py::test_positive_class_reconstruction_uses_all_features[Logistic Regression] PASSED [ 45%]
tests/test_explainability.py::test_categorical_names_align_with_reconstruction_values[Random Forest] PASSED [ 47%]
tests/test_explainability.py::test_categorical_names_align_with_reconstruction_values[Logistic Regression] PASSED [ 50%]
tests/test_explainability.py::test_global_reconstruction_matches_native_model_output[Random Forest] PASSED [ 52%]
tests/test_explainability.py::test_global_reconstruction_matches_native_model_output[Logistic Regression] PASSED [ 54%]
tests/test_explainability.py::test_invalid_input_handling PASSED         [ 56%]
tests/test_explainability.py::test_plots_generate_figures PASSED         [ 59%]
tests/test_explainability.py::test_reconstruction_property_random_forest PASSED [ 61%]
tests/test_explainability.py::test_reconstruction_property_logistic_regression PASSED [ 63%]
tests/test_explainability.py::test_global_batch_reconstruction_property PASSED [ 65%]
tests/test_pipeline.py::test_dataset_loads_and_target_is_binary PASSED   [ 68%]
tests/test_pipeline.py::test_duplicate_cleaning_reports_removed_rows PASSED [ 70%]
tests/test_pipeline.py::test_training_pipeline_returns_both_models PASSED [ 72%]
tests/test_pipeline.py::test_shap_explainability_for_supported_models PASSED [ 75%]
tests/test_pipeline.py::test_shap_feature_names_align_with_transformed_data PASSED [ 77%]
tests/test_pipeline.py::test_shap_rejects_empty_and_invalid_input PASSED [ 79%]
tests/test_workforce_analytics.py::test_department_attrition_calculation PASSED [ 81%]
tests/test_workforce_analytics.py::test_salary_attrition_calculation PASSED [ 84%]
tests/test_workforce_analytics.py::test_tenure_grouping_handles_ranges_and_missing_values PASSED [ 86%]
tests/test_workforce_analytics.py::test_filters_update_filtered_dataset PASSED [ 88%]
tests/test_workforce_analytics.py::test_empty_filtered_dataset PASSED    [ 90%]
tests/test_workforce_analytics.py::test_missing_categories_and_values_are_grouped_as_unknown PASSED [ 93%]
tests/test_workforce_analytics.py::test_small_group_safeguard_prevents_unreliable_ranking PASSED [ 95%]
tests/test_workforce_analytics.py::test_group_attrition_rejects_missing_target PASSED [ 97%]
tests/test_workforce_analytics.py::test_aggregate_export_has_no_employee_identifiers PASSED [100%]

================== 44 passed, 4 warnings in 95.50s (0:01:35) ==================
```

**Verification Highlights:**
- **Code Linting:** `python -m ruff check src/ app.py tests/` returned `All checks passed!`.
- **Syntax Compilation:** `python -m py_compile app.py` passed with 0 errors.
- **Server Health Probe:** Streamlit headless test returned `HTTP 200 OK` on `/_stcore/health`.
- **Existing ML & SHAP Models:** Intact and verified (18/18 tests in `test_explainability.py` and 6/6 tests in `test_pipeline.py` passed).

---

## 5. Manual Browser Verification Checklist

1. **Dashboard Launch:** Run `streamlit run app.py` in your terminal.
2. **Filter Interactivity:**
   - In the sidebar under **Workforce Analytics Filters**, select one or more departments (e.g., `sales`, `technical`).
   - Confirm that the **Filtered employees** metric updates instantly.
   - Confirm that charts in the **Department**, **Salary**, **Tenure**, and **Satisfaction & Performance** tabs refresh automatically.
3. **Small-Group Safeguard:**
   - Set the tenure slider to a very narrow slice (e.g., 7 to 8 years) or select a small department.
   - Verify that groups with $N < 30$ are marked with gray bars / small sample badges rather than falsely highlighted as high-confidence top rankings.
4. **Aggregate CSV Export:**
   - Click **Download filtered aggregate analytics CSV**.
   - Open the downloaded file and confirm it contains only aggregated statistics (`dimension`, `category`, `employee_count`, `left_count`, `stayed_count`, `attrition_rate`, `reliable`).
5. **Machine Learning & SHAP Coexistence:**
   - Scroll down to the **Model comparison**, **Evaluation and threshold explorer**, and **SHAP** sections.
   - Confirm that model metrics, confusion matrix, ROC/PR curves, and local/global SHAP waterfall plots function without degradation.

---

## 6. Known Limitations

1. **Aggregates vs. Causality:** Observed attrition rates across departments or salary tiers reflect historical correlations. They do not demonstrate that salary tier or department assignment causes an employee to quit.
2. **Small Subsets:** When multiple restrictive filters are combined simultaneously, subgroup sample sizes decrease; users should rely on the sample-size safeguard badges to avoid drawing conclusions from small sample cohorts.
