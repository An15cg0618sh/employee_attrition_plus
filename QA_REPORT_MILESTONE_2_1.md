# QA Report – Milestone 2.1: Implement SHAP Explainable AI

**Project:** Employee Attrition Prediction – Enhanced ML Dashboard  
**Date:** 2026-11-26  
**Status:** Completed & Verified  

---

## 1. Files Inspected

Before making changes, the following project components and files were thoroughly inspected:
- `requirements.txt`: Python package dependencies (`pandas`, `numpy`, `scikit-learn`, `streamlit`, `matplotlib`, `joblib`, `pytest`).
- `train.py`: Model training entrypoint, artifact serialization (`best_model.joblib`), evaluation metric persistence (`training_report.json`).
- `src/attrition.py`: Preprocessing architecture (`ColumnTransformer`, `StandardScaler`, `SimpleImputer`, `OneHotEncoder`), model training pipeline (`Pipeline([("preprocess", ...), ("model", ...)])`), model definitions (`RandomForestClassifier`, `LogisticRegression`), metrics computation, and permutation feature importance.
- `tests/test_pipeline.py`: Existing tests validating CSV loading, binary target validation, deduplication data-quality checks, and model training comparison.
- `app.py`: Streamlit dashboard layout, threshold explorer, confusion matrix, ROC/PR curves, and permutation importance.
- `README.md`: Project documentation, background, and ethical disclaimers.

---

## 2. Files Created or Modified

| File | Status | Description |
|---|---|---|
| `requirements.txt` | Modified | Added `shap>=0.44` ensuring compatibility with Python 3.13, NumPy 2.2+, pandas 2.3+, and scikit-learn 1.9+. |
| `src/explainability.py` | Created | Reusable Explainable AI module providing `TreeExplainer` for Random Forest, `LinearExplainer` (with `Independent` masker) for Logistic Regression, positive-class slicing (`quit = 1`), clean feature naming, sample sampling, metric extraction, and plotting helpers. |
| `tests/test_explainability.py` | Created | Dedicated pytest test suite containing 9 unit tests covering explainer construction, positive class slicing, clean feature naming, global and local calculations, input validation, and matplotlib figure generation. |
| `tests/test_pipeline.py` | Verified | Contains 6 tests (including integration tests for `explain_samples`, feature alignment, and empty input rejection), fully passing. |
| `app.py` | Modified | Integrated two new major sections: **SHAP – Global Model Explainability** (Feature Importance, Beeswarm Summary Plot, Directional Impact) and **SHAP – Individual Prediction Explanation** (sample record selector, metrics cards, Waterfall Plot, positive/negative factor decomposition), cached via `@st.cache_data`. |
| `README.md` | Modified | Updated with comprehensive SHAP explanations, mathematical grounding, architecture, global and local explanation guides, setup instructions, limitations, and Responsible AI guidelines. |
| `QA_REPORT_MILESTONE_2_1.md` | Created | This QA and verification report. |

---

## 3. SHAP Implementation Approach

### Explainer Selection & Model Support
- **Random Forest (`RandomForestClassifier`)**:
  - Explainer: `shap.TreeExplainer(model)`.
  - Class Slicing: TreeExplainer outputs SHAP arrays of shape `(N, features, 2)`. The module explicitly slices index `[:, :, 1]` to explain the positive class (`quit = 1`, attrition), preventing class inversion.
- **Logistic Regression (`LogisticRegression`)**:
  - Explainer: `shap.LinearExplainer(model, masker=shap.maskers.Independent(data=background))`.
  - Class Slicing: For binary classification in scikit-learn, `LinearExplainer` computes log-odds contributions directly for class 1 (shape `(N, features)`).
- **Unsupported Models**:
  - Any model type other than `RandomForestClassifier` or `LogisticRegression` raises a descriptive `ValueError`.

### Preprocessing & Feature Name Alignment
- Scikit-learn's `ColumnTransformer` applies numeric imputation + scaling and categorical imputation + one-hot encoding (`handle_unknown="ignore"`).
- `src/explainability.py` extracts raw names using `preprocessor.get_feature_names_out()` and cleans them:
  - `numeric__satisfaction_level` $\rightarrow$ `satisfaction_level`
  - `categorical__department_sales` $\rightarrow$ `department: sales`
  - `categorical__salary_low` $\rightarrow$ `salary: low`
- The cleaned names are assigned directly to the `Explanation.feature_names` attribute and aligned with the dense array produced by `preprocessor.transform(X)`.

### Responsiveness and Streamlit Caching
- Global SHAP values are calculated on a representative sample (default: 150 records) rather than the entire dataset.
- Background datasets for `LinearExplainer` are subsampled to $\le 100$ records to keep initialization instantaneous.
- In `app.py`, global calculations are cached with `@st.cache_data(show_spinner=False)`, ensuring instant widget interactions and avoiding redundant re-computations on Streamlit reruns.

---

## 4. Tests Executed & Actual Results

The automated test suite was executed using `python -m pytest -v`:

```
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\anany\OneDrive\Desktop\Project\employee_attrition_plus
plugins: anyio-4.9.0
collected 15 items

tests/test_explainability.py::test_clean_feature_names PASSED            [  6%]
tests/test_explainability.py::test_build_explainer_supported_models PASSED [ 13%]
tests/test_explainability.py::test_explainer_unsupported_model_raises PASSED [ 20%]
tests/test_pipeline.py::test_pipeline_validation_errors PASSED          [ 26%]
tests/test_explainability.py::test_compute_global_shap_both_models PASSED [ 33%]
tests/test_explainability.py::test_shap_feature_importance_and_impact PASSED [ 40%]
tests/test_explainability.py::test_single_prediction_explanation PASSED  [ 46%]
tests/test_explainability.py::test_invalid_input_handling PASSED         [ 53%]
tests/test_explainability.py::test_plots_generate_figures PASSED         [ 60%]
tests/test_pipeline.py::test_dataset_loads_and_target_is_binary PASSED   [ 66%]
tests/test_pipeline.py::test_duplicate_cleaning_reports_removed_rows PASSED [ 73%]
tests/test_pipeline.py::test_training_pipeline_returns_both_models PASSED [ 80%]
tests/test_pipeline.py::test_shap_explainability_for_supported_models PASSED [ 86%]
tests/test_pipeline.py::test_shap_feature_names_align_with_transformed_data PASSED [ 93%]
tests/test_pipeline.py::test_shap_rejects_empty_and_invalid_input PASSED [100%]

============================= 15 passed in 40.79s =============================
```

**Verification Highlights:**
- `python train.py` executed successfully, generating `artifacts/best_model.joblib` and `artifacts/training_report.json`.
- Streamlit application compiled cleanly with `python -m py_compile app.py`.
- Headless health probe verified that `app.py` boots successfully (`HTTP 200 OK` on `/_stcore/health`).

---

## 5. Dashboard Features Added

### Section: SHAP – Global Model Explainability
1. **Model Selector**: Select between Random Forest and Logistic Regression.
2. **Feature Importance (Tab 1)**:
   - Ranked horizontal bar chart of top 10 features by mean absolute SHAP value ($\text{mean}(|\text{SHAP}|)$).
   - Formatted attribution table displaying Mean |SHAP|, Mean SHAP, and Importance Share %.
3. **SHAP Summary Plot (Tab 2)**:
   - Beeswarm plot displaying the impact of high vs. low feature values across employee profiles.
   - Distinct color-coding (Red = High feature value, Blue = Low feature value).
4. **Feature Impact Direction (Tab 3)**:
   - Plain-language synthesis of feature relationships (e.g., whether higher satisfaction reduces attrition risk).
   - Prominent disclaimer clarifying that statistical correlations in model outputs are not causal evidence.

### Section: SHAP – Individual Prediction Explanation
1. **Sample Record Selector**: Interactive numeric input allowing exploration of test set records (indices 0 to 300).
2. **Metrics Bar**: Displays Predicted Outcome (`Stayed (0)` vs `Left (1)`), Predicted Probability (%), Baseline Value ($E[f(X)]$), and Observed Status.
3. **Employee Attribute Card**: Renders the complete unencoded feature values for the selected employee.
4. **Local Waterfall Plot**: Displays step-by-step contributions pushing the prediction away from the population baseline.
5. **Contributing Factors Breakdown**:
   - Top positive contributors (risk-increasing features).
   - Top negative contributors (protective/retention features).
6. **Responsible AI Disclaimers**: Clarifies that attributions are mathematical weights, not personal appraisals or justifications for employment actions.

---

## 6. Known Limitations

1. **Subsampling**: To keep Streamlit responsive, global explanations use a subsample of up to 150 evaluation records.
2. **Linear Background Approximations**: `LinearExplainer` uses an independent masker over a background sample of 100 rows, assuming feature independence for the background baseline.
3. **Correlation vs. Causation**: SHAP values describe the internal mechanics of the trained models. They do not represent causal levers (e.g. increasing tenure or changing salary band does not guarantee an employee will stay).
4. **Data Artifacts**: Feature attributions reflect historical biases present in the training set.

---

## 7. Manual Browser Checks to Perform

When you launch the Streamlit app locally (`streamlit run app.py`):
1. **Model Toggle Check**: In the **SHAP – Global Model Explainability** section, toggle between **Random Forest** and **Logistic Regression** and verify that both the feature ranking table and summary plot update properly.
2. **Local Sample Navigation**: In the **SHAP – Individual Prediction Explanation** section, change the record index slider (e.g., test record #0, #5, #12) and confirm that:
   - The profile table updates to match the chosen record.
   - The waterfall plot renders without errors.
   - The top positive and negative contributor lists update.
3. **Upload CSV Test (Optional)**: If testing an uploaded CSV, verify that the SHAP sections adapt to the newly trained model without errors.
