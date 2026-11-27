# QA Report: SHAP Consistency Audit

## Scope

Files inspected:

- `src/explainability.py`
- `app.py`
- `tests/test_explainability.py`
- `tests/test_pipeline.py`
- `src/attrition.py`
- `requirements.txt`

No dashboard redesign was performed. This audit was limited to SHAP output
space, positive-class selection, baseline, reconstruction, feature-name
alignment, tests, and startup verification.

## Environment

- Python: 3.13.12
- NumPy: 2.2.4
- pandas: 2.3.3
- scikit-learn: 1.9.0
- SHAP: 0.54.1
- Streamlit: 1.51.0
- pytest: 9.2.0

## Random Forest Output-Space Check

TreeExplainer for the binary Random Forest returns shape `(1, 20, 2)` for the
audited record. Class 1 (`quit = 1`) is selected with index `1`.

TreeExplainer class-1 output is in probability space:

- Pipeline probability: `0.9976391655688709`
- Class-1 baseline: `0.49999999999999944`
- Sum of all 20 feature contributions: `0.4976391655688621`
- Baseline + all contributions: `0.9976391655688616`
- Reconstructed probability: `0.9976391655688615`

Difference from `predict_proba(...)[, 1]`: `9.3e-15` or less in observed
floating-point arithmetic.

The displayed 99.8% probability is consistent with the full reconstruction.

## Logistic Regression Output-Space Check

LinearExplainer is explicitly configured with `model_output="raw"`. Its native
output space is log-odds (margin), not probability.

For the audited record:

- Pipeline probability: `0.6561093607351659`
- Baseline log-odds: `-0.9605213188891633`
- Sum of all 20 feature contributions: `1.6065251046609685`
- Baseline + all contributions: `0.6460037857718053`
- Sigmoid of reconstructed margin: `0.6561093607351658`

The reconstructed margin matches `decision_function(...)[0]`. The sigmoid
matches `predict_proba(...)[, 1]`. The difference is within floating-point
precision.

The implementation does not add log-odds SHAP values to a probability baseline.
The dashboard labels the baseline and reconstruction output space accordingly.

## Positive-Class and Feature Alignment

- Random Forest class 1 is selected explicitly at class index 1.
- Logistic Regression uses the pipeline's class-1 probability and native
  `decision_function` margin for reconstruction.
- All 20 transformed SHAP values correspond one-to-one with 20 feature names.
- Categorical names remain aligned, including `department: technical`,
  `department: support`, `salary: low`, and `salary: medium`.

## Audited Positive and Negative Values

Random Forest top positive values:

- `number_project`: `+0.2421277131`
- `satisfaction_level`: `+0.1787932875`
- `last_evaluation`: `+0.0783245909`
- `Work_accident`: `+0.0061086809`

Random Forest top negative values:

- `salary: low`: `-0.0102693509`
- `salary: medium`: `-0.0024895295`
- `department: technical`: `-0.0014308771`
- `department: support`: `-0.0006181313`

These displayed lists are only top contributions. Reconstruction uses all 20
features, including values outside those lists.

## Automated Tests

Executed:

```text
python -m pytest -q
24 passed in 42.13s
```

Tests added or retained for this audit verify:

- Random Forest positive-class reconstruction using all transformed features.
- Logistic Regression native log-odds reconstruction.
- Sigmoid conversion to pipeline class-1 probability.
- Global batch reconstruction for both model types.
- Correct categorical one-hot feature-name alignment.
- Supported/unsupported explainer handling.
- Empty and invalid input handling.

## Dashboard Startup

Started:

```text
python -m streamlit run app.py --server.headless true --server.port 8501
```

HTTP startup verification returned status `200` from
`http://localhost:8501`. The process was stopped after verification.

## Known Limitations

- HTTP startup verification does not replace manual browser checks of plot
  rendering and widget behavior.
- Logistic Regression SHAP values are log-odds contributions. Positive or
  negative signs indicate margin direction, not directly additive percentage
  points.
- SHAP explanations describe model behavior, not actual employee intentions or
  causal effects.
- Existing source data may contain duplicate feature profiles; results should
  not be interpreted as individual evidence of why an employee left.

## Manual Browser Checks Remaining

1. Open `http://localhost:8501`.
2. Switch the global model between Random Forest and Logistic Regression.
3. Confirm both feature-ranking and summary plots render.
4. Select an individual record and confirm its probability and predicted outcome.
5. Confirm Random Forest displays a probability baseline near `0.500`.
6. Confirm Logistic Regression displays a log-odds baseline and that the
   reconstruction matches the displayed sigmoid probability.
