# Workforce Insights - Employee Attrition Prediction

Workforce Insights is an educational Python/Streamlit application for employee attrition prediction and responsible aggregate workforce analytics. It combines Logistic Regression and Random Forest models with SHAP explainability.

## 1. Project Introduction

The application demonstrates how an organization might explore historical employee retention patterns. It is designed for research, learning, and human-reviewed analysis. It is not a production HR decision system.

## 2. Problem Statement

Employee attrition can affect workforce continuity and organizational planning. This project estimates historical/modelled attrition patterns while preserving privacy and avoiding individual-level employment decisions.

## 3. Key Features

- CSV upload with schema and target validation.
- Exact duplicate removal and missing-feature reporting.
- Random Forest and Logistic Regression comparison.
- User-triggered bounded hyperparameter optimization for Logistic Regression, Random Forest, or both.
- Separate baseline and optimized model/version selection in evaluation and SHAP views.
- Held-out baseline-versus-optimized metric comparison and training-only CV F1/ROC-AUC reporting.
- Separate optimized model and metadata artifact files.
- Adjustable classification threshold.
- Model metrics: accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
- Global and individual SHAP explanations.
- Department, salary, tenure, work-accident, promotion, and tenure-range filters.
- Minimum reliable subgroup-size control.
- Department, salary, tenure, and satisfaction/performance analytics.
- Privacy-safe aggregate CSV export.
- Responsible AI notices and model/causation limitations.

## 4. Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- SHAP
- Streamlit
- Matplotlib and Plotly
- Joblib
- pytest and Ruff

## 5. Project Architecture

```text
app.py                         Streamlit dashboard
train.py                       Model training entry point
src/attrition.py               Schema validation, cleaning, training, metrics, persistence
src/explainability.py          SHAP calculations and plots
src/workforce_analytics.py     Filter and aggregate workforce analytics used by app.py
src/analytics.py               Independent analytics helpers and tests retained for Milestone 2.x coverage
tests/                         Automated validation
data/employee_data.csv         Supplied employee dataset
artifacts/                     Saved model and training report
notebooks/                     Original reference notebook
```

`src/workforce_analytics.py` is the dashboard source of truth. `src/analytics.py` provides a parallel modular implementation that remains useful for independent testable analytics calculations; it is not removed because preserving Milestone 2.1/2.2 behavior is a release requirement.

## 6. Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 7. Dependency Installation

```powershell
python -m pip install -r requirements.txt
```

Key dependencies are declared in `requirements.txt`: pandas, NumPy, scikit-learn, Streamlit, Matplotlib, Joblib, pytest, SHAP, Plotly, SciPy, and Requests.

## 8. Dataset Requirements

The application requires a CSV with these columns:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `quit`
- `promotion_last_5years`
- `department`
- `salary`

The `quit` target must contain only `0` and `1`, and the dataset must contain both classes with enough records for stratified validation.

The supplied dataset is `data/employee_data.csv`. It contains 11,582 raw rows. Exact duplicate removal leaves 11,573 records and removes 9 duplicate rows.

## 9. Application Execution

```powershell
python -m streamlit run app.py
```

Production-mode startup on Windows:

```powershell
$env:APP_ENV="production"
$env:LOG_LEVEL="INFO"
$env:DATA_PATH="data/employee_data.csv"
$env:ARTIFACT_DIR="artifacts"
$env:MONITORING_DIR="artifacts/monitoring"
$env:ALLOW_ARTIFACT_WRITES="true"
.\start_streamlit.ps1
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## 10. Model Training Instructions

Train and save the selected model:

```powershell
python train.py
```

This writes:

- `artifacts/best_model.joblib`
- `artifacts/training_report.json`

User-triggered dashboard optimization writes only separate artifacts:

- `artifacts/optimized_best_model.joblib`
- `artifacts/optimized_training_report.json`

The baseline artifacts are never overwritten by the optimized workflow.

Training uses exact-duplicate-cleaned data, a stratified 80/20 split, 3-fold stratified cross-validation, and preprocessing fitted within each training fold.

## 10.1 Bounded Hyperparameter Optimization

The Streamlit dashboard provides an explicit **Run bounded optimization** form. The run is never triggered by ordinary widget reruns or analytics filters.

- Randomized search candidates: 1–6 per selected model.
- Stratified cross-validation: 3 folds on training records only.
- Parallelism: outer and inner estimator jobs are fixed to one job.
- Random seed: 42.
- Selection criterion: training-fold CV F1.
- CV ROC-AUC: reported alongside CV F1, but not used for parameter selection.
- Held-out test data: used once after tuning for final evaluation.

Successful optimized results are kept in Streamlit session state. Partial success retains successful candidates and reports failures. An all-failed run does not replace prior successful session results.

The dashboard can select either Baseline or Optimized for the evaluation threshold, permutation importance, global SHAP, and individual SHAP views. The optimized SHAP cache key includes the model version and run identity to prevent explanations from being reused across different pipelines.

## 10.2 Model Monitoring and Data Drift

The dashboard includes a user-triggered **Model Monitoring & Data Drift** section for CSV or TXT monitoring batches.

- Monitoring compares the current batch with an aggregate training-reference summary.
- Numeric features use PSI and the two-sample KS test.
- Categorical features use proportion shifts, unseen/disappeared categories, and Chi-square where frequency assumptions are suitable.
- Data-quality checks report missing values, duplicates, invalid values, and out-of-range fields.
- Prediction monitoring uses the selected Baseline or Optimized pipeline and the active threshold.
- Delayed performance metrics appear only when valid binary `quit` ground truth is present.
- Monitoring status combines quality and drift signals into Healthy, Warning, Critical, or Insufficient Data results.
- Reports are saved only below `artifacts/monitoring/` as aggregate reference metadata, history, and JSON/CSV exports.
- Repeated runs with the same reference fingerprint, monitoring fingerprint, model identity, and threshold are deduplicated in monitoring history.

Drift indicates a distribution change; it does not by itself establish bias, inaccuracy, or causation. Monitoring exports contain no employee identifiers, individual records, or individual prediction records.

## 11. ML Model Explanation

The app compares:

- Random Forest, using balanced subsampling.
- Logistic Regression, using balanced class weights.

The decision threshold changes demonstration metrics without retraining the model. ROC-AUC and PR-AUC describe ranking quality; threshold-dependent metrics show effects at the selected operating threshold.

## 12. SHAP Explainability

SHAP explanations support:

- Global feature importance using mean absolute SHAP value.
- Global SHAP beeswarm/summary plots.
- Feature impact direction.
- Individual waterfall plots.
- Exact reconstruction of the selected model output.

Random Forest SHAP reconstruction is in probability space. Logistic Regression reconstruction is in log-odds/margin space and is converted through sigmoid for probability display.

SHAP values describe model behaviour and historical statistical associations. They do not establish causation or employee intent.

## 13. Workforce Analytics

The dashboard provides four sections:

1. Department
2. Salary
3. Tenure
4. Satisfaction and Performance

Tenure is grouped into `0-1`, `2`, `3`, `4`, `5`, `6-10`, and `10+` years. Missing or unknown categories are handled explicitly.

Department, salary, tenure range, work accident, promotion, and minimum reliable group size filter aggregate results. The minimum group size prevents unreliable subgroup ranking.

## 14. Aggregate CSV Export

The Workforce Analytics section provides a `Download filtered aggregate analytics CSV` button.

The export contains:

- `dimension`
- `category`
- `employee_count`
- `left_count`
- `stayed_count`
- `attrition_rate`
- `reliable`

It does not include employee names, employee IDs, individual predictions, individual SHAP values, or row-level employee records.

## 15. Responsible AI Limitations

This application must not be used to rank employees, make promotion or termination decisions, or justify adverse employment actions.

- Observed correlations do not establish causation.
- SHAP explains model behaviour, not human intentions.
- Historical training data may encode organizational bias or unmeasured factors.
- Aggregate workforce patterns require human review and privacy safeguards.
- Individual predictions must not be treated as personal appraisals.

## 16. Known Limitations

- The supplied data is a historical sample, not a complete representation of every employee or organization.
- The exact-row deduplication method cannot distinguish separate people that share identical values in all columns.
- Missing values are handled inside the modeling pipeline.
- SHAP global calculations sample rows for responsiveness.
- Streamlit reruns may rebuild or retrain model state unless cached.
- Browser behavior can vary across environments.
- The app is an educational prototype, not a validated or legally compliant production HR system.

## 17. Testing

Run the complete suite:

```powershell
python -m pytest -v
python -m ruff check src/ app.py tests/
python -m py_compile app.py
python -m streamlit run app.py
```

## 18. Project Documentation

Canonical project knowledge is organized under [docs/](./docs/):

- [PRD](./docs/PRD.md)
- [Architecture](./docs/architecture.md)
- [Design](./docs/design.md)
- [Engineering memory](./docs/memory.md)
- [Repository rules](./docs/rule.md)
- [Task checklist](./docs/task.md)

## 19. Production Deployment

See [architecture and deployment documentation](./docs/architecture.md) for Docker, Render, Streamlit Community Cloud, environment variables, health checks, persistence, security, and troubleshooting instructions.

For a containerized local deployment:

```powershell
docker build -t workforce-insights:m2.7 .
docker run --rm -p 8501:8501 workforce-insights:m2.7
curl.exe -I http://localhost:8501/_stcore/health
```

Copy `.env.example` to `.env` for local overrides. Do not commit `.env` or Streamlit secrets. Monitoring persistence is local-filesystem based and requires external storage for durable hosted deployments.
