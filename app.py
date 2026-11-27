from pathlib import Path
import base64
import html

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.attrition import (
    compare_baseline_and_optimized,
    clean_dataset,
    get_candidate_models,
    get_hyperparameter_search_spaces,
    get_permutation_importance,
    load_dataset,
    optimize_models,
    save_optimized_model,
    threshold_metrics,
    train_and_compare,
    validate_dataset_frame,
)
from src.explainability import (
    ShapExplanationResult,
    compute_global_shap_explanations,
    explain_single_prediction,
    get_feature_impact_summary,
    get_shap_feature_importance,
    plot_shap_importance_bar,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.workforce_analytics import (
    apply_workforce_filters,
    build_aggregate_export,
    aggregate_export_to_csv,
    department_attrition,
    highest_attrition_group,
    salary_attrition,
    tenure_attrition,
)

st.set_page_config(page_title="Workforce Insights | Attrition Lab", page_icon="📊", layout="wide")

@st.cache_data(show_spinner=False)
def cached_training(dataframe):
    """Cache training outputs so Streamlit widget changes do not retrain models."""
    return train_and_compare(dataframe)

@st.cache_data(show_spinner=False)
def cached_global_shap(model_cache_key, _pipeline, dataframe, max_samples=150, random_state=42):
    """Cache SHAP global calculations so reruns remain instant and responsive."""
    return compute_global_shap_explanations(
        _pipeline, dataframe, background_df=dataframe,
        max_samples=max_samples, random_state=random_state
    )

def format_metric_delta(absolute_delta, relative_delta):
    if pd.isna(absolute_delta):
        return "N/A"
    if absolute_delta > 0:
        indicator = "↑"
    elif absolute_delta < 0:
        indicator = "↓"
    else:
        indicator = "→"
    relative = (
        f"{relative_delta:+.1f}%"
        if pd.notna(relative_delta)
        else "relative n/a"
    )
    return f"{indicator} {absolute_delta:+.3f} ({relative})"

st.title("Workforce Insights")
st.caption("Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI")
st.warning(
    "Educational prototype only. Do not use predictions to rank, penalize, hire, "
    "promote, or terminate employees. Use aggregated insights to explore retention "
    "questions, with human review and appropriate privacy safeguards."
)

DEFAULT_DATA = Path("data/employee_data.csv")
uploaded = st.sidebar.file_uploader("Upload an employee CSV", type=["csv"])
try:
    if uploaded is not None:
        raw_upload = pd.read_csv(uploaded)
        data = validate_dataset_frame(raw_upload)
    else:
        data = load_dataset(DEFAULT_DATA)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

# Deduplicate before analytics. The cleaned frame is the shared workforce source
# of truth for all aggregate counts, charts, and CSV exports.
cleaned_data, cleaning_report = clean_dataset(data)
data_fingerprint = joblib.hash(cleaned_data)
if st.session_state.get("m25_data_fingerprint") != data_fingerprint:
    st.session_state["m25_data_fingerprint"] = data_fingerprint
    st.session_state["m25_optimization_results"] = {}
    st.session_state["m25_optimization_failures"] = {}
    st.session_state["m25_optimization_comparison"] = pd.DataFrame()
    st.session_state["m25_optimization_status"] = ""
    st.session_state["m25_optimized_artifacts"] = {}
    st.session_state["m25_optimization_run_version"] = 0

# Sidebar filters affect only the aggregate workforce analytics section.
st.sidebar.divider()
st.sidebar.header("Workforce Analytics Filters")

department_options = sorted(cleaned_data["department"].dropna().astype(str).unique().tolist()) if "department" in cleaned_data.columns else []
salary_options = sorted(cleaned_data["salary"].dropna().astype(str).unique().tolist()) if "salary" in cleaned_data.columns else []
selected_departments = st.sidebar.multiselect("Department", department_options)
selected_salaries = st.sidebar.multiselect("Salary", salary_options)

tenure_values = pd.to_numeric(cleaned_data.get("time_spend_company"), errors="coerce").dropna()
if len(tenure_values):
    tenure_min = int(tenure_values.min())
    tenure_max = int(max(tenure_values.max(), tenure_min))
    selected_tenure = st.sidebar.slider(
        "Years at company",
        min_value=tenure_min,
        max_value=tenure_max,
        value=(tenure_min, tenure_max),
    )
else:
    selected_tenure = None

work_accident_options = sorted(cleaned_data["Work_accident"].dropna().astype(str).unique().tolist()) if "Work_accident" in cleaned_data.columns else []
promotion_options = sorted(cleaned_data["promotion_last_5years"].dropna().astype(str).unique().tolist()) if "promotion_last_5years" in cleaned_data.columns else []
selected_work_accidents = st.sidebar.multiselect("Work accident", work_accident_options)
selected_promotions = st.sidebar.multiselect("Promotion in last 5 years", promotion_options)
min_group_size = st.sidebar.slider(
    "Minimum reliable group size",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
)

analytics_data = apply_workforce_filters(
    cleaned_data,
    departments=selected_departments,
    salaries=selected_salaries,
    tenure_range=selected_tenure,
    work_accidents=selected_work_accidents,
    promotions=selected_promotions,
)

with st.spinner("Training and comparing baseline models..."):
    try:
        results, comparison, cleaning_report = cached_training(data)
    except ValueError as exc:
        st.error(f"Unable to train models on the provided dataset: {exc}")
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

if not results:
    st.error("No trained models are available. Please check the training dataset.")
    st.stop()

best_name = comparison.iloc[0]["model"]
best = results[best_name]

st.subheader("Dataset overview")
cleaned_rows = cleaning_report["rows_after"]
attrition_rate = float(cleaned_data["quit"].mean() * 100)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Uploaded records", f"{len(data):,}")
c2.metric("Records after deduplication", f"{cleaned_rows:,}")
c3.metric("Observed attrition rate", f"{attrition_rate:.1f}%")
c4.metric("Duplicate rows removed", f"{cleaning_report['duplicates_removed']:,}")

st.caption(
    f"Missing feature cells after duplicate removal: "
    f"{cleaning_report['missing_feature_cells']}. Missing values are imputed "
    "inside each training fold."
)

left, right = st.columns(2)
with left:
    st.markdown("**Target distribution**")
    counts = cleaned_data["quit"].value_counts().rename(index={0: "Stayed", 1: "Left"})
    st.bar_chart(counts)
with right:
    st.markdown("**Model comparison**")
    st.dataframe(comparison.style.format({
        col: "{:.3f}" for col in comparison.columns if col != "model"
    }), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Milestone 2.5: User-triggered, isolated optimization workflow
# ---------------------------------------------------------------------------
st.divider()
st.header("Model Comparison & Performance Evaluation")
st.info(
    "Optimization runs only when you submit this form. It uses at most 6 "
    "randomized candidates per selected model, 3 stratified training folds, "
    "random state 42, and one job. All preprocessing is fitted within each "
    "training fold. The shared held-out test split is used only for final "
    "evaluation and is never used to select parameters."
)
with st.form("m25_optimization_form"):
    optimization_choice = st.selectbox(
        "Models to optimize",
        ["Logistic Regression", "Random Forest", "Both models"],
        key="m25_optimization_model_choice",
    )
    optimization_iterations = st.slider(
        "Randomized search candidates per model",
        min_value=1,
        max_value=6,
        value=6,
        key="m25_optimization_n_iter",
        help="Hard-capped at 6 candidates per model; each candidate uses 3 CV folds.",
    )
    run_optimization = st.form_submit_button("Run bounded optimization")

if run_optimization:
    all_candidates = get_candidate_models()
    all_search_spaces = get_hyperparameter_search_spaces()
    selected_names = (
        list(all_candidates)
        if optimization_choice == "Both models"
        else [optimization_choice]
    )
    try:
        with st.spinner("Running the requested bounded optimization..."):
            optimized_results, _, _, optimization_failures = optimize_models(
                data,
                random_state=42,
                n_iter=int(optimization_iterations),
                n_jobs=1,
                candidate_models={
                    name: all_candidates[name] for name in selected_names
                },
                search_spaces={
                    name: all_search_spaces[name] for name in selected_names
                },
            )
            if not optimized_results:
                st.session_state["m25_optimization_failures"] = optimization_failures
                st.session_state["m25_optimization_status"] = (
                    "No selected model completed optimization. Existing successful "
                    "results, if any, have been retained."
                )
            else:
                optimized_comparison = compare_baseline_and_optimized(
                    {name: results[name] for name in optimized_results},
                    optimized_results,
                )
                best_optimized_name = max(
                    optimized_results,
                    key=lambda name: optimized_results[name].best_cv_score,
                )
                model_path, report_path = save_optimized_model(
                    optimized_results[best_optimized_name],
                    cleaning_report,
                )
                st.session_state["m25_optimization_results"] = optimized_results
                st.session_state["m25_optimization_failures"] = optimization_failures
                st.session_state["m25_optimization_comparison"] = optimized_comparison
                st.session_state["m25_optimized_artifacts"] = {
                    "model": str(model_path),
                    "report": str(report_path),
                }
                st.session_state["m25_optimization_run_version"] = (
                    st.session_state.get("m25_optimization_run_version", 0) + 1
                )
                st.session_state["m25_optimization_status"] = (
                    f"Optimization completed. Saved the CV-F1-selected model "
                    f"({best_optimized_name}) to the separate optimized artifacts."
                )
    except Exception as exc:
        st.session_state["m25_optimization_status"] = (
            f"The latest optimization run failed: {exc}"
        )
        st.error(f"Could not complete optimization: {exc}")

optimization_status = st.session_state.get("m25_optimization_status", "")
if optimization_status:
    st.caption(optimization_status)
    if "failed" in optimization_status.lower() or "no selected model" in optimization_status.lower():
        st.warning(optimization_status)
    saved_artifacts = st.session_state.get("m25_optimized_artifacts", {})
    if saved_artifacts:
        st.caption(
            f"Optimized artifacts: `{saved_artifacts['model']}` and "
            f"`{saved_artifacts['report']}`. Baseline artifacts are not modified."
        )

optimization_failures = st.session_state.get("m25_optimization_failures", {})
for failed_name, failure_message in optimization_failures.items():
    st.warning(f"{failed_name} optimization was not available: {failure_message}")

optimized_results = st.session_state.get("m25_optimization_results", {})
optimized_comparison = st.session_state.get("m25_optimization_comparison", pd.DataFrame())
if optimized_results and not optimized_comparison.empty:
    st.subheader("Held-out test comparison")
    st.caption(
        "Baseline and optimized values use the exact same deterministic held-out "
        "records. CV values come from the training split only; positive/negative "
        "arrows indicate numerical change, not desirability or fairness."
    )
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "cv_f1": "CV F1 mean",
        "cv_roc_auc": "CV ROC-AUC mean",
    }
    comparison_rows = []
    for _, comparison_row in optimized_comparison.iterrows():
        display_row = {"Model": comparison_row["model"]}
        for metric, label in metric_labels.items():
            display_row[f"{label} · baseline"] = comparison_row[f"baseline_{metric}"]
            display_row[f"{label} · optimized"] = comparison_row[f"optimized_{metric}"]
            display_row[f"{label} · change"] = format_metric_delta(
                comparison_row[f"{metric}_abs_delta"],
                comparison_row[f"{metric}_rel_pct"],
            )
        display_row["CV F1 std · baseline"] = comparison_row["baseline_cv_f1_std"]
        display_row["CV F1 std · optimized"] = comparison_row["optimized_cv_f1_std"]
        display_row["Baseline train seconds"] = comparison_row["baseline_train_time_sec"]
        display_row["Optimization seconds"] = comparison_row["optimized_optimization_time_sec"]
        display_row["Baseline prediction seconds"] = comparison_row["baseline_predict_time_sec"]
        display_row["Optimized prediction seconds"] = comparison_row["optimized_predict_time_sec"]
        comparison_rows.append(display_row)
    st.dataframe(
        pd.DataFrame(comparison_rows).style.format(
            {
                column: "{:.3f}"
                for column in pd.DataFrame(comparison_rows).columns
                if column != "Model" and "change" not in column
            }
        ),
        width="stretch",
        hide_index=True,
    )

    chart_columns = st.columns(2)
    with chart_columns[0]:
        test_metric_columns = [
            "baseline_accuracy", "optimized_accuracy",
            "baseline_precision", "optimized_precision",
            "baseline_recall", "optimized_recall",
            "baseline_f1", "optimized_f1",
            "baseline_roc_auc", "optimized_roc_auc",
            "baseline_pr_auc", "optimized_pr_auc",
        ]
        test_chart = optimized_comparison.melt(
            id_vars=["model"],
            value_vars=test_metric_columns,
            var_name="version_metric",
            value_name="score",
        )
        test_chart["Version"] = test_chart["version_metric"].str.startswith(
            "optimized_"
        ).map({True: "Optimized", False: "Baseline"})
        test_chart["Metric"] = (
            test_chart["version_metric"]
            .str.replace("baseline_", "", regex=False)
            .str.replace("optimized_", "", regex=False)
            .str.replace("_", " ")
            .str.upper()
        )
        fig = px.bar(
            test_chart,
            x="Metric",
            y="score",
            color="Version",
            barmode="group",
            facet_col="model",
            title="Baseline vs. optimized held-out metrics",
        )
        st.plotly_chart(fig, width="stretch")
    with chart_columns[1]:
        cv_chart = optimized_comparison.melt(
            id_vars=["model"],
            value_vars=[
                "baseline_cv_f1", "optimized_cv_f1",
                "baseline_cv_roc_auc", "optimized_cv_roc_auc",
            ],
            var_name="version_metric",
            value_name="score",
        )
        cv_chart["Version"] = cv_chart["version_metric"].str.startswith(
            "optimized_"
        ).map({True: "Optimized", False: "Baseline"})
        cv_chart["Metric"] = cv_chart["version_metric"].str.contains(
            "roc_auc"
        ).map({True: "CV ROC-AUC mean", False: "CV F1 mean"})
        fig = px.bar(
            cv_chart,
            x="Metric",
            y="score",
            color="Version",
            barmode="group",
            facet_col="model",
            title="Training-only cross-validation means",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "CV F1 standard deviations are reported in the comparison table. "
            "The held-out test set did not participate in CV or parameter selection."
        )

    best_optimized_name = max(
        optimized_results,
        key=lambda name: optimized_results[name].best_cv_score,
    )
    best_optimized = optimized_results[best_optimized_name]
    st.subheader("Best cross-validated optimized model")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Model", best_optimized_name)
    summary_columns[1].metric("Training CV F1", f"{best_optimized.best_cv_score:.3f}")
    summary_columns[2].metric("Training CV F1 std", f"{best_optimized.cv_std:.3f}")
    summary_columns[3].metric(
        "Held-out F1",
        f"{best_optimized.metrics['f1']:.3f}",
        help="Reported for evaluation only; it did not select the model or parameters.",
    )
    st.write("Selected hyperparameters")
    st.json(best_optimized.best_params)
else:
    st.caption(
        "No optimized results are available for this dataset yet. Submit the form "
        "above to run optimization; dashboard reruns do not launch it automatically."
    )

# ---------------------------------------------------------------------------
# Workforce Analytics
# ---------------------------------------------------------------------------
st.divider()
st.header("Interactive Workforce Analytics")
st.info(
    "This dashboard is an educational workforce analytics prototype. "
    "Observed historical patterns and model predictions do not establish causation "
    "or determine an individual's future behavior. Do not use these outputs to rank, "
    "penalize, hire, promote, or terminate employees. Any organizational "
    "interpretation requires appropriate privacy safeguards and human review."
)

if analytics_data.empty:
    st.warning("No records match the selected analytics filters.")
else:
    department_summary = department_attrition(analytics_data, min_group_size=min_group_size)
    salary_summary = salary_attrition(analytics_data, min_group_size=min_group_size)
    tenure_summary = tenure_attrition(analytics_data, min_group_size=min_group_size)

    overall_attrition = float(pd.to_numeric(analytics_data["quit"], errors="coerce").mean())
    department_insight = highest_attrition_group(department_summary, "Department")
    salary_insight = highest_attrition_group(salary_summary, "Salary")
    tenure_insight = highest_attrition_group(tenure_summary, "Tenure")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filtered employees", f"{len(analytics_data):,}")
    k2.metric("Observed attrition rate", f"{overall_attrition:.1%}")
    k3.metric(
        "Highest reliable department rate",
        f"{department_insight['attrition_rate']:.1%}" if department_insight["attrition_rate"] is not None else "Not enough data",
        help=department_insight["group"] if department_insight["group"] is not None else None,
    )
    k4.metric(
        "Highest reliable salary rate",
        f"{salary_insight['attrition_rate']:.1%}" if salary_insight["attrition_rate"] is not None else "Not enough data",
        help=salary_insight["group"] if salary_insight["group"] is not None else None,
    )

    insight_col1, insight_col2, insight_col3 = st.columns(3)
    if department_insight["group"]:
        insight_col1.success(
            f"Department: **{department_insight['group']}** "
            f"({department_insight['employee_count']} employees)"
        )
    else:
        insight_col1.info("Department ranking unavailable: no reliable-size group.")
    if salary_insight["group"]:
        insight_col2.success(
            f"Salary: **{salary_insight['group']}** "
            f"({salary_insight['employee_count']} employees)"
        )
    else:
        insight_col2.info("Salary ranking unavailable: no reliable-size group.")
    if tenure_insight["group"]:
        insight_col3.success(
            f"Tenure: **{tenure_insight['group']}** "
            f"({tenure_insight['employee_count']} employees)"
        )
    else:
        insight_col3.info("Tenure ranking unavailable: no reliable-size group.")

    st.caption(
        f"Rankings require at least {min_group_size} records in a group. "
        "These are descriptive historical rates, not predictions or causal findings."
    )

    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
        "Department", "Salary", "Tenure", "Satisfaction & Performance",
    ])

    with analytics_tab1:
        dept_distribution = analytics_data.copy()
        dept_distribution["Observed outcome"] = dept_distribution["quit"].map({
            0: "Stayed", 1: "Left",
        }).fillna("Unknown")
        if "department" in dept_distribution.columns:
            fig = px.histogram(
                dept_distribution,
                x="department",
                color="Observed outcome",
                barmode="stack",
                title="Employee Distribution and Attrition by Department",
                labels={"department": "Department", "count": "Employee count"},
                category_orders={"Observed outcome": ["Stayed", "Left"]},
                color_discrete_map={"Stayed": "#4C78A8", "Left": "#E45756", "Unknown": "#B0B0B0"},
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, width="stretch")

        if not department_summary.empty:
            dept_chart = department_summary.copy()
            dept_chart["attrition_percent"] = dept_chart["attrition_rate"] * 100
            fig = px.bar(
                dept_chart,
                x="group",
                y="attrition_percent",
                color="reliable",
                color_discrete_map={True: "#E45756", False: "#B0B0B0"},
                title="Observed Attrition Rate by Department",
                labels={"group": "Department", "attrition_percent": "Attrition rate (%)", "reliable": "Reliable sample"},
                hover_data=["employee_count", "left_count", "stayed_count"],
                text=dept_chart["attrition_percent"].round(1),
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis_ticksuffix="%", xaxis_tickangle=-35)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                dept_chart.drop(columns=["attrition_percent"]).style.format({
                    "attrition_rate": "{:.1%}",
                }),
                width="stretch",
                hide_index=True,
            )

    with analytics_tab2:
        salary_distribution = analytics_data.copy()
        salary_distribution["Observed outcome"] = salary_distribution["quit"].map({
            0: "Stayed", 1: "Left",
        }).fillna("Unknown")
        if "salary" in salary_distribution.columns:
            fig = px.histogram(
                salary_distribution,
                x="salary",
                color="Observed outcome",
                barmode="group",
                title="Employee Distribution and Attrition by Salary Category",
                labels={"salary": "Salary category", "count": "Employee count"},
                category_orders={"Observed outcome": ["Stayed", "Left"]},
                color_discrete_map={"Stayed": "#4C78A8", "Left": "#E45756", "Unknown": "#B0B0B0"},
            )
            st.plotly_chart(fig, width="stretch")

        if not salary_summary.empty:
            salary_chart = salary_summary.copy()
            salary_chart["attrition_percent"] = salary_chart["attrition_rate"] * 100
            fig = px.bar(
                salary_chart,
                x="group",
                y="attrition_percent",
                color="reliable",
                color_discrete_map={True: "#F58518", False: "#B0B0B0"},
                title="Observed Attrition Rate by Salary Category",
                labels={"group": "Salary category", "attrition_percent": "Attrition rate (%)", "reliable": "Reliable sample"},
                hover_data=["employee_count", "left_count", "stayed_count"],
                text=salary_chart["attrition_percent"].round(1),
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis_ticksuffix="%")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                salary_chart.drop(columns=["attrition_percent"]).style.format({
                    "attrition_rate": "{:.1%}",
                }),
                width="stretch",
                hide_index=True,
            )

    with analytics_tab3:
        tenure_chart = analytics_data.copy()
        tenure_chart["Observed outcome"] = tenure_chart["quit"].map({
            0: "Stayed", 1: "Left",
        }).fillna("Unknown")
        if "time_spend_company" in tenure_chart.columns:
            tenure_chart["Years at company"] = pd.to_numeric(
                tenure_chart["time_spend_company"], errors="coerce"
            )
            fig = px.histogram(
                tenure_chart,
                x="Years at company",
                color="Observed outcome",
                barmode="stack",
                nbins=max(10, int(tenure_chart["Years at company"].max(skipna=True) or 10)),
                title="Employee Distribution by Years at Company",
                labels={"Years at company": "Years at company", "count": "Employee count"},
                color_discrete_map={"Stayed": "#4C78A8", "Left": "#E45756", "Unknown": "#B0B0B0"},
            )
            st.plotly_chart(fig, width="stretch")

        if not tenure_summary.empty:
            tenure_rate_chart = tenure_summary.copy()
            tenure_rate_chart["attrition_percent"] = tenure_rate_chart["attrition_rate"] * 100
            fig = px.bar(
                tenure_rate_chart,
                x="group",
                y="attrition_percent",
                color="reliable",
                color_discrete_map={True: "#54A24B", False: "#B0B0B0"},
                title="Observed Attrition Rate by Tenure Group",
                labels={"group": "Tenure group", "attrition_percent": "Attrition rate (%)", "reliable": "Reliable sample"},
                hover_data=["employee_count", "left_count", "stayed_count"],
                text=tenure_rate_chart["attrition_percent"].round(1),
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis_ticksuffix="%")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                tenure_rate_chart.drop(columns=["attrition_percent"]).style.format({
                    "attrition_rate": "{:.1%}",
                }),
                width="stretch",
                hide_index=True,
            )

    with analytics_tab4:
        satisfaction_chart = analytics_data.copy()
        satisfaction_chart["Observed outcome"] = satisfaction_chart["quit"].map({
            0: "Stayed", 1: "Left",
        }).fillna("Unknown")
        if "satisfaction_level" in satisfaction_chart.columns:
            fig = px.histogram(
                satisfaction_chart,
                x="satisfaction_level",
                color="Observed outcome",
                barmode="overlay",
                opacity=0.65,
                nbins=20,
                title="Satisfaction-Level Distribution by Observed Outcome",
                labels={"satisfaction_level": "Satisfaction level", "count": "Employee count"},
                color_discrete_map={"Stayed": "#4C78A8", "Left": "#E45756", "Unknown": "#B0B0B0"},
            )
            st.plotly_chart(fig, width="stretch")
        if {"satisfaction_level", "last_evaluation"}.issubset(satisfaction_chart.columns):
            fig = px.scatter(
                satisfaction_chart,
                x="satisfaction_level",
                y="last_evaluation",
                color="Observed outcome",
                title="Satisfaction vs. Performance Evaluation",
                labels={
                    "satisfaction_level": "Satisfaction level",
                    "last_evaluation": "Last evaluation",
                },
                color_discrete_map={"Stayed": "#4C78A8", "Left": "#E45756", "Unknown": "#B0B0B0"},
                opacity=0.65,
            )
            st.plotly_chart(fig, width="stretch")
        if {"last_evaluation", "quit"}.issubset(satisfaction_chart.columns):
            fig = px.box(
                satisfaction_chart,
                x="Observed outcome",
                y="last_evaluation",
                color="Observed outcome",
                title="Performance Evaluation Distribution by Observed Outcome",
                labels={"last_evaluation": "Last evaluation", "Observed outcome": "Observed outcome"},
                color_discrete_map={"Stayed": "#4C78A8", "Left": "#E45756", "Unknown": "#B0B0B0"},
            )
            st.plotly_chart(fig, width="stretch")

    aggregate_export = build_aggregate_export(
        department_summary,
        salary_summary,
        tenure_summary,
    )
    csv_payload = aggregate_export_to_csv(aggregate_export)
    encoded_csv = base64.b64encode(csv_payload).decode("ascii")
    link_label = html.escape("Download filtered aggregate analytics CSV")
    st.markdown(
        f'<a href="data:text/csv;base64,{encoded_csv}" '
        'download="workforce_analytics_aggregate.csv" '
        'style="display:inline-block;padding:0.4rem 0.75rem;'
        'border:1px solid rgba(49,51,63,0.2);border-radius:0.25rem;'
        'text-decoration:none;color:inherit;">'
        f"{link_label}</a>",
        unsafe_allow_html=True,
    )
    st.caption(
        "The export excludes individual employee records and personal identifiers."
    )

st.subheader("Evaluation and threshold explorer")
model_name = st.selectbox("Choose a model", list(results.keys()), index=list(results.keys()).index(best_name))
evaluation_versions = ["Baseline"]
if model_name in optimized_results:
    evaluation_versions.append("Optimized")
evaluation_version = st.selectbox(
    "Model version for evaluation",
    evaluation_versions,
    key=f"m25_evaluation_version_{model_name.replace(' ', '_')}",
)
selected = (
    optimized_results[model_name]
    if evaluation_version == "Optimized"
    else results[model_name]
)
threshold = st.slider(
    "Decision threshold for the demonstration metrics",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="Changing this threshold changes the classification metrics. It does not retrain the model."
)
metrics, predictions = threshold_metrics(selected, threshold)

m1, m2, m3, m4, m5, m6 = st.columns(6)
for col, label, key in [
    (m1, "Accuracy", "accuracy"), (m2, "Precision", "precision"),
    (m3, "Recall", "recall"), (m4, "F1", "f1"),
    (m5, "ROC-AUC", "roc_auc"), (m6, "PR-AUC", "pr_auc")
]:
    col.metric(label, f"{metrics[key]:.3f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    selected.y_test, predictions, display_labels=["Stayed", "Left"],
    cmap="Blues", colorbar=False, ax=ax
)
ax.set_title("Confusion matrix at selected threshold")
st.pyplot(fig)
plt.close(fig)

roc_col, pr_col = st.columns(2)
with roc_col:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(selected.y_test, selected.probabilities, ax=ax)
    ax.set_title("ROC curve")
    st.pyplot(fig)
    plt.close(fig)
with pr_col:
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(selected.y_test, selected.probabilities, ax=ax)
    ax.set_title("Precision-recall curve")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Model explainability: Permutation Importance")
st.write(
    "Permutation importance measures how much the selected model's ROC-AUC "
    "changes when one input feature is shuffled. It describes model behaviour, "
    "not causation."
)
if st.button("Calculate permutation feature importance"):
    with st.spinner("Calculating permutation importance..."):
        importance = get_permutation_importance(selected)
    st.dataframe(importance.style.format({
        "importance_mean": "{:.4f}", "importance_std": "{:.4f}"
    }), width="stretch", hide_index=True)
    fig, ax = plt.subplots()
    top = importance.head(10).sort_values("importance_mean")
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Mean decrease in ROC-AUC after permutation")
    ax.set_title("Top input feature importance")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Section 4: SHAP – Global Model Explainability
# ---------------------------------------------------------------------------
st.divider()
st.header("SHAP – Global Model Explainability")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** is rooted in cooperative game theory.
It attributes each feature's contribution to moving a model's prediction away from
the average baseline prediction for the positive class (**Quit = 1**).

> **Responsible AI Reminder:** SHAP explains **model mechanics**, not human intentions.
> High feature attribution indicates strong predictive leverage in the model, **never real-world causality**.
""")

shap_model_name = st.selectbox(
    "Select model for SHAP global analysis",
    options=list(results.keys()),
    index=list(results.keys()).index(model_name),
    key="shap_global_model_selector",
    help="Switch between Random Forest (TreeExplainer) and Logistic Regression (LinearExplainer)."
)
shap_versions = ["Baseline"]
if shap_model_name in optimized_results:
    shap_versions.append("Optimized")
shap_version = st.selectbox(
    "SHAP model version",
    shap_versions,
    key=f"m25_shap_global_version_{shap_model_name.replace(' ', '_')}",
)
shap_selected = (
    optimized_results[shap_model_name]
    if shap_version == "Optimized"
    else results[shap_model_name]
)

# Check that test data exists
if shap_selected.X_test is None or len(shap_selected.X_test) == 0:
    st.error(f"Test dataset is unavailable for {shap_model_name}. Please re-train models.")
else:
    with st.spinner(f"Computing SHAP values for {shap_model_name} (cached)..."):
        try:
            shap_cache_key = (
                f"optimized:{data_fingerprint}:"
                f"{st.session_state['m25_optimization_run_version']}:{shap_model_name}"
                if shap_version == "Optimized"
                else f"baseline:{data_fingerprint}:{shap_model_name}"
            )
            shap_result: ShapExplanationResult = cached_global_shap(
                shap_cache_key,
                shap_selected.pipeline,
                shap_selected.X_test,
                max_samples=150,
            )
        except Exception as exc:
            st.error(f"Could not compute SHAP explanations for {shap_model_name}: {exc}")
            shap_result = None

    if shap_result is not None:
        importance_df = get_shap_feature_importance(shap_result)
        impact_summaries = get_feature_impact_summary(shap_result, top_n=6)

        tab1, tab2, tab3 = st.tabs([
            "📊 Feature Importance", "🌈 SHAP Summary Plot", "🧭 Feature Impact Direction"
        ])

        with tab1:
            st.markdown("#### 1. SHAP Feature Importance Ranking")
            st.write(
                "Features are ranked by **Mean Absolute SHAP Value** "
                "($\\operatorname{mean}(\\lvert \\text{SHAP} \\rvert)$). "
                "This metric quantifies how much each feature alters the model's predicted attrition output on average."
            )
            col_bar, col_table = st.columns([1.2, 1.0])
            with col_bar:
                fig_bar = plot_shap_importance_bar(
                    importance_df, top_n=10,
                    title=f"Top 10 Features by Mean |SHAP| ({shap_model_name})"
                )
                st.pyplot(fig_bar)
                plt.close(fig_bar)
            with col_table:
                st.markdown("**Top Feature Attribution Table**")
                st.dataframe(
                    importance_df.head(10).style.format({
                        "mean_abs_shap": "{:.4f}",
                        "mean_shap": "{:.4f}",
                        "importance_share": "{:.1%}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "• **mean_abs_shap**: Average absolute impact on prediction magnitude.<br>"
                    "• **mean_shap**: Average directional push across sampled profiles.<br>"
                    "• **importance_share**: Share of total attribution mass.",
                    unsafe_allow_html=True,
                )

        with tab2:
            st.markdown("#### 2. SHAP Beeswarm Summary Plot")
            st.write(
                "The summary plot reveals the **magnitude and direction** of impact for each feature across employee profiles. "
                "Each dot represents an employee record. Dots positioned to the **right** of the centerline increase "
                "the model's predicted probability of attrition, while dots to the **left** decrease it. "
                "Color denotes feature value (**Red = High**, **Blue = Low**)."
            )
            fig_summary = plot_shap_summary(
                shap_result, max_display=10,
                title=f"SHAP Summary Distribution ({shap_model_name})"
            )
            st.pyplot(fig_summary)
            plt.close(fig_summary)

        with tab3:
            st.markdown("#### 3. Feature Impact Direction")
            st.write(
                "This summary describes whether higher feature values generally push model predictions "
                "toward attrition or retention. **Remember: Correlation in model outputs does NOT imply causation.**"
            )
            for item in impact_summaries:
                with st.container():
                    st.markdown(
                        f"**{item['feature']}** — *{item['tendency']}* (Mean |SHAP|: `{item['mean_abs_shap']:.4f}`)"
                    )
                    st.caption(f"{item['impact_description']} (Statistical correlation: `{item['correlation']:.2f}`)")

# ---------------------------------------------------------------------------
# Section 5: SHAP – Individual Prediction Explanation
# ---------------------------------------------------------------------------
st.divider()
st.header("SHAP – Individual Prediction Explanation")
st.markdown("""
Examine how the selected model arrived at its prediction for an **individual employee profile**.
SHAP decomposes the final prediction into step-by-step contributions from each feature relative to the
dataset baseline $E[f(X)]$.

> **Responsible AI Notice:** These attributions reflect statistical weights inside the mathematical model.
> They are **NOT personal appraisals**, reasons for performance evaluation, or justifications for employment actions.
""")

local_model_name = st.selectbox(
    "Select model for individual explanation",
    options=list(results.keys()),
    index=list(results.keys()).index(shap_model_name),
    key="shap_local_model_selector",
)
local_versions = ["Baseline"]
if local_model_name in optimized_results:
    local_versions.append("Optimized")
local_version = st.selectbox(
    "Individual explanation model version",
    local_versions,
    key=f"m25_shap_local_version_{local_model_name.replace(' ', '_')}",
)
local_selected = (
    optimized_results[local_model_name]
    if local_version == "Optimized"
    else results[local_model_name]
)
X_pool = local_selected.X_test

if X_pool is None or len(X_pool) == 0:
    st.error("No test records available to explain.")
else:
    max_idx = min(len(X_pool) - 1, 300)
    sample_index = st.number_input(
        f"Select sample record index (0 to {max_idx})",
        min_value=0,
        max_value=max_idx,
        value=0,
        step=1,
        help="Choose an employee profile from the evaluation set to inspect its local SHAP waterfall explanation."
    )

    selected_record = X_pool.iloc[[sample_index]]
    actual_quit = local_selected.y_test.iloc[sample_index] if local_selected.y_test is not None else None

    with st.spinner("Generating local SHAP explanation..."):
        try:
            local_exp = explain_single_prediction(
                local_selected.pipeline,
                selected_record,
                background_df=X_pool,
            )
        except Exception as exc:
            st.error(f"Failed to generate local SHAP explanation: {exc}")
            local_exp = None

    if local_exp is not None:
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Predicted Outcome", local_exp["predicted_label"])
        p_col2.metric("Predicted Attrition Probability", f"{local_exp['attrition_probability'] * 100:.1f}%")
        scale_label = "Probability" if local_exp["output_space"] == "probability" else "Log-Odds"
        p_col3.metric(f"Baseline E[f(X)] ({scale_label})", f"{local_exp['base_value']:.3f}")
        actual_str = "Left (1)" if actual_quit == 1 else "Stayed (0)" if actual_quit == 0 else "Unknown"
        p_col4.metric("Actual Observed Status", actual_str)

        st.markdown("#### Sample Employee Profile Features")
        profile_df = pd.DataFrame([local_exp["raw_record"]])
        st.dataframe(profile_df, width="stretch", hide_index=True)

        wf_col, factor_col = st.columns([1.3, 1.0])
        with wf_col:
            st.markdown("#### Local SHAP Waterfall Plot")
            fig_wf = plot_shap_waterfall(
                local_exp["explanation"], max_display=10,
                title=f"Prediction Explanation for Record #{sample_index}"
            )
            st.pyplot(fig_wf)
            plt.close(fig_wf)

        with factor_col:
            st.markdown("#### Top Factors for this Prediction")
            st.markdown("**Factors Increasing Predicted Attrition Risk (Positive SHAP):**")
            pos_list = local_exp["positive_contributors"][:4]
            if pos_list:
                for c in pos_list:
                    st.write(f"• **{c['feature']}**: `+{c['shap_value']:.4f}`")
            else:
                st.caption("No positive contributors relative to baseline.")

            st.markdown("**Factors Decreasing Predicted Attrition Risk (Negative SHAP):**")
            neg_list = local_exp["negative_contributors"][:4]
            if neg_list:
                for c in neg_list:
                    st.write(f"• **{c['feature']}**: `{c['shap_value']:.4f}`")
            else:
                st.caption("No negative contributors relative to baseline.")

            with st.expander("Exact Mathematical Reconstruction (All 20 Features)"):
                st.markdown(f"**Output Space:** `{local_exp['output_space']}`")
                st.write(f"• Baseline $E[f(X)]$: `{local_exp['base_value']:.4f}`")
                st.write(f"• Top {len(pos_list)} positive displayed: `+{sum(c['shap_value'] for c in pos_list):.4f}`")
                st.write(f"• Top {len(neg_list)} negative displayed: `{sum(c['shap_value'] for c in neg_list):.4f}`")
                if local_exp['other_contributors_count'] > 0:
                    st.write(f"• Remaining {local_exp['other_contributors_count']} features net: `{local_exp['other_contributors_sum']:+.4f}`")
                st.write(f"• **Sum of all {len(local_exp['feature_names'])} features:** `{local_exp['sum_all_shap_values']:+.4f}`")
                st.write(f"• **Baseline + Sum = Reconstructed Output:** `{local_exp['reconstructed_output']:.4f}`")
                if local_exp['output_space'] == "log-odds (margin)":
                    st.write(f"• **Converted via Sigmoid:** $\\sigma({local_exp['reconstructed_output']:.4f}) = {local_exp['reconstructed_probability'] * 100:.1f}\\%$ (Matches predicted probability `{local_exp['attrition_probability'] * 100:.1f}\\%`)")
                else:
                    st.write(
                        f"• **Reconstructed Probability:** "
                        f"${local_exp['reconstructed_probability'] * 100:.1f}\\%$ "
                        f"(Matches predicted probability "
                        f"`{local_exp['attrition_probability'] * 100:.1f}%`)."
                    )

# ---------------------------------------------------------------------------
# Section 8: Responsible Use and Data Quality Details
# ---------------------------------------------------------------------------
with st.expander("Data quality details"):
    st.json(cleaning_report)
    st.write("Missing values by column")
    st.dataframe(cleaned_data.isna().sum().rename("missing_count"), width="stretch")

with st.expander("Responsible-use notes, SHAP governance & limitations"):
    st.markdown("""
    ### Responsible AI & Explainability Principles
    - **Model Behaviour vs. Human Intent:** SHAP values explain how the algorithm computes its mathematical outputs; they do **not** reflect why an employee actually decides to stay or leave.
    - **Correlation is Not Causation:** A high positive or negative SHAP value indicates statistical association in the historical training set. Changing an employee's feature (e.g. assigning more projects) will not necessarily alter attrition risk in reality.
    - **Historical Biases:** Machine learning models learn patterns from past organizational data, which may encode historical inequities, organizational friction, or unmeasured external factors (such as commute distance or management culture).
    - **Prohibition on Automated Adverse Actions:** Under no circumstances should predictions or SHAP feature attributions be used to rank employees, make promotion or termination decisions, or penalize individuals.
    - **Privacy and Data Protection:** Employee data contains sensitive workforce information. Production deployments must enforce access controls, anonymization, and regulatory compliance (e.g. GDPR, EEOC).
    - **Educational Scope:** This dashboard is an analytical prototype designed to demonstrate Explainable AI techniques, not a production HR decision system.
    """)
