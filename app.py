import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.attrition import (
    clean_dataset,
    compare_baseline_and_optimized,
    get_candidate_models,
    get_hyperparameter_search_spaces,
    get_permutation_importance,
    load_dataset,
    optimize_models,
    save_optimized_model,
    split_dataset,
    threshold_metrics,
    train_and_compare,
    validate_dataset_frame,
)
from src.deployment import get_deployment_config
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
from src.model_monitoring import (
    build_aggregate_export as build_monitoring_aggregate_export,
)
from src.model_monitoring import (
    build_feature_drift_csv,
    build_monitoring_run_signature,
    calculate_categorical_drift_by_feature,
    calculate_data_quality,
    calculate_delayed_performance,
    calculate_numerical_drift,
    calculate_prediction_drift,
    create_reference_metadata,
    evaluate_monitoring_status,
    load_monitoring_history,
    save_monitoring_history,
    save_monitoring_report,
    save_reference_metadata,
    validate_monitoring_data,
)
from src.ui_theme import (
    inject_saas_theme,
    render_download_link,
    render_hero_header,
    render_profile_grid,
    render_section_header,
    render_status_banner,
    style_matplotlib_figure,
    style_plotly_figure,
)
from src.workforce_analytics import (
    aggregate_export_to_csv,
    apply_workforce_filters,
    build_aggregate_export,
    department_attrition,
    highest_attrition_group,
    salary_attrition,
    tenure_attrition,
)

try:
    DEPLOYMENT_CONFIG = get_deployment_config()
except ValueError as exc:
    DEPLOYMENT_CONFIG = None
    DEPLOYMENT_CONFIG_ERROR = str(exc)
else:
    DEPLOYMENT_CONFIG_ERROR = None

logging.basicConfig(
    level=getattr(logging, DEPLOYMENT_CONFIG["log_level"] if DEPLOYMENT_CONFIG else "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("workforce_insights")

st.set_page_config(page_title="Workforce Insights | Attrition Lab", page_icon="📊", layout="wide")
inject_saas_theme()

if DEPLOYMENT_CONFIG_ERROR:
    st.error(f"Invalid deployment configuration: {DEPLOYMENT_CONFIG_ERROR}")
    st.stop()

if not DEPLOYMENT_CONFIG["data_path"].exists():
    LOGGER.error("Configured data file not found: %s", DEPLOYMENT_CONFIG["data_path"])
    st.error(
        f"Configured dataset is unavailable: {DEPLOYMENT_CONFIG['data_path']}"
    )
    st.stop()


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


NAV_OPTIONS = [
    "All Sections (Full Dashboard)",
    "1. Overview & Workforce Analytics",
    "2. Model Comparison & Tuning",
    "3. Evaluation & SHAP Explainability",
    "4. Model Monitoring & Data Drift",
]

with st.sidebar:
    st.markdown("### Workspace Navigation")
    active_view = st.radio(
        "Select workspace",
        NAV_OPTIONS,
        index=0,
        key="ui_workspace_nav",
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Dataset Source")
    DEFAULT_DATA = DEPLOYMENT_CONFIG["data_path"]
    uploaded = st.file_uploader("Upload an employee CSV", type=["csv"])

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
    st.session_state["m26_monitoring_report"] = None
    st.session_state["m26_monitoring_status"] = ""
    try:
        st.session_state["m26_monitoring_history"] = load_monitoring_history(
            artifact_dir=DEPLOYMENT_CONFIG["monitoring_dir"],
        )
    except Exception:
        st.session_state["m26_monitoring_history"] = []

# Sidebar filters affect only the aggregate workforce analytics section.
with st.sidebar:
    st.divider()
    st.markdown("### Workforce Analytics Filters")

    department_options = (
        sorted(cleaned_data["department"].dropna().astype(str).unique().tolist())
        if "department" in cleaned_data.columns
        else []
    )
    salary_options = (
        sorted(cleaned_data["salary"].dropna().astype(str).unique().tolist())
        if "salary" in cleaned_data.columns
        else []
    )
    selected_departments = st.multiselect("Department", department_options)
    selected_salaries = st.multiselect("Salary", salary_options)

    tenure_values = pd.to_numeric(cleaned_data.get("time_spend_company"), errors="coerce").dropna()
    if len(tenure_values):
        tenure_min = int(tenure_values.min())
        tenure_max = int(max(tenure_values.max(), tenure_min))
        selected_tenure = st.slider(
            "Years at company",
            min_value=tenure_min,
            max_value=tenure_max,
            value=(tenure_min, tenure_max),
        )
    else:
        selected_tenure = None

    work_accident_options = (
        sorted(cleaned_data["Work_accident"].dropna().astype(str).unique().tolist())
        if "Work_accident" in cleaned_data.columns
        else []
    )
    promotion_options = (
        sorted(cleaned_data["promotion_last_5years"].dropna().astype(str).unique().tolist())
        if "promotion_last_5years" in cleaned_data.columns
        else []
    )
    selected_work_accidents = st.multiselect("Work accident", work_accident_options)
    selected_promotions = st.multiselect("Promotion in last 5 years", promotion_options)
    min_group_size = st.slider(
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
optimized_results = st.session_state.get("m25_optimization_results", {})
optimized_comparison = st.session_state.get("m25_optimization_comparison", pd.DataFrame())

env_badge = f"ENV: {DEPLOYMENT_CONFIG['app_env'].upper()} · Educational Prototype"
render_hero_header(
    "Workforce Insights",
    "Employee Attrition Prediction, Responsible Workforce Analytics, SHAP Explainability & Drift Monitoring",
    env_label=env_badge,
)
st.warning(
    "Educational prototype only. Do not use predictions to rank, penalize, hire, "
    "promote, or terminate employees. Use aggregated insights to explore retention "
    "questions, with human review and appropriate privacy safeguards."
)

show_all = active_view == NAV_OPTIONS[0]
show_overview = show_all or active_view == NAV_OPTIONS[1]
show_tuning = show_all or active_view == NAV_OPTIONS[2]
show_eval_shap = show_all or active_view == NAV_OPTIONS[3]
show_monitoring = show_all or active_view == NAV_OPTIONS[4]

# ---------------------------------------------------------------------------
# Section 1: Dataset Overview & Baseline Summary
# ---------------------------------------------------------------------------
if show_overview:
    render_section_header(
        "Executive Summary",
        "Dataset overview",
        "Deduplicated workforce baseline, target balance, and baseline classifier benchmark.",
    )
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

    left, right = st.columns([1.0, 1.35])
    with left:
        st.markdown("**Target distribution**")
        counts_df = (
            cleaned_data["quit"]
            .value_counts()
            .rename(index={0: "Stayed", 1: "Left"})
            .reset_index()
        )
        counts_df.columns = ["Outcome", "Employees"]
        fig_target = px.bar(
            counts_df,
            x="Outcome",
            y="Employees",
            color="Outcome",
            text="Employees",
            color_discrete_map={"Stayed": "#38bdf8", "Left": "#f43f5e"},
            title="Observed Retention vs. Attrition",
        )
        fig_target.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_target.update_layout(showlegend=False)
        style_plotly_figure(fig_target, height=340)
        st.plotly_chart(fig_target, width="stretch")

    with right:
        st.markdown("**Model comparison**")
        st.dataframe(
            comparison.style.format({
                col: "{:.3f}" for col in comparison.columns if col != "model"
            }),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Baseline models are ranked by 3-fold training CV F1 mean. "
            "Held-out test metrics are evaluated on the shared stratified 20% split."
        )

# ---------------------------------------------------------------------------
# Section 2: Interactive Workforce Analytics
# ---------------------------------------------------------------------------
if show_overview:
    st.divider()
    render_section_header(
        "Workforce Insights",
        "Interactive Workforce Analytics",
        "Descriptive historical turnover across departments, salary bands, tenure, and satisfaction.",
    )
    st.info(
        "This dashboard is an educational workforce analytics prototype. "
        "Observed historical patterns and model predictions do not establish causation "
        "or determine an individual's future behavior. Do not use these outputs to rank, "
        "penalize, hire, promote, or terminate employees. Any organizational "
        "interpretation requires appropriate privacy safeguards and human review."
    )

    if analytics_data.empty:
        st.warning("No records match the selected analytics filters. Broaden sidebar filters to view charts.")
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
                    color_discrete_map={"Stayed": "#38bdf8", "Left": "#f43f5e", "Unknown": "#64748b"},
                )
                fig.update_layout(xaxis_tickangle=-35)
                style_plotly_figure(fig)
                st.plotly_chart(fig, width="stretch")

            if not department_summary.empty:
                dept_chart = department_summary.copy()
                dept_chart["attrition_percent"] = dept_chart["attrition_rate"] * 100
                fig = px.bar(
                    dept_chart,
                    x="group",
                    y="attrition_percent",
                    color="reliable",
                    color_discrete_map={True: "#f43f5e", False: "#64748b"},
                    title="Observed Attrition Rate by Department",
                    labels={"group": "Department", "attrition_percent": "Attrition rate (%)", "reliable": "Reliable sample"},
                    hover_data=["employee_count", "left_count", "stayed_count"],
                    text=dept_chart["attrition_percent"].round(1),
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(yaxis_ticksuffix="%", xaxis_tickangle=-35)
                style_plotly_figure(fig)
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
                    color_discrete_map={"Stayed": "#38bdf8", "Left": "#f43f5e", "Unknown": "#64748b"},
                )
                style_plotly_figure(fig)
                st.plotly_chart(fig, width="stretch")

            if not salary_summary.empty:
                salary_chart = salary_summary.copy()
                salary_chart["attrition_percent"] = salary_chart["attrition_rate"] * 100
                fig = px.bar(
                    salary_chart,
                    x="group",
                    y="attrition_percent",
                    color="reliable",
                    color_discrete_map={True: "#f59e0b", False: "#64748b"},
                    title="Observed Attrition Rate by Salary Category",
                    labels={"group": "Salary category", "attrition_percent": "Attrition rate (%)", "reliable": "Reliable sample"},
                    hover_data=["employee_count", "left_count", "stayed_count"],
                    text=salary_chart["attrition_percent"].round(1),
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(yaxis_ticksuffix="%")
                style_plotly_figure(fig)
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
                    color_discrete_map={"Stayed": "#38bdf8", "Left": "#f43f5e", "Unknown": "#64748b"},
                )
                style_plotly_figure(fig)
                st.plotly_chart(fig, width="stretch")

            if not tenure_summary.empty:
                tenure_rate_chart = tenure_summary.copy()
                tenure_rate_chart["attrition_percent"] = tenure_rate_chart["attrition_rate"] * 100
                fig = px.bar(
                    tenure_rate_chart,
                    x="group",
                    y="attrition_percent",
                    color="reliable",
                    color_discrete_map={True: "#10b981", False: "#64748b"},
                    title="Observed Attrition Rate by Tenure Group",
                    labels={"group": "Tenure group", "attrition_percent": "Attrition rate (%)", "reliable": "Reliable sample"},
                    hover_data=["employee_count", "left_count", "stayed_count"],
                    text=tenure_rate_chart["attrition_percent"].round(1),
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(yaxis_ticksuffix="%")
                style_plotly_figure(fig)
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
                    opacity=0.68,
                    nbins=20,
                    title="Satisfaction-Level Distribution by Observed Outcome",
                    labels={"satisfaction_level": "Satisfaction level", "count": "Employee count"},
                    color_discrete_map={"Stayed": "#38bdf8", "Left": "#f43f5e", "Unknown": "#64748b"},
                )
                style_plotly_figure(fig)
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
                    color_discrete_map={"Stayed": "#38bdf8", "Left": "#f43f5e", "Unknown": "#64748b"},
                    opacity=0.62,
                )
                style_plotly_figure(fig)
                st.plotly_chart(fig, width="stretch")
            if {"last_evaluation", "quit"}.issubset(satisfaction_chart.columns):
                fig = px.box(
                    satisfaction_chart,
                    x="Observed outcome",
                    y="last_evaluation",
                    color="Observed outcome",
                    title="Performance Evaluation Distribution by Observed Outcome",
                    labels={"last_evaluation": "Last evaluation", "Observed outcome": "Observed outcome"},
                    color_discrete_map={"Stayed": "#38bdf8", "Left": "#f43f5e", "Unknown": "#64748b"},
                )
                style_plotly_figure(fig)
                st.plotly_chart(fig, width="stretch")

        aggregate_export = build_aggregate_export(
            department_summary,
            salary_summary,
            tenure_summary,
        )
        csv_payload = aggregate_export_to_csv(aggregate_export)
        render_download_link(
            csv_payload,
            "workforce_analytics_aggregate.csv",
            "Download filtered aggregate analytics CSV",
            mime_type="text/csv",
        )
        st.caption(
            "The export excludes individual employee records and personal identifiers."
        )

# ---------------------------------------------------------------------------
# Section 3: Milestone 2.5 — Model Comparison & Bounded Optimization
# ---------------------------------------------------------------------------
if show_tuning:
    st.divider()
    render_section_header(
        "Model Benchmarking & Tuning",
        "Model Comparison & Performance Evaluation",
        "User-triggered bounded RandomizedSearchCV (max 6 candidates, 3 CV folds, n_jobs=1) isolated from baseline artifacts.",
    )
    st.info(
        "Optimization runs only when you submit this form. It uses at most 6 "
        "randomized candidates per selected model, 3 stratified training folds, "
        "random state 42, and one job. All preprocessing is fitted within each "
        "training fold. The shared held-out test split is used only for final "
        "evaluation and is never used to select parameters."
    )
    with st.form("m25_optimization_form"):
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            optimization_choice = st.selectbox(
                "Models to optimize",
                ["Logistic Regression", "Random Forest", "Both models"],
                key="m25_optimization_model_choice",
            )
        with opt_col2:
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
                    if not DEPLOYMENT_CONFIG["allow_artifact_writes"]:
                        raise RuntimeError(
                            "Artifact writes are disabled by ALLOW_ARTIFACT_WRITES=false."
                        )
                    best_optimized_name = max(
                        optimized_results,
                        key=lambda name: optimized_results[name].best_cv_score,
                    )
                    model_path, report_path = save_optimized_model(
                        optimized_results[best_optimized_name],
                        cleaning_report,
                        artifact_dir=DEPLOYMENT_CONFIG["artifact_dir"],
                    )
                    st.session_state["m25_optimization_results"] = optimized_results
                    st.session_state["m25_optimization_failures"] = optimization_failures
                    st.session_state["m25_optimization_comparison"] = optimized_comparison
                    st.session_state["m25_optimization_cleaning_report"] = cleaning_report
                    st.session_state["m25_optimization_run_id"] = (
                        st.session_state.get("m25_optimization_run_id", 0) + 1
                    )
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

    optimized_results = st.session_state.get("m25_optimization_results", {})
    optimized_comparison = st.session_state.get("m25_optimization_comparison", pd.DataFrame())

    optimization_status = st.session_state.get("m25_optimization_status", "")
    if optimization_status:
        if "failed" in optimization_status.lower() or "no selected model" in optimization_status.lower():
            st.warning(optimization_status)
        else:
            st.success(optimization_status)
        saved_artifacts = st.session_state.get("m25_optimized_artifacts", {})
        if saved_artifacts:
            st.caption(
                f"Optimized artifacts: `{saved_artifacts['model']}` and "
                f"`{saved_artifacts['report']}`. Baseline artifacts are not modified."
            )

    optimization_failures = st.session_state.get("m25_optimization_failures", {})
    for failed_name, failure_message in optimization_failures.items():
        st.warning(f"{failed_name} optimization was not available: {failure_message}")

    if optimized_results and not optimized_comparison.empty:
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

        opt_tab1, opt_tab2, opt_tab3 = st.tabs([
            "Held-out & CV Comparison",
            "Visual Metric Charts",
            "Selected Hyperparameters",
        ])

        with opt_tab1:
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
                display_row["Optimization + refit seconds"] = comparison_row["optimized_optimization_time_sec"]
                display_row["Baseline prediction seconds"] = comparison_row["baseline_predict_time_sec"]
                display_row["Optimized prediction seconds"] = comparison_row["optimized_predict_time_sec"]
                comparison_rows.append(display_row)
            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(
                comp_df.style.format({
                    column: "{:.3f}"
                    for column in comp_df.columns
                    if column != "Model" and "change" not in column
                }),
                width="stretch",
                hide_index=True,
            )

        with opt_tab2:
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
                    color_discrete_map={"Baseline": "#64748b", "Optimized": "#38bdf8"},
                    title="Baseline vs. optimized held-out metrics",
                )
                style_plotly_figure(fig)
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
                    color_discrete_map={"Baseline": "#64748b", "Optimized": "#10b981"},
                    title="Training-only cross-validation means",
                )
                style_plotly_figure(fig)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "CV F1 standard deviations are reported in the comparison table. "
                    "The held-out test set did not participate in CV or parameter selection."
                )

        with opt_tab3:
            st.write("Selected hyperparameters for best cross-validated model:")
            st.json(best_optimized.best_params)
    else:
        st.caption(
            "No optimized results are available for this dataset yet. Submit the form "
            "above to run optimization; dashboard reruns do not launch it automatically."
        )

# ---------------------------------------------------------------------------
# Section 4: Evaluation, Threshold Explorer & SHAP Explainability
# ---------------------------------------------------------------------------
model_name = best_name
if show_eval_shap:
    st.divider()
    render_section_header(
        "Model Evaluation",
        "Evaluation and threshold explorer",
        "Inspect classification metrics, confusion matrix, ROC/PR curves, and permutation importance.",
    )
    eval_c1, eval_c2, eval_c3 = st.columns([1.2, 1.0, 1.4])
    with eval_c1:
        model_name = st.selectbox(
            "Choose a model",
            list(results.keys()),
            index=list(results.keys()).index(best_name),
        )
    evaluation_versions = ["Baseline"]
    if model_name in optimized_results:
        evaluation_versions.append("Optimized")
    with eval_c2:
        evaluation_version = st.selectbox(
            "Model version for evaluation",
            evaluation_versions,
            key=f"m25_evaluation_version_{model_name.replace(' ', '_')}",
        )
    with eval_c3:
        threshold = st.slider(
            "Decision threshold for the demonstration metrics",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Changing this threshold changes the classification metrics. It does not retrain the model.",
        )

    selected = (
        optimized_results[model_name]
        if evaluation_version == "Optimized"
        else results[model_name]
    )
    metrics, predictions = threshold_metrics(selected, threshold)

    row1_m1, row1_m2, row1_m3 = st.columns(3)
    row1_m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    row1_m2.metric("Precision", f"{metrics['precision']:.3f}")
    row1_m3.metric("Recall", f"{metrics['recall']:.3f}")

    row2_m4, row2_m5, row2_m6 = st.columns(3)
    row2_m4.metric("F1", f"{metrics['f1']:.3f}")
    row2_m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    row2_m6.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    cm_col, roc_col, pr_col = st.columns(3)
    with cm_col:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            selected.y_test, predictions, display_labels=["Stayed", "Left"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title("Confusion matrix at threshold")
        style_matplotlib_figure(fig)
        st.pyplot(fig)
        plt.close(fig)
    with roc_col:
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(selected.y_test, selected.probabilities, ax=ax)
        ax.set_title("ROC curve")
        style_matplotlib_figure(fig)
        st.pyplot(fig)
        plt.close(fig)
    with pr_col:
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(selected.y_test, selected.probabilities, ax=ax)
        ax.set_title("Precision-recall curve")
        style_matplotlib_figure(fig)
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
        perm_c1, perm_c2 = st.columns([1.0, 1.2])
        with perm_c1:
            st.dataframe(importance.style.format({
                "importance_mean": "{:.4f}", "importance_std": "{:.4f}",
            }), width="stretch", hide_index=True)
        with perm_c2:
            fig, ax = plt.subplots()
            top = importance.head(10).sort_values("importance_mean")
            ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#38bdf8")
            ax.set_xlabel("Mean decrease in ROC-AUC after permutation")
            ax.set_title("Top input feature importance")
            style_matplotlib_figure(fig)
            st.pyplot(fig)
            plt.close(fig)

    # -----------------------------------------------------------------------
    # SHAP – Global Model Explainability
    # -----------------------------------------------------------------------
    st.divider()
    render_section_header(
        "Explainable AI · Global",
        "SHAP – Global Model Explainability",
        "Co-operative game-theoretic attribution of feature impact across sampled held-out profiles (Quit = 1).",
    )
    st.markdown("""
> **Responsible AI Reminder:** SHAP explains **model mechanics**, not human intentions.
> High feature attribution indicates strong predictive leverage in the model, **never real-world causality**.
""")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        shap_model_name = st.selectbox(
            "Select model for SHAP global analysis",
            options=list(results.keys()),
            index=list(results.keys()).index(model_name),
            key="shap_global_model_selector",
            help="Switch between Random Forest (TreeExplainer) and Logistic Regression (LinearExplainer).",
        )
    shap_versions = ["Baseline"]
    if shap_model_name in optimized_results:
        shap_versions.append("Optimized")
    with g_col2:
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
                "📊 Feature Importance", "🌈 SHAP Summary Plot", "🧭 Feature Impact Direction",
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
                        title=f"Top 10 Features by Mean |SHAP| ({shap_model_name})",
                    )
                    style_matplotlib_figure(fig_bar)
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
                    title=f"SHAP Summary Distribution ({shap_model_name})",
                )
                style_matplotlib_figure(fig_summary)
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

    # -----------------------------------------------------------------------
    # SHAP – Individual Prediction Explanation
    # -----------------------------------------------------------------------
    st.divider()
    render_section_header(
        "Explainable AI · Local",
        "SHAP – Individual Prediction Explanation",
        "Inspect step-by-step feature contributions relative to baseline E[f(X)] for a single evaluation record.",
    )
    st.markdown("""
> **Responsible AI Notice:** These attributions reflect statistical weights inside the mathematical model.
> They are **NOT personal appraisals**, reasons for performance evaluation, or justifications for employment actions.
""")

    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        local_model_name = st.selectbox(
            "Select model for individual explanation",
            options=list(results.keys()),
            index=list(results.keys()).index(shap_model_name),
            key="shap_local_model_selector",
        )
    local_versions = ["Baseline"]
    if local_model_name in optimized_results:
        local_versions.append("Optimized")
    with l_col2:
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
        with l_col3:
            sample_index = st.number_input(
                f"Select sample record index (0 to {max_idx})",
                min_value=0,
                max_value=max_idx,
                value=0,
                step=1,
                help="Choose an employee profile from the evaluation set to inspect its local SHAP waterfall explanation.",
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
            render_profile_grid(local_exp["raw_record"])

            wf_col, factor_col = st.columns([1.3, 1.0])
            with wf_col:
                st.markdown("#### Local SHAP Waterfall Plot")
                fig_wf = plot_shap_waterfall(
                    local_exp["explanation"], max_display=10,
                    title=f"Prediction Explanation for Record #{sample_index}",
                )
                style_matplotlib_figure(fig_wf)
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
                    if local_exp["other_contributors_count"] > 0:
                        st.write(f"• Remaining {local_exp['other_contributors_count']} features net: `{local_exp['other_contributors_sum']:+.4f}`")
                    st.write(f"• **Sum of all {len(local_exp['feature_names'])} features:** `{local_exp['sum_all_shap_values']:+.4f}`")
                    st.write(f"• **Baseline + Sum = Reconstructed Output:** `{local_exp['reconstructed_output']:.4f}`")
                    if local_exp["output_space"] == "log-odds (margin)":
                        st.write(
                            f"• **Converted via Sigmoid:** $\\sigma({local_exp['reconstructed_output']:.4f}) = "
                            f"{local_exp['reconstructed_probability'] * 100:.1f}\\%$ "
                            f"(Matches predicted probability `{local_exp['attrition_probability'] * 100:.1f}\\%`)"
                        )
                    else:
                        st.write(
                            f"• **Reconstructed Probability:** "
                            f"${local_exp['reconstructed_probability'] * 100:.1f}\\%$ "
                            f"(Matches predicted probability "
                            f"`{local_exp['attrition_probability'] * 100:.1f}%`)."
                        )

# ---------------------------------------------------------------------------
# Section 5: Milestone 2.6 — Model Monitoring & Data Drift
# ---------------------------------------------------------------------------
if show_monitoring:
    st.divider()
    render_section_header(
        "Observability & Drift",
        "Model Monitoring & Data Drift",
        "Compare incoming batches against the training reference distribution with aggregate-only reporting.",
    )
    st.info(
        "Monitoring is educational and uses aggregate statistics only. "
        "Drift does not prove bias, reduced model accuracy, or causation."
    )

    mon_c1, mon_c2 = st.columns([1.2, 1.3])
    with mon_c1:
        monitoring_upload = st.file_uploader(
            "Upload a current monitoring batch (CSV or TXT)",
            type=["csv", "txt"],
            key="m26_monitoring_file",
        )
    with mon_c2:
        available_monitoring_models = list(results.keys())
        m_sub1, m_sub2 = st.columns(2)
        with m_sub1:
            monitoring_model_name = st.selectbox(
                "Monitoring model",
                available_monitoring_models,
                index=available_monitoring_models.index(best_name),
                key="m26_monitoring_model_name",
            )
        monitoring_versions = ["Baseline"]
        if monitoring_model_name in optimized_results:
            monitoring_versions.append("Optimized")
        with m_sub2:
            monitoring_version = st.selectbox(
                "Monitoring model version",
                monitoring_versions,
                key=f"m26_monitoring_version_{monitoring_model_name.replace(' ', '_')}",
            )
        monitoring_threshold = st.slider(
            "Prediction positive threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            key="m26_monitoring_threshold",
        )

    monitoring_selected = (
        optimized_results[monitoring_model_name]
        if monitoring_version == "Optimized"
        else results[monitoring_model_name]
    )

    with st.form("m26_monitoring_form"):
        st.caption(
            "Monitoring executes only when you submit this form. "
            "Ordinary Streamlit reruns will not re-compute drift or predictions."
        )
        run_monitoring = st.form_submit_button("Run monitoring for uploaded batch")

    if run_monitoring:
        if monitoring_upload is None:
            st.session_state["m26_monitoring_status"] = (
                "Monitoring was not run: upload a CSV or TXT batch first."
            )
        else:
            try:
                if not DEPLOYMENT_CONFIG["allow_artifact_writes"]:
                    raise RuntimeError(
                        "Artifact writes are disabled by ALLOW_ARTIFACT_WRITES=false."
                    )
                suffix = Path(monitoring_upload.name).suffix.lower()
                if suffix not in {".csv", ".txt"}:
                    raise ValueError("Monitoring supports CSV and TXT files only.")
                monitoring_data = pd.read_csv(monitoring_upload)
                if monitoring_data.empty:
                    raise ValueError("Monitoring data contains no rows.")

                X_train, _, _, _, _ = split_dataset(data, random_state=42)
                reference_metadata = create_reference_metadata(X_train)
                validation = validate_monitoring_data(
                    monitoring_data,
                    reference_metadata,
                )

                numerical_drift = calculate_numerical_drift(
                    X_train,
                    monitoring_data,
                    alpha=0.05,
                )
                categorical_drift = calculate_categorical_drift_by_feature(
                    X_train,
                    monitoring_data,
                )
                data_quality = calculate_data_quality(
                    X_train,
                    monitoring_data,
                )
                prediction_drift = calculate_prediction_drift(
                    monitoring_selected,
                    monitoring_data,
                    threshold=monitoring_threshold,
                )
                performance = calculate_delayed_performance(
                    monitoring_selected,
                    monitoring_data,
                    threshold=monitoring_threshold,
                )
                status_report = evaluate_monitoring_status(
                    data_quality=data_quality,
                    numerical_drift=numerical_drift,
                    categorical_drift=categorical_drift,
                    prediction_drift=prediction_drift,
                    performance=performance,
                )

                monitoring_report = {
                    "model": monitoring_model_name,
                    "model_version": monitoring_version,
                    "model_identity": (
                        f"{monitoring_version.lower()}:{monitoring_model_name}"
                    ),
                    "threshold": monitoring_threshold,
                    "reference_fingerprint": reference_metadata["fingerprint"],
                    "monitoring_fingerprint": joblib.hash(monitoring_data),
                    "validation": validation,
                    "data_quality": data_quality,
                    "numerical_drift": numerical_drift,
                    "categorical_drift": categorical_drift,
                    "prediction_distribution": prediction_drift,
                    "delayed_performance": performance,
                    "status": status_report,
                }
                monitoring_report["run_signature"] = build_monitoring_run_signature(
                    monitoring_report
                )

                save_reference_metadata(
                    reference_metadata,
                    artifact_dir=DEPLOYMENT_CONFIG["monitoring_dir"],
                )
                report_path = save_monitoring_report(
                    monitoring_report,
                    artifact_dir=DEPLOYMENT_CONFIG["monitoring_dir"],
                )
                history = st.session_state.get("m26_monitoring_history", []).copy()
                signature = monitoring_report["run_signature"]
                if not any(
                    row.get("run_signature") == signature
                    or (
                        "run_signature" not in row
                        and build_monitoring_run_signature(row) == signature
                    )
                    for row in history
                ):
                    history.append(monitoring_report)
                save_monitoring_history(
                    history,
                    artifact_dir=DEPLOYMENT_CONFIG["monitoring_dir"],
                )
                st.session_state["m26_monitoring_history"] = history[-20:]
                st.session_state["m26_monitoring_report"] = monitoring_report
                st.session_state["m26_monitoring_status"] = (
                    f"Monitoring completed. Aggregate report saved to `{report_path}`."
                )
            except Exception as exc:
                st.session_state["m26_monitoring_status"] = (
                    f"Monitoring failed: {exc}"
                )

    monitoring_status = st.session_state.get("m26_monitoring_status", "")
    if monitoring_status:
        if monitoring_status.startswith("Monitoring failed") or "not run" in monitoring_status:
            st.warning(monitoring_status)
        else:
            st.success(monitoring_status)

    monitoring_report = st.session_state.get("m26_monitoring_report")
    if monitoring_report:
        status_value = monitoring_report["status"]["status"]
        render_status_banner(
            f"Monitoring status: {status_value}",
            status_value,
            monitoring_report["status"].get("reasons", []),
        )

        quality = monitoring_report["data_quality"]
        drifted_num = sum(
            1
            for item in monitoring_report["numerical_drift"].values()
            if item.get("psi_level") in {"Moderate", "Significant"}
        )
        drifted_cat = sum(
            1
            for item in monitoring_report["categorical_drift"].values()
            if item.get("unseen_categories") or item.get("disappeared_categories") or item.get("max_proportion_shift", 0) >= 0.10
        )

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows monitored", f"{monitoring_report['validation']['row_count']:,}")
        q2.metric("Duplicate rows", f"{quality['duplicate_count']:,}")
        q3.metric("Data-quality status", quality["status"])
        q4.metric("Features with drift signals", f"{drifted_num + drifted_cat}")

        mon_tab1, mon_tab2, mon_tab3, mon_tab4, mon_tab5 = st.tabs([
            "Numerical Drift",
            "Categorical Drift",
            "Data Quality & Schema",
            "Predictions & Performance",
            "History & Exports",
        ])

        with mon_tab1:
            numerical_rows = [
                {"Feature": feature, **summary}
                for feature, summary in monitoring_report["numerical_drift"].items()
            ]
            if numerical_rows:
                num_df = pd.DataFrame(numerical_rows)
                fig_psi = px.bar(
                    num_df,
                    x="Feature",
                    y="psi",
                    color="psi_level",
                    color_discrete_map={
                        "Low": "#10b981",
                        "Moderate": "#f59e0b",
                        "Significant": "#ef4444",
                    },
                    title="Population Stability Index (PSI) by Numerical Feature",
                    labels={"psi": "PSI", "psi_level": "Heuristic level"},
                    text=num_df["psi"].round(3),
                )
                fig_psi.add_hline(y=0.10, line_dash="dash", line_color="#f59e0b", annotation_text="0.10 Moderate")
                fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#ef4444", annotation_text="0.25 Significant")
                fig_psi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                style_plotly_figure(fig_psi)
                st.plotly_chart(fig_psi, width="stretch")

                st.dataframe(
                    num_df.style.format(
                        {"psi": "{:.4f}", "ks_statistic": "{:.4f}", "ks_p_value": "{:.4f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "PSI thresholds (<0.10 Low, 0.10–0.25 Moderate, >0.25 Significant) are configurable heuristics. "
                    "KS p-values are sensitive to sample size."
                )

        with mon_tab2:
            categorical_rows = [
                {
                    "Feature": feature,
                    "Maximum proportion shift": summary["max_proportion_shift"],
                    "Unseen categories": ", ".join(summary["unseen_categories"]) or "None",
                    "Disappeared categories": ", ".join(summary["disappeared_categories"]) or "None",
                }
                for feature, summary in monitoring_report["categorical_drift"].items()
            ]
            if categorical_rows:
                cat_df = pd.DataFrame(categorical_rows)
                fig_cat = px.bar(
                    cat_df,
                    x="Feature",
                    y="Maximum proportion shift",
                    title="Maximum Category Proportion Shift by Feature",
                    text=cat_df["Maximum proportion shift"].round(3),
                    color_discrete_sequence=["#38bdf8"],
                )
                fig_cat.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                style_plotly_figure(fig_cat)
                st.plotly_chart(fig_cat, width="stretch")
                st.dataframe(cat_df, width="stretch", hide_index=True)

        with mon_tab3:
            val_info = monitoring_report["validation"]
            d1, d2, d3 = st.columns(3)
            d1.metric("Schema valid", "Yes" if val_info.get("valid") else "No")
            d2.metric("Target (`quit`) status", str(val_info.get("quit_status", "unknown")).title())
            d3.metric("Missing columns", len(val_info.get("missing_columns", [])))
            st.json({
                "validation": val_info,
                "data_quality": quality,
            })

        with mon_tab4:
            prediction = monitoring_report["prediction_distribution"]
            st.caption("These values are model predictions, not confirmed employee outcomes.")
            p1, p2, p3 = st.columns(3)
            p1.metric("Mean prediction", f"{prediction['mean']:.3f}")
            p2.metric("Median prediction", f"{prediction['median']:.3f}")
            p3.metric("Predicted-positive rate", f"{prediction['predicted_positive_rate']:.1%}")

            performance = monitoring_report["delayed_performance"]
            st.subheader("Delayed ground-truth performance")
            if performance["status"] == "Available":
                st.dataframe(
                    pd.DataFrame([performance["metrics"]]).style.format("{:.3f}"),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(
                    "Model performance cannot be verified until valid binary "
                    "ground-truth outcomes are available."
                )

        with mon_tab5:
            drift_payload = build_feature_drift_csv(
                monitoring_report["numerical_drift"],
                monitoring_report["categorical_drift"],
            )
            agg_json_payload = json.dumps(
                build_monitoring_aggregate_export(monitoring_report),
                indent=2,
            ).encode("utf-8")

            render_download_link(
                drift_payload,
                "monitoring_feature_drift.csv",
                "Download aggregate feature drift CSV",
                mime_type="text/csv",
            )
            render_download_link(
                agg_json_payload,
                "monitoring_aggregate_report.json",
                "Download aggregate monitoring JSON",
                mime_type="application/json",
            )
            st.caption("All monitoring exports contain aggregate statistics only—no individual employee records.")

            monitoring_history = st.session_state.get("m26_monitoring_history", [])
            if monitoring_history:
                st.markdown("#### Recent monitoring runs")
                history_rows = [
                    {
                        "Model": row.get("model"),
                        "Version": row.get("model_version"),
                        "Threshold": row.get("threshold"),
                        "Status": row.get("status", {}).get("status"),
                        "Monitoring fingerprint": row.get("monitoring_fingerprint"),
                    }
                    for row in monitoring_history[-20:]
                ]
                st.dataframe(history_rows, width="stretch", hide_index=True)
    else:
        st.caption(
            "Upload a monitoring CSV/TXT batch above and click **Run monitoring for uploaded batch** to inspect drift and quality."
        )

# ---------------------------------------------------------------------------
# Section 6: Responsible Use & Data Quality Details
# ---------------------------------------------------------------------------
st.divider()
gov_col1, gov_col2 = st.columns(2)
with gov_col1, st.expander("Data quality details"):
    st.json(cleaning_report)
    st.write("Missing values by column")
    st.dataframe(cleaned_data.isna().sum().rename("missing_count"), width="stretch")

with gov_col2, st.expander("Responsible-use notes, SHAP governance & limitations"):
    st.markdown("""
### Responsible AI & Explainability Principles
- **Model Behaviour vs. Human Intent:** SHAP values explain how the algorithm computes its mathematical outputs; they do **not** reflect why an employee actually decides to stay or leave.
- **Correlation is Not Causation:** A high positive or negative SHAP value indicates statistical association in the historical training set. Changing an employee's feature (e.g. assigning more projects) will not necessarily alter attrition risk in reality.
- **Historical Biases:** Machine learning models learn patterns from past organizational data, which may encode historical inequities, organizational friction, or unmeasured external factors (such as commute distance or management culture).
- **Prohibition on Automated Adverse Actions:** Under no circumstances should predictions or SHAP feature attributions be used to rank employees, make promotion or termination decisions, or penalize individuals.
- **Privacy and Data Protection:** Employee data contains sensitive workforce information. Production deployments must enforce access controls, anonymization, and regulatory compliance (e.g. GDPR, EEOC).
- **Educational Scope:** This dashboard is an analytical prototype designed to demonstrate Explainable AI techniques, not a production HR decision system.
""")
