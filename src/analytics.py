"""Workforce Analytics and Exploratory Data Analysis module.

This module provides data aggregation, group-level attrition analysis,
statistical distribution summaries, dynamic filtering, small-sample safeguards,
and interactive Plotly visualization helpers for the Workforce Insights dashboard.

Responsible AI Note:
This analytics module reports aggregate historical patterns observed in the dataset.
Correlations and observed group attrition rates describe past workforce patterns,
NOT individual employee intentions, performance evaluations, or causal determinants.
These outputs must NEVER be used to automate employment decisions, rank individuals,
or justify adverse actions against employees.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TENURE_BINS = [-float("inf"), 1, 2, 3, 4, 5, 10, float("inf")]
TENURE_LABELS = ["0-1 years", "2 years", "3 years", "4 years", "5 years", "6-10 years", "10+ years"]
SALARY_ORDER = ["low", "medium", "high", "Unknown"]


def assign_tenure_group(series: pd.Series) -> pd.Series:
    """Group numeric tenure (years at company) into meaningful discrete ranges."""
    numeric_tenure = pd.to_numeric(series, errors="coerce")
    numeric_tenure = numeric_tenure.mask(
        numeric_tenure.isna() | ~np.isfinite(numeric_tenure)
    )
    grouped = pd.cut(
        numeric_tenure,
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
        include_lowest=True,
    )
    return grouped.astype("string").fillna("Unknown")


def prepare_analytics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean copy of the dataset with standardized categorical fields."""
    if df.empty:
        df_clean = df.copy()
        if "tenure_group" not in df_clean.columns:
            df_clean["tenure_group"] = pd.Series(dtype=str)
        return df_clean

    data = df.copy()
    if "time_spend_company" in data.columns:
        data["tenure_group"] = assign_tenure_group(data["time_spend_company"])
    else:
        data["tenure_group"] = "Unknown"

    if "department" in data.columns:
        data["department"] = data["department"].fillna("Unknown").astype(str)
    if "salary" in data.columns:
        data["salary"] = data["salary"].fillna("Unknown").astype(str)
    if "quit" in data.columns:
        data["quit"] = pd.to_numeric(data["quit"], errors="coerce").fillna(0).astype(int)

    return data


def compute_group_attrition(
    df: pd.DataFrame,
    group_col: str,
    min_sample_size: int = 30,
) -> pd.DataFrame:
    """Aggregate headcount, attrition count, and attrition rate by a given attribute.

    Applies small-sample safeguards to identify when subgroup counts are too small
    to support reliable ranking.
    """
    columns = [
        group_col, "headcount", "attrition_count", "stayed_count",
        "attrition_rate", "headcount_share", "is_reliable", "sample_flag",
    ]
    if df.empty or group_col not in df.columns or "quit" not in df.columns:
        return pd.DataFrame(columns=columns)

    work_df = df.copy()
    work_df[group_col] = work_df[group_col].fillna("Unknown").astype(str)

    total_headcount = len(work_df)

    grouped = (
        work_df.groupby(group_col, observed=False)["quit"]
        .agg(
            headcount="count",
            attrition_count=lambda s: int((s == 1).sum()),
            stayed_count=lambda s: int((s == 0).sum()),
        )
        .reset_index()
    )

    grouped["attrition_rate"] = np.where(
        grouped["headcount"] > 0,
        grouped["attrition_count"] / grouped["headcount"],
        0.0,
    )

    grouped["headcount_share"] = np.where(
        total_headcount > 0,
        grouped["headcount"] / total_headcount,
        0.0,
    )

    grouped["is_reliable"] = grouped["headcount"] >= min_sample_size
    grouped["sample_flag"] = np.where(
        grouped["is_reliable"],
        "Sufficient sample",
        "Small sample (N < " + str(min_sample_size) + ")",
    )

    return grouped.sort_values("attrition_rate", ascending=False).reset_index(drop=True)


def compute_department_analytics(df: pd.DataFrame, min_sample_size: int = 30) -> pd.DataFrame:
    """Compute employee distribution, attrition count, and attrition rate by department."""
    clean_df = prepare_analytics_dataframe(df)
    return compute_group_attrition(clean_df, "department", min_sample_size)


def compute_salary_analytics(df: pd.DataFrame, min_sample_size: int = 30) -> pd.DataFrame:
    """Compute employee distribution, attrition count, and attrition rate by salary category."""
    clean_df = prepare_analytics_dataframe(df)
    summary = compute_group_attrition(clean_df, "salary", min_sample_size)
    if not summary.empty:
        # Preserve intuitive salary hierarchy ordering
        cat_type = pd.CategoricalDtype(categories=SALARY_ORDER, ordered=True)
        summary["salary_order"] = summary["salary"].astype(cat_type)
        summary = summary.sort_values("salary_order").drop(columns=["salary_order"]).reset_index(drop=True)
    return summary


def compute_tenure_analytics(df: pd.DataFrame, min_sample_size: int = 30) -> pd.DataFrame:
    """Compute employee distribution and attrition rate across discrete tenure ranges."""
    clean_df = prepare_analytics_dataframe(df)
    summary = compute_group_attrition(clean_df, "tenure_group", min_sample_size)
    if not summary.empty:
        cat_type = pd.CategoricalDtype(categories=TENURE_LABELS + ["Unknown"], ordered=True)
        summary["tenure_order"] = summary["tenure_group"].astype(cat_type)
        summary = summary.sort_values("tenure_order").drop(columns=["tenure_order"]).reset_index(drop=True)
    return summary


def compute_satisfaction_performance_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze distributions of satisfaction level and last evaluation."""
    empty_result: dict[str, Any] = {
        "mean_satisfaction": 0.0,
        "median_satisfaction": 0.0,
        "mean_satisfaction_stayed": 0.0,
        "mean_satisfaction_left": 0.0,
        "mean_evaluation": 0.0,
        "median_evaluation": 0.0,
        "mean_evaluation_stayed": 0.0,
        "mean_evaluation_left": 0.0,
        "correlation": 0.0,
        "quadrants": [],
    }
    if df.empty or "satisfaction_level" not in df.columns or "last_evaluation" not in df.columns:
        return empty_result

    clean_df = df.dropna(subset=["satisfaction_level", "last_evaluation"]).copy()
    if clean_df.empty:
        return empty_result

    sat = clean_df["satisfaction_level"]
    eval_score = clean_df["last_evaluation"]
    quit_series = clean_df["quit"] if "quit" in clean_df.columns else pd.Series(0, index=clean_df.index)

    stayed_mask = quit_series == 0
    left_mask = quit_series == 1

    std_sat = float(sat.std())
    std_eval = float(eval_score.std())
    corr = float(sat.corr(eval_score)) if (std_sat > 1e-8 and std_eval > 1e-8) else 0.0

    # Quadrant analysis thresholds: Median / conventional benchmarks
    sat_midpoint = 0.60
    eval_midpoint = 0.70

    quadrants = [
        {
            "name": "High Satisfaction, High Performance",
            "condition": (sat >= sat_midpoint) & (eval_score >= eval_midpoint),
            "description": "Engaged High Contributors",
        },
        {
            "name": "Low Satisfaction, High Performance",
            "condition": (sat < sat_midpoint) & (eval_score >= eval_midpoint),
            "description": "At-Risk High Contributors (Burnout / Retention Risk)",
        },
        {
            "name": "High Satisfaction, Moderate Performance",
            "condition": (sat >= sat_midpoint) & (eval_score < eval_midpoint),
            "description": "Content Core Contributors",
        },
        {
            "name": "Low Satisfaction, Moderate Performance",
            "condition": (sat < sat_midpoint) & (eval_score < eval_midpoint),
            "description": "Disengaged / Underperforming Group",
        },
    ]

    quadrant_metrics = []
    total_count = len(clean_df)
    for q in quadrants:
        subset = clean_df[q["condition"]]
        q_count = len(subset)
        q_attr = int((subset["quit"] == 1).sum()) if "quit" in subset.columns else 0
        rate = float(q_attr / q_count) if q_count > 0 else 0.0
        share = float(q_count / total_count) if total_count > 0 else 0.0
        quadrant_metrics.append({
            "quadrant": q["name"],
            "description": q["description"],
            "headcount": q_count,
            "headcount_share": share,
            "attrition_count": q_attr,
            "attrition_rate": rate,
        })

    return {
        "mean_satisfaction": float(sat.mean()),
        "median_satisfaction": float(sat.median()),
        "mean_satisfaction_stayed": float(sat[stayed_mask].mean()) if stayed_mask.any() else 0.0,
        "mean_satisfaction_left": float(sat[left_mask].mean()) if left_mask.any() else 0.0,
        "mean_evaluation": float(eval_score.mean()),
        "median_evaluation": float(eval_score.median()),
        "mean_evaluation_stayed": float(eval_score[stayed_mask].mean()) if stayed_mask.any() else 0.0,
        "mean_evaluation_left": float(eval_score[left_mask].mean()) if left_mask.any() else 0.0,
        "correlation": corr,
        "quadrants": quadrant_metrics,
    }


def compute_workforce_kpis(df: pd.DataFrame, min_sample_size: int = 30) -> dict[str, Any]:
    """Calculate dynamic high-level KPIs and top observed attrition segments."""
    if df.empty:
        return {
            "total_employees": 0,
            "total_attrition": 0,
            "overall_attrition_rate": 0.0,
            "highest_attrition_dept": "N/A",
            "highest_dept_rate": 0.0,
            "dept_sample_reliable": False,
            "highest_attrition_salary": "N/A",
            "highest_salary_rate": 0.0,
            "salary_sample_reliable": False,
            "highest_attrition_tenure": "N/A",
            "highest_tenure_rate": 0.0,
            "tenure_sample_reliable": False,
        }

    total_employees = len(df)
    total_attrition = int((df["quit"] == 1).sum()) if "quit" in df.columns else 0
    overall_rate = float(total_attrition / total_employees) if total_employees > 0 else 0.0

    # Department top segment
    dept_df = compute_department_analytics(df, min_sample_size)
    reliable_depts = dept_df[dept_df["is_reliable"]]
    if not reliable_depts.empty:
        top_dept_row = reliable_depts.iloc[0]
        top_dept = top_dept_row["department"]
        top_dept_rate = float(top_dept_row["attrition_rate"])
        dept_reliable = True
    elif not dept_df.empty:
        top_dept_row = dept_df.iloc[0]
        top_dept = f"{top_dept_row['department']} (N={top_dept_row['headcount']})"
        top_dept_rate = float(top_dept_row["attrition_rate"])
        dept_reliable = False
    else:
        top_dept = "N/A"
        top_dept_rate = 0.0
        dept_reliable = False

    # Salary top segment
    sal_df = compute_salary_analytics(df, min_sample_size)
    reliable_sal = sal_df[sal_df["is_reliable"]]
    if not reliable_sal.empty:
        top_sal_row = reliable_sal.sort_values("attrition_rate", ascending=False).iloc[0]
        top_sal = top_sal_row["salary"]
        top_sal_rate = float(top_sal_row["attrition_rate"])
        sal_reliable = True
    elif not sal_df.empty:
        top_sal_row = sal_df.sort_values("attrition_rate", ascending=False).iloc[0]
        top_sal = f"{top_sal_row['salary']} (N={top_sal_row['headcount']})"
        top_sal_rate = float(top_sal_row["attrition_rate"])
        sal_reliable = False
    else:
        top_sal = "N/A"
        top_sal_rate = 0.0
        sal_reliable = False

    # Tenure top segment
    ten_df = compute_tenure_analytics(df, min_sample_size)
    reliable_ten = ten_df[ten_df["is_reliable"]]
    if not reliable_ten.empty:
        top_ten_row = reliable_ten.sort_values("attrition_rate", ascending=False).iloc[0]
        top_ten = top_ten_row["tenure_group"]
        top_ten_rate = float(top_ten_row["attrition_rate"])
        ten_reliable = True
    elif not ten_df.empty:
        top_ten_row = ten_df.sort_values("attrition_rate", ascending=False).iloc[0]
        top_ten = f"{top_ten_row['tenure_group']} (N={top_ten_row['headcount']})"
        top_ten_rate = float(top_ten_row["attrition_rate"])
        ten_reliable = False
    else:
        top_ten = "N/A"
        top_ten_rate = 0.0
        ten_reliable = False

    return {
        "total_employees": total_employees,
        "total_attrition": total_attrition,
        "overall_attrition_rate": overall_rate,
        "highest_attrition_dept": top_dept,
        "highest_dept_rate": top_dept_rate,
        "dept_sample_reliable": dept_reliable,
        "highest_attrition_salary": top_sal,
        "highest_salary_rate": top_sal_rate,
        "salary_sample_reliable": sal_reliable,
        "highest_attrition_tenure": top_ten,
        "highest_tenure_rate": top_ten_rate,
        "tenure_sample_reliable": ten_reliable,
    }


def filter_workforce_data(
    df: pd.DataFrame,
    departments: list[str] | None = None,
    salaries: list[str] | None = None,
    tenure_groups: list[str] | None = None,
    min_satisfaction: float | None = None,
    max_satisfaction: float | None = None,
    work_accident: int | None = None,
    promotion_last_5years: int | None = None,
) -> pd.DataFrame:
    """Filter workforce dataset according to user-selected criteria."""
    if df.empty:
        return df.copy()

    clean_df = prepare_analytics_dataframe(df)
    filtered = clean_df.copy()

    if departments is not None and len(departments) > 0:
        filtered = filtered[filtered["department"].isin(departments)]

    if salaries is not None and len(salaries) > 0:
        filtered = filtered[filtered["salary"].isin(salaries)]

    if tenure_groups is not None and len(tenure_groups) > 0:
        filtered = filtered[filtered["tenure_group"].isin(tenure_groups)]

    if min_satisfaction is not None and "satisfaction_level" in filtered.columns:
        filtered = filtered[filtered["satisfaction_level"] >= min_satisfaction]

    if max_satisfaction is not None and "satisfaction_level" in filtered.columns:
        filtered = filtered[filtered["satisfaction_level"] <= max_satisfaction]

    if work_accident is not None and "Work_accident" in filtered.columns:
        filtered = filtered[filtered["Work_accident"] == work_accident]

    if promotion_last_5years is not None and "promotion_last_5years" in filtered.columns:
        filtered = filtered[filtered["promotion_last_5years"] == promotion_last_5years]

    return filtered.reset_index(drop=True)


def generate_aggregate_export(df: pd.DataFrame, min_sample_size: int = 30) -> pd.DataFrame:
    """Generate aggregated summary tables suitable for export without employee identifiers."""
    if df.empty:
        return pd.DataFrame(columns=[
            "dimension", "category", "headcount", "attrition_count",
            "stayed_count", "attrition_rate", "headcount_share", "is_reliable", "sample_flag",
        ])

    dept_summary = compute_department_analytics(df, min_sample_size).rename(columns={"department": "category"})
    dept_summary.insert(0, "dimension", "Department")

    sal_summary = compute_salary_analytics(df, min_sample_size).rename(columns={"salary": "category"})
    sal_summary.insert(0, "dimension", "Salary Band")

    ten_summary = compute_tenure_analytics(df, min_sample_size).rename(columns={"tenure_group": "category"})
    ten_summary.insert(0, "dimension", "Tenure Range")

    export_df = pd.concat([dept_summary, sal_summary, ten_summary], ignore_index=True)
    return export_df


# ---------------------------------------------------------------------------
# Plotly Chart Generators
# ---------------------------------------------------------------------------

def create_department_chart(dept_df: pd.DataFrame) -> go.Figure:
    """Create dual-axis interactive Plotly bar chart for Department analytics."""
    if dept_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No department data available for selected filters")
        return fig

    sorted_df = dept_df.sort_values("attrition_rate", ascending=True)

    fig = go.Figure()

    # Headcount bar
    fig.add_trace(go.Bar(
        y=sorted_df["department"],
        x=sorted_df["headcount"],
        orientation="h",
        name="Total Headcount",
        marker=dict(color="#1f77b4", opacity=0.8),
        hovertemplate="<b>%{y}</b><br>Headcount: %{x:,}<br>Share: %{customdata:.1%}<extra></extra>",
        customdata=sorted_df["headcount_share"],
    ))

    # Attrition count bar
    fig.add_trace(go.Bar(
        y=sorted_df["department"],
        x=sorted_df["attrition_count"],
        orientation="h",
        name="Observed Attrition",
        marker=dict(color="#d62728", opacity=0.85),
        hovertemplate="<b>%{y}</b><br>Attrition: %{x:,}<br>Rate: %{customdata:.1%}<extra></extra>",
        customdata=sorted_df["attrition_rate"],
    ))

    fig.update_layout(
        title="Department Headcount & Observed Attrition Breakdown",
        barmode="group",
        xaxis=dict(title="Number of Employees", gridcolor="#e5e5e5"),
        yaxis=dict(title="Department"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_salary_chart(salary_df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart comparing retention vs attrition across salary bands."""
    if salary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No salary data available for selected filters")
        return fig

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=salary_df["salary"],
        y=salary_df["stayed_count"],
        name="Stayed",
        marker=dict(color="#2ca02c"),
        hovertemplate="<b>%{x} Salary</b><br>Stayed: %{y:,}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=salary_df["salary"],
        y=salary_df["attrition_count"],
        name="Left",
        marker=dict(color="#d62728"),
        hovertemplate="<b>%{x} Salary</b><br>Left: %{y:,}<br>Attrition Rate: %{customdata:.1%}<extra></extra>",
        customdata=salary_df["attrition_rate"],
    ))

    fig.update_layout(
        title="Attrition by Salary Tier (Stayed vs Left)",
        barmode="stack",
        xaxis=dict(title="Salary Category"),
        yaxis=dict(title="Number of Employees", gridcolor="#e5e5e5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_tenure_chart(tenure_df: pd.DataFrame) -> go.Figure:
    """Create combined bar + rate line chart across tenure ranges."""
    if tenure_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No tenure data available for selected filters")
        return fig

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tenure_df["tenure_group"],
        y=tenure_df["headcount"],
        name="Headcount",
        marker=dict(color="#4a90e2", opacity=0.7),
        hovertemplate="<b>%{x}</b><br>Headcount: %{y:,}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=tenure_df["tenure_group"],
        y=tenure_df["attrition_rate"] * 100,
        name="Attrition Rate (%)",
        yaxis="y2",
        mode="lines+markers",
        marker=dict(color="#e74c3c", size=8),
        line=dict(color="#e74c3c", width=3),
        hovertemplate="<b>%{x}</b><br>Attrition Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Tenure Distribution & Observed Attrition Rate",
        xaxis=dict(title="Tenure Range"),
        yaxis=dict(title="Total Headcount", gridcolor="#e5e5e5"),
        yaxis2=dict(
            title="Attrition Rate (%)",
            overlaying="y",
            side="right",
            ticksuffix="%",
            range=[0, max(60, float(tenure_df["attrition_rate"].max() * 110))],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_satisfaction_performance_scatter(df: pd.DataFrame, sample_limit: int = 1500) -> go.Figure:
    """Generate scatter plot exploring satisfaction vs performance evaluation colored by quit status."""
    if df.empty or "satisfaction_level" not in df.columns or "last_evaluation" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No satisfaction / performance data available")
        return fig

    plot_df = df.dropna(subset=["satisfaction_level", "last_evaluation"]).copy()
    if len(plot_df) > sample_limit:
        plot_df = plot_df.sample(n=sample_limit, random_state=42)

    plot_df["status"] = plot_df["quit"].map({0: "Stayed", 1: "Left"}).fillna("Unknown")

    color_map = {"Stayed": "#2ca02c", "Left": "#d62728", "Unknown": "#7f7f7f"}

    fig = px.scatter(
        plot_df,
        x="satisfaction_level",
        y="last_evaluation",
        color="status",
        color_discrete_map=color_map,
        opacity=0.6,
        labels={
            "satisfaction_level": "Job Satisfaction Score (0.0 to 1.0)",
            "last_evaluation": "Last Performance Evaluation (0.0 to 1.0)",
            "status": "Observed Status",
        },
        title=f"Job Satisfaction vs Performance Evaluation (Sample N={len(plot_df):,})",
        hover_data=["time_spend_company", "salary", "department"] if "department" in plot_df.columns else None,
    )

    # Reference median benchmark lines
    fig.add_vline(x=0.60, line_dash="dash", line_color="#888888", opacity=0.5)
    fig.add_hline(y=0.70, line_dash="dash", line_color="#888888", opacity=0.5)

    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_satisfaction_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create histogram distribution of satisfaction scores separated by quit status."""
    if df.empty or "satisfaction_level" not in df.columns or "quit" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No satisfaction data available")
        return fig

    plot_df = df.dropna(subset=["satisfaction_level", "quit"]).copy()
    plot_df["status"] = plot_df["quit"].map({0: "Stayed", 1: "Left"}).fillna("Unknown")

    fig = px.histogram(
        plot_df,
        x="satisfaction_level",
        color="status",
        barmode="overlay",
        nbins=25,
        color_discrete_map={"Stayed": "#2ca02c", "Left": "#d62728"},
        opacity=0.65,
        labels={"satisfaction_level": "Satisfaction Level", "status": "Status"},
        title="Satisfaction Score Distribution (Stayed vs Left)",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Satisfaction Score", range=[0, 1]),
        yaxis=dict(title="Number of Employees", gridcolor="#e5e5e5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
