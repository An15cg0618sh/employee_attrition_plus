import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.analytics import (
    assign_tenure_group,
    compute_department_analytics,
    compute_group_attrition,
    compute_salary_analytics,
    compute_satisfaction_performance_summary,
    compute_tenure_analytics,
    compute_workforce_kpis,
    create_department_chart,
    create_salary_chart,
    create_satisfaction_distribution_chart,
    create_satisfaction_performance_scatter,
    create_tenure_chart,
    filter_workforce_data,
    generate_aggregate_export,
    prepare_analytics_dataframe,
)
from src.attrition import load_dataset

DATA_PATH = Path(__file__).parents[1] / "data" / "employee_data.csv"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA_PATH)


def test_tenure_grouping():
    series = pd.Series([0, 1, 2, 3, 4, 5, 6, 10, 11, None, np.nan, "invalid"])
    grouped = assign_tenure_group(series)
    expected = [
        "0-1 years", "0-1 years", "2 years", "3 years",
        "4 years", "5 years", "6-10 years", "6-10 years", "10+ years",
        "Unknown", "Unknown", "Unknown",
    ]
    assert grouped.tolist() == expected


def test_tenure_grouping_missing_and_boundary_values():
    series = pd.Series([
        None,
        pd.NA,
        np.nan,
        "",
        "invalid",
        np.inf,
        -np.inf,
        -1,
        0.999,
        1.0,
        1.001,
        2.0,
        2.001,
        5.0,
        5.001,
        10.0,
        10.001,
        "6",
    ])
    grouped = assign_tenure_group(series)

    assert grouped.iloc[:7].tolist() == ["Unknown"] * 7
    assert grouped.iloc[7:17].tolist() == [
        "0-1 years",
        "0-1 years",
        "0-1 years",
        "2 years",
        "2 years",
        "3 years",
        "5 years",
        "6-10 years",
        "6-10 years",
        "10+ years",
    ]
    assert grouped.iloc[17] == "6-10 years"


def test_department_attrition_calculations(dataset):
    dept_df = compute_department_analytics(dataset)
    assert not dept_df.empty
    expected_cols = {
        "department", "headcount", "attrition_count", "stayed_count",
        "attrition_rate", "headcount_share", "is_reliable", "sample_flag",
    }
    assert expected_cols.issubset(dept_df.columns)

    # Check conservation of headcount
    assert (dept_df["headcount"] == dept_df["attrition_count"] + dept_df["stayed_count"]).all()
    assert dept_df["headcount"].sum() == len(dataset)
    assert (dept_df["attrition_rate"] >= 0.0).all() and (dept_df["attrition_rate"] <= 1.0).all()
    assert pytest.approx(dept_df["headcount_share"].sum(), 0.001) == 1.0


def test_salary_attrition_calculations(dataset):
    salary_df = compute_salary_analytics(dataset)
    assert not salary_df.empty
    assert (salary_df["headcount"] == salary_df["attrition_count"] + salary_df["stayed_count"]).all()
    assert salary_df["headcount"].sum() == len(dataset)
    assert (salary_df["attrition_rate"] >= 0.0).all() and (salary_df["attrition_rate"] <= 1.0).all()

    observed_salaries = salary_df["salary"].tolist()
    assert set(observed_salaries).issubset({"low", "medium", "high", "Unknown"})


def test_tenure_attrition_calculations(dataset):
    tenure_df = compute_tenure_analytics(dataset)
    assert not tenure_df.empty
    assert (tenure_df["headcount"] == tenure_df["attrition_count"] + tenure_df["stayed_count"]).all()
    assert tenure_df["headcount"].sum() == len(dataset)
    assert (tenure_df["attrition_rate"] >= 0.0).all() and (tenure_df["attrition_rate"] <= 1.0).all()

    groups = set(tenure_df["tenure_group"])
    assert groups.issubset({"0-1 years", "2 years", "3 years", "4 years", "5 years", "6-10 years", "10+ years", "Unknown"})


def test_small_sample_safeguards():
    # Synthetic data with 1 large group (N=60) and 1 small group (N=5)
    data = pd.DataFrame({
        "department": ["Engineering"] * 60 + ["SpecialOps"] * 5,
        "quit": [1] * 20 + [0] * 40 + [1] * 4 + [0] * 1,  # SpecialOps has high attrition 80%
        "salary": ["high"] * 65,
        "time_spend_company": [3] * 65,
    })

    summary = compute_group_attrition(data, "department", min_sample_size=30)
    eng_row = summary[summary["department"] == "Engineering"].iloc[0]
    ops_row = summary[summary["department"] == "SpecialOps"].iloc[0]

    assert bool(eng_row["is_reliable"]) is True
    assert eng_row["sample_flag"] == "Sufficient sample"

    assert bool(ops_row["is_reliable"]) is False
    assert "Small sample" in ops_row["sample_flag"]

    # In KPIs, the small sample should be flagged or secondary
    kpis = compute_workforce_kpis(data, min_sample_size=30)
    assert kpis["highest_attrition_dept"] == "Engineering"
    assert kpis["dept_sample_reliable"] is True


def test_filter_workforce_data(dataset):
    all_count = len(dataset)
    # Filter by department
    filtered_dept = filter_workforce_data(dataset, departments=["sales", "support"])
    assert len(filtered_dept) < all_count
    assert set(filtered_dept["department"].unique()).issubset({"sales", "support"})

    # Filter by salary
    filtered_sal = filter_workforce_data(dataset, salaries=["low"])
    assert set(filtered_sal["salary"].unique()) == {"low"}

    # Filter by tenure
    filtered_tenure = filter_workforce_data(dataset, tenure_groups=["3 years"])
    assert set(filtered_tenure["tenure_group"].unique()) == {"3 years"}

    # Filter by satisfaction range
    filtered_sat = filter_workforce_data(dataset, min_satisfaction=0.4, max_satisfaction=0.6)
    assert (filtered_sat["satisfaction_level"] >= 0.4).all()
    assert (filtered_sat["satisfaction_level"] <= 0.6).all()


def test_empty_dataset_handling():
    empty_df = pd.DataFrame(columns=[
        "satisfaction_level", "last_evaluation", "number_project",
        "average_montly_hours", "time_spend_company", "Work_accident",
        "quit", "promotion_last_5years", "department", "salary",
    ])

    prep = prepare_analytics_dataframe(empty_df)
    assert prep.empty

    dept_res = compute_department_analytics(empty_df)
    assert dept_res.empty

    sal_res = compute_salary_analytics(empty_df)
    assert sal_res.empty

    ten_res = compute_tenure_analytics(empty_df)
    assert ten_res.empty

    sat_res = compute_satisfaction_performance_summary(empty_df)
    assert sat_res["mean_satisfaction"] == 0.0

    kpis = compute_workforce_kpis(empty_df)
    assert kpis["total_employees"] == 0
    assert kpis["overall_attrition_rate"] == 0.0

    filtered = filter_workforce_data(empty_df, departments=["sales"])
    assert filtered.empty

    export_df = generate_aggregate_export(empty_df)
    assert export_df.empty

    fig_dept = create_department_chart(dept_res)
    assert isinstance(fig_dept, go.Figure)

    fig_sal = create_salary_chart(sal_res)
    assert isinstance(fig_sal, go.Figure)

    fig_ten = create_tenure_chart(ten_res)
    assert isinstance(fig_ten, go.Figure)

    fig_scatter = create_satisfaction_performance_scatter(empty_df)
    assert isinstance(fig_scatter, go.Figure)

    fig_hist = create_satisfaction_distribution_chart(empty_df)
    assert isinstance(fig_hist, go.Figure)


def test_missing_values_handling():
    data_with_nans = pd.DataFrame({
        "department": ["sales", None, np.nan, "support"],
        "salary": ["low", None, "high", np.nan],
        "time_spend_company": [2, np.nan, 4, None],
        "satisfaction_level": [0.5, np.nan, 0.8, 0.2],
        "last_evaluation": [0.7, 0.6, np.nan, 0.9],
        "quit": [0, 1, np.nan, 1],
    })

    dept = compute_department_analytics(data_with_nans, min_sample_size=1)
    assert "Unknown" in dept["department"].values

    sal = compute_salary_analytics(data_with_nans, min_sample_size=1)
    assert "Unknown" in sal["salary"].values

    ten = compute_tenure_analytics(data_with_nans, min_sample_size=1)
    assert "Unknown" in ten["tenure_group"].values


def test_satisfaction_performance_summary(dataset):
    summary = compute_satisfaction_performance_summary(dataset)
    assert 0.0 < summary["mean_satisfaction"] < 1.0
    assert 0.0 < summary["mean_evaluation"] < 1.0
    assert len(summary["quadrants"]) == 4

    # Quadrant headcount sum should equal valid record count
    valid_count = dataset[["satisfaction_level", "last_evaluation"]].dropna().shape[0]
    quadrant_sum = sum(q["headcount"] for q in summary["quadrants"])
    assert quadrant_sum == valid_count


def test_aggregate_export_and_csv_generation(dataset):
    export_df = generate_aggregate_export(dataset, min_sample_size=20)
    assert not export_df.empty

    expected_cols = [
        "dimension", "category", "headcount", "attrition_count",
        "stayed_count", "attrition_rate", "headcount_share", "is_reliable", "sample_flag",
    ]
    assert list(export_df.columns) == expected_cols

    dims = set(export_df["dimension"])
    assert dims == {"Department", "Salary Band", "Tenure Range"}

    # Export to CSV buffer and verify roundtrip
    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    buffer.seek(0)
    reloaded = pd.read_csv(buffer)
    assert len(reloaded) == len(export_df)
    assert set(reloaded["dimension"]) == dims


def test_plotly_charts_generate_valid_figures(dataset):
    dept_df = compute_department_analytics(dataset)
    sal_df = compute_salary_analytics(dataset)
    ten_df = compute_tenure_analytics(dataset)

    fig1 = create_department_chart(dept_df)
    assert isinstance(fig1, go.Figure)
    assert len(fig1.data) >= 2

    fig2 = create_salary_chart(sal_df)
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) >= 2

    fig3 = create_tenure_chart(ten_df)
    assert isinstance(fig3, go.Figure)
    assert len(fig3.data) >= 2

    fig4 = create_satisfaction_performance_scatter(dataset, sample_limit=500)
    assert isinstance(fig4, go.Figure)

    fig5 = create_satisfaction_distribution_chart(dataset)
    assert isinstance(fig5, go.Figure)
