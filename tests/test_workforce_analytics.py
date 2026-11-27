from pathlib import Path
from io import BytesIO

import pandas as pd
import pytest

from src.attrition import clean_dataset, load_dataset
from src.workforce_analytics import (
    TENURE_LABELS,
    add_tenure_groups,
    aggregate_export_to_csv,
    apply_workforce_filters,
    build_aggregate_export,
    department_attrition,
    group_attrition,
    highest_attrition_group,
    salary_attrition,
    tenure_attrition,
)

DATA = Path(__file__).parents[1] / "data" / "employee_data.csv"


@pytest.fixture(scope="module")
def workforce():
    return load_dataset(DATA)


def test_department_attrition_calculation(workforce):
    summary = department_attrition(workforce)
    assert set(summary.columns) == {
        "group", "employee_count", "left_count", "stayed_count",
        "attrition_rate", "reliable",
    }
    assert summary["employee_count"].sum() == len(workforce)
    assert ((summary["attrition_rate"] >= 0) & (summary["attrition_rate"] <= 1)).all()


def test_salary_attrition_calculation(workforce):
    summary = salary_attrition(workforce)
    expected_groups = set(workforce["salary"].dropna().astype(str))
    if workforce["salary"].isna().any():
        expected_groups.add("Unknown")
    assert set(summary["group"]) == expected_groups
    assert summary["left_count"].sum() == workforce["quit"].sum()


def test_tenure_grouping_handles_ranges_and_missing_values():
    data = pd.DataFrame({
        "time_spend_company": [0, 1, 2, 5, 10, 12, None],
        "quit": [0, 1, 1, 0, 1, 0, 1],
    })
    grouped = add_tenure_groups(data)
    assert "tenure_group" in grouped.columns
    assert set(grouped["tenure_group"].dropna()) <= {
        "0-1 years", "2 years", "3 years", "4 years", "5 years",
        "6-10 years", "10+ years", "Unknown",
    }

    summary = tenure_attrition(data, min_group_size=1)
    assert summary["employee_count"].sum() == 7
    # Verify chronological ordering of tenure groups
    expected_order = [g for g in (TENURE_LABELS + ["Unknown"]) if g in set(summary["group"])]
    assert summary["group"].tolist() == expected_order


def test_deduplicated_source_of_truth_and_no_filter_counts(workforce):
    cleaned, report = clean_dataset(workforce)
    assert report["rows_before"] == 11582
    assert report["duplicates_removed"] == 9
    assert report["rows_after"] == 11573
    assert len(cleaned) == 11573

    tenure_vals = pd.to_numeric(cleaned["time_spend_company"], errors="coerce").dropna()
    no_filter = apply_workforce_filters(
        cleaned,
        departments=[],
        salaries=[],
        tenure_range=(int(tenure_vals.min()), int(tenure_vals.max())),
        work_accidents=[],
        promotions=[],
    )
    assert len(no_filter) == len(cleaned) == 11573
    assert department_attrition(no_filter)["employee_count"].sum() == 11573
    assert salary_attrition(no_filter)["employee_count"].sum() == 11573
    assert tenure_attrition(no_filter)["employee_count"].sum() == 11573


def test_filters_update_filtered_dataset(workforce):
    cleaned, _ = clean_dataset(workforce)
    first_department = str(cleaned["department"].dropna().iloc[0])
    filtered = apply_workforce_filters(
        cleaned,
        departments=[first_department],
        salaries=[str(cleaned["salary"].dropna().iloc[0])],
    )
    assert 0 < len(filtered) < len(cleaned)
    assert set(filtered["department"].dropna().astype(str)).issubset({first_department})


def test_empty_filtered_dataset():
    empty = pd.DataFrame(columns=["department", "salary", "time_spend_company", "quit"])
    department = department_attrition(empty)
    salary = salary_attrition(empty)
    tenure = tenure_attrition(empty)
    assert department.empty
    assert salary.empty
    assert tenure.empty


def test_missing_categories_and_values_are_grouped_as_unknown():
    data = pd.DataFrame({
        "department": ["Sales", None, "Sales"],
        "salary": ["low", "medium", None],
        "time_spend_company": [1, None, 3],
        "quit": [0, 1, 1],
    })
    summary = department_attrition(data, min_group_size=1)
    assert "Unknown" in set(summary["group"])
    assert summary["employee_count"].sum() == 3


def test_small_group_safeguard_prevents_unreliable_ranking():
    data = pd.DataFrame({
        "department": ["A", "A", "B", "B", "B", "B", "B"],
        "quit": [1, 1, 0, 0, 0, 0, 0],
    })
    summary = department_attrition(data, min_group_size=5)
    assert not summary.loc[summary["group"] == "A", "reliable"].iloc[0]
    insight = highest_attrition_group(summary, "Department")
    assert insight["group"] == "B"


def test_group_attrition_rejects_missing_target():
    data = pd.DataFrame({"department": ["Sales"]})
    with pytest.raises(ValueError, match="quit"):
        group_attrition(data, "department")


def test_aggregate_export_has_no_employee_identifiers(workforce):
    cleaned, _ = clean_dataset(workforce)
    dept = department_attrition(cleaned)
    sal = salary_attrition(cleaned)
    ten = tenure_attrition(cleaned)
    export = build_aggregate_export(dept, sal, ten)
    assert "employee_name" not in export.columns
    assert "employee_id" not in export.columns
    assert set(export["dimension"]) == {"department", "salary", "tenure"}
    assert export.loc[export["dimension"] == "department", "employee_count"].sum() == 11573
    assert export.loc[export["dimension"] == "salary", "employee_count"].sum() == 11573
    assert export.loc[export["dimension"] == "tenure", "employee_count"].sum() == 11573


def test_filtered_aggregate_export_reflects_combined_filters(workforce):
    cleaned, _ = clean_dataset(workforce)
    filtered = apply_workforce_filters(
        cleaned,
        departments=["sales"],
        salaries=["low"],
        work_accidents=["0"],
        promotions=["0.0"],
    )

    export = build_aggregate_export(
        department_attrition(filtered),
        salary_attrition(filtered),
        tenure_attrition(filtered),
    )

    assert not export.empty
    assert list(export.columns) == [
        "dimension",
        "category",
        "employee_count",
        "left_count",
        "stayed_count",
        "attrition_rate",
        "reliable",
    ]
    assert export.loc[export["dimension"] == "department", "employee_count"].sum() == len(filtered)
    assert export.loc[export["dimension"] == "salary", "employee_count"].sum() == len(filtered)
    assert export.loc[export["dimension"] == "tenure", "employee_count"].sum() == len(filtered)
    assert len(filtered) < len(cleaned)
    assert set(export.columns).isdisjoint({
        "employee_name",
        "employee_id",
        "prediction",
        "individual_shap",
    })


def test_empty_filtered_dataset_exports_required_headers():
    export = build_aggregate_export(
        department_attrition(pd.DataFrame()),
        salary_attrition(pd.DataFrame()),
        tenure_attrition(pd.DataFrame()),
    )
    assert export.empty
    assert list(export.columns) == [
        "dimension",
        "category",
        "employee_count",
        "left_count",
        "stayed_count",
        "attrition_rate",
        "reliable",
    ]


def test_aggregate_export_csv_payload_round_trip(workforce):
    cleaned, _ = clean_dataset(workforce)
    export = build_aggregate_export(
        department_attrition(cleaned),
        salary_attrition(cleaned),
        tenure_attrition(cleaned),
    )

    payload = aggregate_export_to_csv(export)
    reloaded = pd.read_csv(BytesIO(payload))

    assert list(reloaded.columns) == [
        "dimension",
        "category",
        "employee_count",
        "left_count",
        "stayed_count",
        "attrition_rate",
        "reliable",
    ]
    assert reloaded.loc[reloaded["dimension"] == "department", "employee_count"].sum() == 11573
    assert reloaded.loc[reloaded["dimension"] == "salary", "employee_count"].sum() == 11573
    assert reloaded.loc[reloaded["dimension"] == "tenure", "employee_count"].sum() == 11573
