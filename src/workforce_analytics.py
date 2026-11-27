"""Aggregate workforce analytics helpers.

These functions operate only on aggregate workforce fields. They must not be
used to rank or recommend actions about individual employees.
"""
from __future__ import annotations

import pandas as pd

TENURE_BINS = [-float("inf"), 1, 2, 3, 4, 5, 10, float("inf")]
TENURE_LABELS = ["0-1 years", "2 years", "3 years", "4 years", "5 years", "6-10 years", "10+ years"]
SALARY_ORDER = ["low", "medium", "high", "Unknown"]
MIN_GROUP_SIZE = 30


def apply_workforce_filters(
    data: pd.DataFrame,
    departments: list[str] | None = None,
    salaries: list[str] | None = None,
    tenure_range: tuple[float, float] | None = None,
    work_accidents: list[str] | None = None,
    promotions: list[str] | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters without modifying the source DataFrame."""
    filtered = data.copy()

    if departments:
        filtered = filtered[filtered["department"].astype("string").isin(departments)]
    if salaries:
        filtered = filtered[filtered["salary"].astype("string").isin(salaries)]
    if tenure_range is not None and "time_spend_company" in filtered.columns:
        tenure = pd.to_numeric(filtered["time_spend_company"], errors="coerce")
        filtered = filtered[
            tenure.between(float(tenure_range[0]), float(tenure_range[1]), inclusive="both")
        ]
    if work_accidents:
        filtered = filtered[
            filtered["Work_accident"].astype("string").isin(work_accidents)
        ]
    if promotions:
        filtered = filtered[
            filtered["promotion_last_5years"].astype("string").isin(promotions)
        ]
    return filtered


def add_tenure_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical tenure group column while retaining missing tenure records."""
    grouped = data.copy()
    if "time_spend_company" not in grouped.columns:
        grouped["tenure_group"] = "Unknown"
        return grouped

    tenure = pd.to_numeric(grouped["time_spend_company"], errors="coerce")
    tenure_group = pd.cut(
        tenure,
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")
    grouped["tenure_group"] = tenure_group.fillna("Unknown")
    return grouped


def _sort_by_order(summary: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Sort an attrition summary table by an explicit category order."""
    if summary.empty:
        return summary
    rank = {label: idx for idx, label in enumerate(order)}
    return (
        summary.assign(_order=summary["group"].map(lambda x: rank.get(str(x), len(order))))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def group_attrition(
    data: pd.DataFrame,
    column: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Calculate employee count and observed attrition statistics by a category."""
    if data.empty:
        return pd.DataFrame(columns=[
            "group", "employee_count", "left_count", "stayed_count",
            "attrition_rate", "reliable",
        ])
    if column not in data.columns:
        return pd.DataFrame(columns=[
            "group", "employee_count", "left_count", "stayed_count",
            "attrition_rate", "reliable",
        ])
    if "quit" not in data.columns:
        raise ValueError("The data must include the 'quit' target column.")

    values = data[column].astype("string").fillna("Unknown")
    work = pd.DataFrame({"group": values, "quit": pd.to_numeric(data["quit"], errors="coerce")})
    work = work.dropna(subset=["quit"])
    if work.empty:
        return pd.DataFrame(columns=[
            "group", "employee_count", "left_count", "stayed_count",
            "attrition_rate", "reliable",
        ])

    summary = (
        work.groupby("group", observed=True)
        .agg(employee_count=("quit", "size"), left_count=("quit", "sum"))
        .reset_index()
    )
    summary["stayed_count"] = summary["employee_count"] - summary["left_count"]
    summary["attrition_rate"] = (
        summary["left_count"] / summary["employee_count"].replace(0, pd.NA)
    ).fillna(0.0)
    summary["reliable"] = summary["employee_count"] >= int(min_group_size)
    return summary.sort_values(["reliable", "attrition_rate"], ascending=[False, False]).reset_index(drop=True)


def department_attrition(data: pd.DataFrame, min_group_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    return group_attrition(data, "department", min_group_size)


def salary_attrition(data: pd.DataFrame, min_group_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    return _sort_by_order(group_attrition(data, "salary", min_group_size), SALARY_ORDER)


def tenure_attrition(data: pd.DataFrame, min_group_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    grouped = add_tenure_groups(data)
    return _sort_by_order(group_attrition(grouped, "tenure_group", min_group_size), TENURE_LABELS + ["Unknown"])


def highest_attrition_group(
    summary: pd.DataFrame,
    label: str,
) -> dict[str, str | int | float | None]:
    """Return the highest-rate group only when it meets the sample-size safeguard."""
    reliable = summary[summary["reliable"]] if not summary.empty else summary
    if reliable.empty:
        return {
            "dimension": label,
            "group": None,
            "employee_count": 0,
            "attrition_rate": None,
        }
    row = reliable.sort_values("attrition_rate", ascending=False).iloc[0]
    return {
        "dimension": label,
        "group": str(row["group"]),
        "employee_count": int(row["employee_count"]),
        "attrition_rate": float(row["attrition_rate"]),
    }


def build_aggregate_export(
    department_summary: pd.DataFrame,
    salary_summary: pd.DataFrame,
    tenure_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine aggregate summaries into a privacy-safe CSV export."""
    frames = []
    for dimension, summary in [
        ("department", department_summary),
        ("salary", salary_summary),
        ("tenure", tenure_summary),
    ]:
        if summary.empty:
            continue
        frame = summary.rename(columns={"group": "category"}).copy()
        frame.insert(0, "dimension", dimension)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[
            "dimension", "category", "employee_count", "left_count",
            "stayed_count", "attrition_rate", "reliable",
        ])
    return pd.concat(frames, ignore_index=True)


def aggregate_export_to_csv(export: pd.DataFrame) -> bytes:
    """Serialize an aggregate export as a UTF-8 CSV download payload."""
    return export.to_csv(index=False).encode("utf-8")
