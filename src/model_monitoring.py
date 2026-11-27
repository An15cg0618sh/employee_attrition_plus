"""Aggregate-only model monitoring and data drift utilities."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.attrition import EXPECTED_COLUMNS, TrainingResult, OptimizationResult

MONITORING_DIR = Path(os.getenv("MONITORING_DIR", "artifacts/monitoring"))
REFERENCE_METADATA_FILENAME = "reference_metadata.json"
MONITORING_HISTORY_FILENAME = "monitoring_history.json"


def _finite(series: Any) -> np.ndarray:
    converted = pd.to_numeric(series, errors="coerce")
    if isinstance(converted, np.ndarray):
        values = np.asarray(converted, dtype=float)
    elif isinstance(converted, pd.Series):
        values = converted.to_numpy(dtype=float)
    else:
        values = np.asarray(converted, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_reference_metadata(
    reference_data: pd.DataFrame,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create aggregate reference metadata without storing employee records."""
    if not isinstance(reference_data, pd.DataFrame):
        raise TypeError("reference_data must be a pandas DataFrame.")
    if reference_data.empty:
        raise ValueError("Reference data contains no rows.")

    features = reference_data.drop(columns=["quit"], errors="ignore").copy()
    schema: dict[str, dict[str, Any]] = {}
    numeric: dict[str, dict[str, Any]] = {}
    categorical: dict[str, dict[str, Any]] = {}

    for column in features.columns:
        series = features[column]
        role = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
        missing = int(series.isna().sum())
        schema[column] = {
            "dtype": str(series.dtype),
            "role": role,
            "missing_count": missing,
            "missing_rate": float(missing / len(series)) if len(series) else 0.0,
        }
        if role == "numeric":
            values = _finite(series)
            numeric[column] = {
                "mean": float(np.mean(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "std": float(np.std(values, ddof=0)) if len(values) else None,
                "quantiles": (
                    np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
                    if len(values)
                    else []
                ),
                "missing_count": missing,
                "missing_rate": float(missing / len(series)),
            }
        else:
            counts = series.astype("object").where(series.notna(), "__MISSING__").value_counts(dropna=False)
            categorical[column] = {
                "counts": {str(key): int(value) for key, value in counts.items()},
                "proportions": {
                    str(key): float(value / len(series))
                    for key, value in counts.items()
                },
            }

    return {
        "created_at": created_at or _utc_now(),
        "fingerprint": joblib.hash(features),
        "row_count": int(len(features)),
        "feature_count": int(len(features.columns)),
        "feature_schema": schema,
        "numeric": numeric,
        "categorical": categorical,
    }


build_reference_metadata = create_reference_metadata


def ingest_monitoring_file(path: str | Path) -> pd.DataFrame:
    """Load CSV or delimited text monitoring data without retaining employee records."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Monitoring data file not found: {target}")
    suffix = target.suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise ValueError(
            f"Unsupported monitoring file type '{suffix}'. Use CSV or delimited TXT."
        )
    try:
        if suffix == ".txt":
            data = pd.read_csv(target, sep=None, engine="python")
        else:
            data = pd.read_csv(target)
    except Exception as exc:
        raise ValueError(f"Could not read monitoring data '{target}': {exc}") from exc
    if data.empty:
        raise ValueError("Monitoring data file contains no rows.")
    return data


def validate_monitoring_data(
    monitoring_data: pd.DataFrame,
    reference_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a monitoring batch and report aggregate-only quality issues."""
    if not isinstance(monitoring_data, pd.DataFrame):
        raise TypeError("monitoring_data must be a pandas DataFrame.")
    if monitoring_data.empty:
        raise ValueError("Monitoring data contains no rows.")

    expected = sorted(EXPECTED_COLUMNS - {"quit"})
    missing_columns = sorted(set(expected).difference(monitoring_data.columns))
    unexpected_columns = sorted(set(monitoring_data.columns).difference(EXPECTED_COLUMNS))
    duplicate_count = int(monitoring_data.duplicated().sum())
    invalid_numeric: dict[str, int] = {}
    unexpected_categories: dict[str, int] = {}

    for column in expected:
        if column not in monitoring_data:
            continue
        if column in {"department", "salary"}:
            continue
        if column in {"Work_accident", "promotion_last_5years", "quit"}:
            continue
        invalid_numeric[column] = int(
            pd.to_numeric(monitoring_data[column], errors="coerce").isna().sum()
            - monitoring_data[column].isna().sum()
        )

    if "satisfaction_level" in monitoring_data:
        invalid_numeric["satisfaction_level"] = int(
            ((pd.to_numeric(monitoring_data["satisfaction_level"], errors="coerce") < 0)
             | (pd.to_numeric(monitoring_data["satisfaction_level"], errors="coerce") > 1)).sum()
        )
    if "last_evaluation" in monitoring_data:
        invalid_numeric["last_evaluation"] = int(
            ((pd.to_numeric(monitoring_data["last_evaluation"], errors="coerce") < 0)
             | (pd.to_numeric(monitoring_data["last_evaluation"], errors="coerce") > 1)).sum()
        )

    if reference_metadata:
        reference_categorical = reference_metadata.get("categorical", {})
        for column, summary in reference_categorical.items():
            if column not in monitoring_data:
                continue
            known = set(summary.get("counts", {}).keys())
            known.discard("__MISSING__")
            non_missing = monitoring_data[column].dropna().astype(str)
            unexpected_categories[column] = int(
                sum(value not in known for value in non_missing)
            )

    quit_status = "absent"
    quit_valid = False
    quit_note = "Ground truth absent."
    if "quit" in monitoring_data:
        quit_status = "present"
        raw_quit = monitoring_data["quit"]
        missing_count = int(raw_quit.isna().sum())
        observed = set(raw_quit.dropna().unique())
        invalid_labels = any(value not in {0, 1} for value in observed)
        if invalid_labels:
            quit_status = "invalid_labels"
            quit_note = "quit contains invalid labels."
        elif missing_count:
            quit_status = "partial_binary"
            quit_note = "quit contains missing labels and is not complete ground truth."
        elif observed.issubset({0, 1}):
            quit_valid = True
            quit_status = "valid_binary"
            quit_note = "Valid binary ground truth is available."
        else:
            quit_note = "quit contains missing or invalid labels."

    missing_rates = {
        column: float(monitoring_data[column].isna().mean())
        for column in monitoring_data.columns
    }
    return {
        "row_count": int(len(monitoring_data)),
        "fingerprint": joblib.hash(monitoring_data),
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "duplicate_count": duplicate_count,
        "invalid_numeric": {k: v for k, v in invalid_numeric.items() if v > 0},
        "unexpected_categories": {
            k: v for k, v in unexpected_categories.items() if v > 0
        },
        "missing_rates": missing_rates,
        "quit_status": quit_status,
        "quit_valid": quit_valid,
        "ground_truth_note": quit_note,
        "valid": not missing_columns and not unexpected_columns,
    }


def calculate_psi(reference: Any, current: Any, epsilon: float = 1e-6) -> float:
    ref = _finite(reference)
    cur = _finite(current)
    if not len(ref) or not len(cur):
        return float("nan")
    if np.all(ref == ref[0]):
        cut_points = np.array([ref[0] - 1.0, ref[0] + 1.0])
    else:
        cut_points = np.quantile(ref, [0, 0.25, 0.5, 0.75, 1])
        cut_points = np.unique(cut_points)
    if len(cut_points) < 2:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=cut_points)
    cur_counts, _ = np.histogram(cur, bins=cut_points)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), epsilon, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_drift(
    reference: Any,
    current: Any,
    alpha: float = 0.05,
) -> dict[str, Any]:
    ref = _finite(reference)
    cur = _finite(current)
    psi = calculate_psi(ref, cur)
    if len(ref) and len(cur):
        ks = ks_2samp(ref, cur)
        statistic = float(ks.statistic)
        p_value = float(ks.pvalue)
    else:
        statistic = float("nan")
        p_value = float("nan")
    if psi < 0.10:
        level = "Low"
    elif psi <= 0.25:
        level = "Moderate"
    else:
        level = "Significant"
    return {
        "psi": psi,
        "psi_level": level,
        "ks_statistic": statistic,
        "ks_p_value": p_value,
        "ks_drift": bool(np.isfinite(p_value) and p_value < alpha),
        "alpha": float(alpha),
        "reference_mean": float(np.mean(ref)) if len(ref) else None,
        "current_mean": float(np.mean(cur)) if len(cur) else None,
        "reference_median": float(np.median(ref)) if len(ref) else None,
        "current_median": float(np.median(cur)) if len(cur) else None,
        "reference_std": float(np.std(ref, ddof=0)) if len(ref) else None,
        "current_std": float(np.std(cur, ddof=0)) if len(cur) else None,
        "reference_missing_rate": float(pd.isna(reference).mean()),
        "current_missing_rate": float(pd.isna(current).mean()),
    }


calculate_numeric_drift = numeric_drift


def categorical_drift(reference: Any, current: Any) -> dict[str, Any]:
    ref = pd.Series(reference, dtype="object").fillna("__MISSING__").astype(str)
    cur = pd.Series(current, dtype="object").fillna("__MISSING__").astype(str)
    categories = sorted(set(ref.unique()).union(cur.unique()))
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).astype(int)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).astype(int)
    ref_props = ref_counts / max(ref_counts.sum(), 1)
    cur_props = cur_counts / max(cur_counts.sum(), 1)
    max_shift = float(np.max(np.abs(cur_props.to_numpy() - ref_props.to_numpy()))) if categories else 0.0
    chi_square_status = "suitable"
    chi_square: float | None = None
    p_value: float | None = None
    if len(categories) >= 2 and int(ref_counts.sum()) > 0 and int(cur_counts.sum()) > 0:
        table = np.vstack([ref_counts.to_numpy(), cur_counts.to_numpy()])
        row_totals = table.sum(axis=1)
        col_totals = table.sum(axis=0)
        grand = table.sum()
        expected = np.outer(row_totals, col_totals) / grand
        if np.any(expected < 1) or np.mean(expected >= 5) < 0.80:
            chi_square_status = "skipped_small_expected_frequencies"
        else:
            chi = chi2_contingency(table, correction=False)
            chi_square = float(chi.statistic)
            p_value = float(chi.pvalue)
    else:
        chi_square_status = "skipped_insufficient_categories"

    unseen = sorted(set(cur.unique()).difference(ref.unique()))
    disappeared = sorted(set(ref.unique()).difference(cur.unique()))
    return {
        "category_counts": {
            category: {
                "reference": int(ref_counts.loc[category]),
                "current": int(cur_counts.loc[category]),
            }
            for category in categories
        },
        "reference_proportions": {k: float(v) for k, v in ref_props.items()},
        "current_proportions": {k: float(v) for k, v in cur_props.items()},
        "proportion_shift": {k: float(cur_props[k] - ref_props[k]) for k in categories},
        "max_proportion_shift": max_shift,
        "unseen_categories": unseen,
        "disappeared_categories": disappeared,
        "chi_square": chi_square,
        "chi_square_p_value": p_value,
        "chi_square_status": chi_square_status,
    }


calculate_categorical_drift = categorical_drift


def calculate_numerical_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Calculate numerical drift for shared numeric columns."""
    results: dict[str, dict[str, Any]] = {}
    for column in reference_data.columns:
        if column in current_data and pd.api.types.is_numeric_dtype(reference_data[column]):
            results[column] = numeric_drift(
                reference_data[column],
                current_data[column],
                alpha=alpha,
            )
    return results


def calculate_categorical_drift_by_feature(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Calculate categorical drift for shared categorical columns."""
    results: dict[str, dict[str, Any]] = {}
    for column in reference_data.columns:
        if column in current_data and not pd.api.types.is_numeric_dtype(reference_data[column]):
            results[column] = categorical_drift(
                reference_data[column],
                current_data[column],
            )
    return results


def calculate_data_quality(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    warning_missing_delta: float = 0.05,
    critical_missing_delta: float = 0.10,
) -> dict[str, Any]:
    issues: list[str] = []
    missing = {}
    missing_delta = {}
    for column in current_data.columns:
        reference_rate = float(reference_data[column].isna().mean()) if column in reference_data else 0.0
        current_rate = float(current_data[column].isna().mean())
        missing[column] = current_rate
        missing_delta[column] = current_rate - reference_rate
        if missing_delta[column] >= critical_missing_delta:
            issues.append(f"Critical missing-value increase in {column}.")
        elif missing_delta[column] >= warning_missing_delta:
            issues.append(f"Warning missing-value increase in {column}.")

    duplicate_count = int(current_data.duplicated().sum())
    invalid_numeric = {}
    numeric_columns = {
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "promotion_last_5years",
        "quit",
    }
    out_of_range = {}
    for column in numeric_columns.intersection(current_data.columns):
        original = current_data[column]
        coerced = pd.to_numeric(original, errors="coerce")
        invalid = int(coerced.isna().sum() - original.isna().sum())
        if invalid:
            invalid_numeric[column] = invalid
    for column in {"satisfaction_level", "last_evaluation"}:
        if column in current_data:
            values = pd.to_numeric(current_data[column], errors="coerce")
            out_of_range[column] = int(((values < 0) | (values > 1)).sum())
    for column in {"average_montly_hours", "time_spend_company"}:
        if column in current_data:
            values = pd.to_numeric(current_data[column], errors="coerce")
            out_of_range[column] = int((values < 0).sum())

    out_of_range = {k: int(v) for k, v in out_of_range.items() if int(v) > 0}
    if out_of_range:
        issues.append("Out-of-range values detected.")
    if duplicate_count:
        issues.append("Duplicate rows detected.")

    if any("Critical" in issue for issue in issues):
        status = "Critical"
    elif issues:
        status = "Warning"
    else:
        status = "Healthy"
    return {
        "status": status,
        "missing_rates": missing,
        "missing_rate_delta": missing_delta,
        "duplicate_count": duplicate_count,
        "invalid_numeric": invalid_numeric,
        "out_of_range": out_of_range,
        "issues": issues,
    }


def calculate_prediction_drift(
    model_result: TrainingResult | OptimizationResult,
    current_data: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, Any]:
    features = current_data.drop(columns=["quit"], errors="ignore")
    probabilities = model_result.pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "row_count": int(len(features)),
        "mean": float(np.mean(probabilities)),
        "median": float(np.median(probabilities)),
        "std": float(np.std(probabilities, ddof=0)),
        "predicted_positive_rate": float(np.mean(predictions)),
        "threshold": float(threshold),
        "quantiles": np.quantile(probabilities, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


def calculate_delayed_performance(
    model_result: TrainingResult | OptimizationResult,
    current_data: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if "quit" not in current_data:
        return {
            "status": "Insufficient Data",
            "reason": "Ground-truth outcomes are not available.",
        }
    y_true = pd.to_numeric(current_data["quit"], errors="coerce")
    valid_mask = y_true.isin([0, 1]) & y_true.notna()
    y_true = y_true.loc[valid_mask].astype(int)
    features = current_data.drop(columns=["quit"]).loc[valid_mask]
    if y_true.nunique() < 2:
        return {
            "status": "Insufficient Data",
            "reason": "Valid binary ground truth requires both classes.",
        }
    probabilities = model_result.pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    return {
        "status": "Available",
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "valid_row_count": int(len(y_true)),
    }


evaluate_delayed_performance = calculate_delayed_performance


def build_monitoring_run_signature(report: dict[str, Any]) -> str:
    """Build a stable signature for duplicate-run prevention."""
    identity = {
        "reference_fingerprint": report.get("reference_fingerprint"),
        "monitoring_fingerprint": report.get("monitoring_fingerprint"),
        "model_identity": report.get("model_identity"),
        "threshold": report.get("threshold"),
    }
    return joblib.hash(identity)


def evaluate_monitoring_status(
    data_quality: dict[str, Any] | None = None,
    numerical_drift: dict[str, Any] | None = None,
    categorical_drift: dict[str, Any] | None = None,
    prediction_drift: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    status = "Healthy"

    if data_quality:
        quality_status = data_quality.get("status")
        if quality_status == "Critical":
            status = "Critical"
            reasons.append("Data-quality monitoring reported a critical issue.")
        elif quality_status == "Warning" and status == "Healthy":
            status = "Warning"
            reasons.append("Data-quality monitoring reported a warning.")

    if numerical_drift:
        for feature, summary in numerical_drift.items():
            if isinstance(summary, dict) and summary.get("psi_level") == "Significant":
                status = "Critical"
                reasons.append(f"Significant PSI drift detected in {feature}.")
            elif isinstance(summary, dict) and summary.get("psi_level") == "Moderate" and status == "Healthy":
                status = "Warning"
                reasons.append(f"Moderate PSI drift detected in {feature}.")

    if categorical_drift:
        for feature, summary in categorical_drift.items():
            max_shift = summary.get("max_proportion_shift")
            if isinstance(max_shift, (int, float)) and max_shift > 0.25:
                if status == "Healthy":
                    status = "Warning"
                reasons.append(f"Large category proportion shift detected in {feature}.")
            if isinstance(summary, dict) and summary.get("unseen_categories"):
                if status == "Healthy":
                    status = "Warning"
                reasons.append(f"Unseen categories detected in {feature}.")

    if prediction_drift and prediction_drift.get("insufficient_data"):
        if status == "Healthy":
            status = "Insufficient Data"
        reasons.append("Prediction monitoring has insufficient data.")

    if performance and performance.get("status") == "Insufficient Data" and status == "Healthy":
        status = "Insufficient Data"
        reasons.append(performance.get("reason", "Ground truth is unavailable."))

    if not reasons:
        reasons.append("No configured drift or data-quality threshold was exceeded.")
    return {"status": status, "reasons": reasons}


def build_feature_drift_csv(numerical_drift: dict[str, Any], categorical_drift: dict[str, Any]) -> bytes:
    rows: list[dict[str, Any]] = []
    for feature, summary in numerical_drift.items():
        rows.append({"feature": feature, "feature_type": "numeric", **summary})
    for feature, summary in categorical_drift.items():
        rows.append({
            "feature": feature,
            "feature_type": "categorical",
            "max_proportion_shift": summary.get("max_proportion_shift"),
            "unseen_categories": json.dumps(summary.get("unseen_categories", [])),
            "disappeared_categories": json.dumps(summary.get("disappeared_categories", [])),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=[
            "feature", "feature_type", "psi", "ks_statistic", "ks_p_value",
            "max_proportion_shift", "unseen_categories", "disappeared_categories",
        ])
    return frame.to_csv(index=False).encode("utf-8")


def build_aggregate_export(report: dict[str, Any]) -> dict[str, Any]:
    """Return a deliberately aggregate-only monitoring export."""
    export = json.loads(json.dumps(report, default=_json_default))
    forbidden = {"probabilities", "individual_predictions", "employee_id", "employee_name"}

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(child)
                for key, child in value.items()
                if key.lower() not in forbidden
            }
        if isinstance(value, list):
            return [scrub(child) for child in value]
        return value

    return scrub(export)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_monitoring_report(
    report: dict[str, Any],
    artifact_dir: str | Path = MONITORING_DIR,
) -> Path:
    target_dir = Path(artifact_dir)
    reports_dir = target_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = joblib.hash(report)
    report_path = reports_dir / f"{fingerprint[:12]}.json"
    report_path.write_text(
        json.dumps(build_aggregate_export(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def save_reference_metadata(
    metadata: dict[str, Any],
    artifact_dir: str | Path = MONITORING_DIR,
) -> Path:
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / REFERENCE_METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_monitoring_history(
    history: list[dict[str, Any]],
    artifact_dir: str | Path = MONITORING_DIR,
) -> Path:
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / MONITORING_HISTORY_FILENAME
    deduplicated: dict[str, dict[str, Any]] = {}
    for report in history:
        signature = report.get("run_signature")
        if not signature:
            signature = build_monitoring_run_signature(report)
        deduplicated[str(signature)] = report
    path.write_text(
        json.dumps([build_aggregate_export(v) for v in deduplicated.values()], indent=2),
        encoding="utf-8",
    )
    return path


def load_reference_metadata(
    artifact_dir: str | Path = MONITORING_DIR,
) -> dict[str, Any]:
    path = Path(artifact_dir) / REFERENCE_METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Monitoring reference metadata not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Corrupted monitoring reference '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Corrupted monitoring reference '{path}'.")
    return value


def load_monitoring_history(
    artifact_dir: str | Path = MONITORING_DIR,
) -> list[dict[str, Any]]:
    path = Path(artifact_dir) / MONITORING_HISTORY_FILENAME
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Corrupted monitoring history '{path}': {exc}") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"Corrupted monitoring history '{path}'.")
    return value
