from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model_monitoring import (
    build_aggregate_export,
    build_feature_drift_csv,
    build_monitoring_run_signature,
    calculate_categorical_drift,
    calculate_data_quality,
    calculate_delayed_performance,
    calculate_numeric_drift,
    calculate_prediction_drift,
    calculate_psi,
    create_reference_metadata,
    evaluate_monitoring_status,
    ingest_monitoring_file,
    load_monitoring_history,
    load_reference_metadata,
    save_monitoring_history,
    save_monitoring_report,
    save_reference_metadata,
    validate_monitoring_data,
)
from src.attrition import load_dataset, split_dataset, train_and_compare


DATA_PATH = Path(__file__).parents[1] / "data" / "employee_data.csv"


@pytest.fixture(scope="module")
def reference_and_current():
    data = load_dataset(DATA_PATH)
    _, reference, _, _, _ = split_dataset(data)
    current = reference.copy()
    current = current.drop(columns=["quit"], errors="ignore")
    return reference, current


def test_reference_metadata_is_aggregate_only(reference_and_current):
    reference, _ = reference_and_current
    metadata = create_reference_metadata(reference)
    assert metadata["row_count"] == len(reference)
    assert metadata["fingerprint"]
    assert "feature_schema" in metadata
    assert "numeric" in metadata
    assert "categorical" in metadata


def test_monitoring_file_ingestion_and_validation(tmp_path, reference_and_current):
    _, current = reference_and_current
    path = tmp_path / "batch.csv"
    current.to_csv(path, index=False)
    loaded = ingest_monitoring_file(path)
    validation = validate_monitoring_data(
        loaded,
        create_reference_metadata(current),
    )
    assert validation["row_count"] == len(loaded)
    assert validation["quit_status"] == "absent"
    assert validation["valid"]


def test_monitoring_file_rejects_empty(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_monitoring_file(path)


def test_txt_monitoring_file_is_parsed(tmp_path):
    path = tmp_path / "batch.txt"
    path.write_text(
        "satisfaction_level;last_evaluation\n0.4;0.5\n0.6;0.7\n",
        encoding="utf-8",
    )
    loaded = ingest_monitoring_file(path)
    assert len(loaded) == 2
    assert "satisfaction_level" in loaded.columns


def test_numeric_psi_identical_and_shifted():
    identical = calculate_psi(np.arange(100), np.arange(100))
    shifted = calculate_psi(np.zeros(100), np.ones(100) * 10)
    assert identical >= 0
    assert shifted > identical


def test_numeric_drift_summary():
    result = calculate_numeric_drift(np.arange(100), np.arange(100) + 5)
    assert "psi" in result
    assert "ks_statistic" in result
    assert "ks_p_value" in result
    assert "psi_level" in result


def test_categorical_drift_detects_unseen_and_disappeared():
    result = calculate_categorical_drift(
        pd.Series(["A", "A", "B"]),
        pd.Series(["B", "B", "C"]),
    )
    assert "C" in result["unseen_categories"]
    assert "A" in result["disappeared_categories"]


def test_validation_reports_missing_and_extra_columns(reference_and_current):
    _, current = reference_and_current
    incomplete = current.drop(columns=["department"])
    validation = validate_monitoring_data(incomplete)
    assert "department" in validation["missing_columns"]


def test_status_engine_combines_quality_and_drift():
    status = evaluate_monitoring_status(
        data_quality={"status": "Healthy"},
        numerical_drift={
            "age": {
                "psi_level": "Significant",
            }
        },
    )
    assert status["status"] == "Critical"
    assert status["reasons"]


def test_status_engine_insufficient_ground_truth():
    status = evaluate_monitoring_status(
        data_quality={"status": "Healthy"},
        performance={"status": "Insufficient Data", "reason": "Ground truth absent"},
    )
    assert status["status"] == "Insufficient Data"


def test_prediction_distribution_matches_model(reference_and_current):
    _, current = reference_and_current
    data = load_dataset(DATA_PATH)
    pos = data[data["quit"] == 1].head(60)
    neg = data[data["quit"] == 0].head(140)
    small = data.loc[pos.index.union(neg.index)].copy()
    results, _, _ = train_and_compare(small)
    prediction = calculate_prediction_drift(results["Logistic Regression"], current.head(40))
    assert prediction["row_count"] == 40
    assert 0 <= prediction["mean"] <= 1
    assert 0 <= prediction["predicted_positive_rate"] <= 1
    assert "probabilities" not in prediction


def test_delayed_performance_without_labels(reference_and_current):
    _, current = reference_and_current
    data = load_dataset(DATA_PATH)
    pos = data[data["quit"] == 1].head(60)
    neg = data[data["quit"] == 0].head(140)
    small = data.loc[pos.index.union(neg.index)].copy()
    results, _, _ = train_and_compare(small)
    result = calculate_delayed_performance(
        results["Logistic Regression"],
        current.head(40),
    )
    assert result["status"] == "Insufficient Data"


def test_delayed_performance_with_labels(reference_and_current):
    _, _ = reference_and_current
    data = load_dataset(DATA_PATH)
    pos = data[data["quit"] == 1].head(60)
    neg = data[data["quit"] == 0].head(140)
    small = data.loc[pos.index.union(neg.index)].copy()
    results, _, _ = train_and_compare(small)
    batch = pd.concat([
        small[small["quit"] == 0].head(20),
        small[small["quit"] == 1].head(20),
    ]).copy()
    result = calculate_delayed_performance(results["Logistic Regression"], batch)
    assert result["status"] == "Available"
    assert set(result["metrics"]) == {
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    }


def test_monitoring_persistence_is_aggregate_only(tmp_path, reference_and_current):
    reference, _ = reference_and_current
    metadata = create_reference_metadata(reference)
    metadata_path = save_reference_metadata(metadata, tmp_path)
    history_path = save_monitoring_history([{"status": "Healthy"}], tmp_path)
    report_path = save_monitoring_report({"status": "Healthy"}, tmp_path)
    assert metadata_path.exists()
    assert history_path.exists()
    assert report_path.exists()
    assert load_monitoring_history(tmp_path) == [{"status": "Healthy"}]


def test_feature_drift_csv_has_headers():
    payload = build_feature_drift_csv(
        {"tenure": {"psi": 0.1, "ks_p_value": 0.2}},
        {"salary": {"max_proportion_shift": 0.2, "unseen_categories": ["high"]}},
    )
    text = payload.decode("utf-8")
    assert "feature" in text
    assert "tenure" in text
    assert "salary" in text


def test_reference_metadata_round_trip(tmp_path, reference_and_current):
    reference, _ = reference_and_current
    metadata = create_reference_metadata(reference)
    save_reference_metadata(metadata, tmp_path)
    loaded = load_reference_metadata(tmp_path)
    assert loaded["fingerprint"] == metadata["fingerprint"]
    assert loaded["row_count"] == metadata["row_count"]


def test_dashboard_monitoring_contract():
    app_path = Path(__file__).parents[1] / "app.py"
    app_text = app_path.read_text(encoding="utf-8")
    assert "Model Monitoring & Data Drift" in app_text
    assert 'with st.form("m26_monitoring_form"):' in app_text
    assert 'run_monitoring = st.form_submit_button("Run monitoring for uploaded batch")' in app_text
    assert "m26_monitoring_report" in app_text
    assert "artifacts/monitoring" not in app_text


def test_single_class_delayed_performance_is_insufficient(tmp_path):
    model_data = load_dataset(DATA_PATH)
    pos = model_data[model_data["quit"] == 1].head(30)
    neg = model_data[model_data["quit"] == 0].head(70)
    small = model_data.loc[pos.index.union(neg.index)].copy()
    results, _, _ = train_and_compare(small)
    data = small.head(20).copy()
    data["quit"] = 0
    result = calculate_delayed_performance(
        results["Logistic Regression"],
        data,
    )
    assert result["status"] == "Insufficient Data"


def test_corrupted_monitoring_artifacts_raise_contextual_errors(tmp_path):
    (tmp_path / "monitoring_history.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Corrupted monitoring history"):
        load_monitoring_history(tmp_path)

    (tmp_path / "reference_metadata.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Corrupted monitoring reference"):
        load_reference_metadata(tmp_path)


def test_duplicate_run_signature_and_history_dedup(tmp_path):
    report = {
        "reference_fingerprint": "ref",
        "monitoring_fingerprint": "batch",
        "model_identity": "baseline:Logistic Regression",
        "threshold": 0.5,
    }
    signature = build_monitoring_run_signature(report)
    duplicate = dict(report)
    assert build_monitoring_run_signature(duplicate) == signature
    save_monitoring_history([report, duplicate], tmp_path)
    history = load_monitoring_history(tmp_path)
    assert len(history) == 1


def test_aggregate_export_and_prediction_report_exclude_individual_probability_rows():
    prediction = {
        "mean": 0.4,
        "median": 0.3,
        "probabilities": [0.1, 0.9],
        "individual_predictions": [0, 1],
    }
    report = build_aggregate_export({"prediction_distribution": prediction})
    exported = report["prediction_distribution"]
    assert "probabilities" not in exported
    assert "individual_predictions" not in exported


def test_zero_out_of_range_data_quality_is_healthy():
    reference = pd.DataFrame({
        "satisfaction_level": [0.4, 0.5, 0.6],
    })
    current = pd.DataFrame({
        "satisfaction_level": [0.4, 0.5, 0.6],
    })
    quality = calculate_data_quality(reference, current)
    assert quality["status"] == "Healthy"
    assert quality["out_of_range"] == {}


def test_partial_quit_labels_are_not_marked_valid_binary():
    data = pd.DataFrame({
        "quit": [0, None],
        "satisfaction_level": [0.4, 0.5],
    })
    validation = validate_monitoring_data(data)
    assert validation["quit_status"] == "partial_binary"
    assert validation["quit_valid"] is False


def test_invalid_quit_labels_are_not_valid_binary():
    data = pd.DataFrame({
        "quit": [0, 1, 2],
    })
    validation = validate_monitoring_data(data)
    assert validation["quit_status"] == "invalid_labels"
    assert validation["quit_valid"] is False


def test_numeric_strings_are_reported_as_invalid_numeric_values():
    reference = pd.DataFrame({
        "satisfaction_level": [0.4, 0.5, 0.6],
    })
    current = pd.DataFrame({
        "satisfaction_level": ["bad", "0.5", "0.6"],
    })
    quality = calculate_data_quality(reference, current)
    assert quality["invalid_numeric"]["satisfaction_level"] == 1


def test_history_deduplicates_by_run_signature(tmp_path):
    first = {
        "model_identity": "baseline:Random Forest",
        "reference_fingerprint": "reference",
        "monitoring_fingerprint": "batch",
        "threshold": 0.5,
        "status": {"status": "Critical"},
    }
    second = dict(first)
    second["status"] = {"status": "Warning"}
    save_monitoring_history([first, second], tmp_path)
    history = load_monitoring_history(tmp_path)
    assert len(history) == 1
