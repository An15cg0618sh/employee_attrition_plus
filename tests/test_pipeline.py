from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attrition import (
    clean_dataset,
    load_model,
    load_dataset,
    save_model,
    threshold_metrics,
    train_and_compare,
    validate_dataset_frame,
)
from src.explainability import explain_samples, get_transformed_feature_names

DATA = Path(__file__).parents[1] / "data" / "employee_data.csv"

def test_dataset_loads_and_target_is_binary():
    data = load_dataset(DATA)
    assert "quit" in data.columns
    assert set(data["quit"].unique()).issubset({0, 1})

def test_duplicate_cleaning_reports_removed_rows():
    data = load_dataset(DATA)
    cleaned, report = clean_dataset(data)
    assert len(cleaned) <= len(data)
    assert report["duplicates_removed"] == len(data) - len(cleaned)


def test_saved_model_round_trip(tmp_path):
    data = load_dataset(DATA)
    results, _, _ = train_and_compare(data)
    original = results["Random Forest"]
    model_path = tmp_path / "random_forest.joblib"

    save_model(original, model_path)
    restored = load_model(model_path)
    sample = original.X_test.iloc[[0]]

    assert restored.predict_proba(sample).shape == (1, 2)
    np.testing.assert_allclose(
        restored.predict_proba(sample),
        original.pipeline.predict_proba(sample),
        rtol=1e-12,
        atol=1e-12,
    )


def test_app_uses_streamlit_compatible_shap_notation():
    app_text = (DATA.parents[1] / "app.py").read_text(encoding="utf-8")
    assert r"\\operatorname{mean}(\\lvert \\text{SHAP} \\rvert)" in app_text
    assert r"\\text{mean}(|\\text{SHAP}|)" not in app_text
    assert r"\(" not in app_text

def test_training_pipeline_returns_both_models():
    data = load_dataset(DATA)
    results, comparison, report = train_and_compare(data)
    assert set(results) == {"Logistic Regression", "Random Forest"}
    assert len(comparison) == 2
    assert 0 <= results["Random Forest"].metrics["roc_auc"] <= 1

def test_shap_explainability_for_supported_models():
    data = load_dataset(DATA)
    results, _, _ = train_and_compare(data)
    X = data.drop(columns=["quit"]).head(25)

    for name, result in results.items():
        explanation = explain_samples(
            result.pipeline, X, background=X.head(10),
            max_samples=10, max_background=10,
        )
        assert explanation.values.shape[0] == 10
        assert explanation.values.shape[1] == len(explanation.feature_names)
        assert explanation.positive_class_index == 1
        assert set(explanation.feature_names)

def test_shap_feature_names_align_with_transformed_data():
    data = load_dataset(DATA)
    results, _, _ = train_and_compare(data)
    X = data.drop(columns=["quit"]).head(5)
    for result in results.values():
        explanation = explain_samples(
            result.pipeline, X, background=X,
            max_samples=5, max_background=5,
        )
        names = get_transformed_feature_names(result.pipeline)
        assert explanation.feature_names == names
        assert explanation.transformed_values.shape[1] == len(names)

def test_shap_rejects_empty_and_invalid_input():
    data = load_dataset(DATA)
    results, _, _ = train_and_compare(data)
    result = results["Logistic Regression"]
    try:
        explain_samples(result.pipeline, pd.DataFrame())
        raise AssertionError("Empty input should fail")
    except ValueError:
        pass


def test_validate_dataset_frame_rejects_invalid_schemas():
    data = load_dataset(DATA)

    with pytest.raises(ValueError, match="no rows"):
        validate_dataset_frame(data.iloc[0:0])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_frame(data.drop(columns=["quit"]))

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_frame(data.drop(columns=["satisfaction_level"]))

    bad_target = data.head(10).copy()
    bad_target.loc[bad_target.index[0], "quit"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_dataset_frame(bad_target)

    nan_target = data.head(10).copy()
    nan_target.loc[nan_target.index[0], "quit"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_dataset_frame(nan_target)


def test_training_rejects_tiny_dataset():
    data = load_dataset(DATA).head(5).copy()
    data["quit"] = 0
    with pytest.raises(ValueError, match="both classes"):
        train_and_compare(data)


def test_threshold_metrics_and_unseen_categories():
    data = load_dataset(DATA)
    results, _, _ = train_and_compare(data)
    rf = results["Random Forest"]

    low_metrics, low_preds = threshold_metrics(rf, 0.20)
    high_metrics, high_preds = threshold_metrics(rf, 0.80)
    assert low_preds.sum() >= high_preds.sum()
    assert low_metrics["recall"] >= high_metrics["recall"]

    unseen = rf.X_test.head(2).copy()
    unseen["department"] = "unseen_dept_xyz"
    unseen["salary"] = "unseen_salary_xyz"
    probs = rf.pipeline.predict_proba(unseen)[:, 1]
    assert len(probs) == 2
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
