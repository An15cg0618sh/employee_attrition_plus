from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.attrition import build_preprocessor, load_dataset, train_and_compare
from src.explainability import (
    ShapExplanationResult,
    build_explainer,
    clean_feature_names,
    compute_global_shap_explanations,
    explain_single_prediction,
    extract_preprocessor_and_model,
    get_feature_impact_summary,
    get_shap_feature_importance,
    plot_shap_importance_bar,
    plot_shap_summary,
    plot_shap_waterfall,
)

DATA_PATH = Path(__file__).parents[1] / "data" / "employee_data.csv"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="module")
def trained_models(dataset):
    results, comparison, _ = train_and_compare(dataset)
    return results


def test_clean_feature_names():
    raw_names = [
        "numeric__satisfaction_level",
        "numeric__average_montly_hours",
        "categorical__department_sales",
        "categorical__department_IT",
        "categorical__salary_low",
        "other_feature",
    ]
    cleaned = clean_feature_names(raw_names)
    assert cleaned == [
        "satisfaction_level",
        "average_montly_hours",
        "department: sales",
        "department: IT",
        "salary: low",
        "other_feature",
    ]


def test_build_explainer_supported_models(trained_models):
    rf_pipe = trained_models["Random Forest"].pipeline
    lr_pipe = trained_models["Logistic Regression"].pipeline
    X_test = trained_models["Random Forest"].X_test

    rf_explainer, rf_type = build_explainer(rf_pipe)
    assert rf_type == "tree"

    lr_explainer, lr_type = build_explainer(lr_pipe, background_df=X_test)
    assert lr_type == "linear"


def test_explainer_unsupported_model_raises(trained_models):
    X_test = trained_models["Random Forest"].X_test
    preprocessor = build_preprocessor(X_test)
    dummy_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", SVC(probability=True)),
    ])
    with pytest.raises(ValueError, match="Unsupported model type"):
        build_explainer(dummy_pipe, background_df=X_test)


def test_pipeline_validation_errors(trained_models):
    with pytest.raises(TypeError, match="Expected a scikit-learn Pipeline"):
        extract_preprocessor_and_model("not_a_pipeline")

    incomplete_pipe = Pipeline([("only_model", LogisticRegression())])
    with pytest.raises(ValueError, match="must contain 'preprocess' and 'model'"):
        extract_preprocessor_and_model(incomplete_pipe)


def test_compute_global_shap_both_models(trained_models):
    for name in ["Random Forest", "Logistic Regression"]:
        pipe = trained_models[name].pipeline
        X_test = trained_models[name].X_test

        result = compute_global_shap_explanations(
            pipe, X_test, background_df=X_test, max_samples=25, random_state=42
        )

        assert isinstance(result, ShapExplanationResult)
        assert result.explanation.shape[0] == 25
        assert result.explanation.shape[1] == len(result.feature_names)
        assert len(result.explanation.shape) == 2  # 2D shape for positive class quit=1
        assert isinstance(result.base_value, float)
        assert len(result.raw_features) == 25
        assert result.transformed_features.shape == (25, len(result.feature_names))


def test_shap_feature_importance_and_impact(trained_models):
    rf_pipe = trained_models["Random Forest"].pipeline
    X_test = trained_models["Random Forest"].X_test

    result = compute_global_shap_explanations(rf_pipe, X_test, max_samples=30)
    importance_df = get_shap_feature_importance(result)

    assert not importance_df.empty
    assert list(importance_df.columns) == ["feature", "mean_abs_shap", "mean_shap", "importance_share"]
    # Verify strictly descending order
    assert (importance_df["mean_abs_shap"].diff().dropna() <= 1e-9).all()
    # Verify importance share sums to ~1.0
    assert pytest.approx(importance_df["importance_share"].sum(), 0.01) == 1.0

    impacts = get_feature_impact_summary(result, top_n=5)
    assert len(impacts) == 5
    for item in impacts:
        assert "feature" in item
        assert "tendency" in item
        assert "correlation" in item
        assert "impact_description" in item


def test_single_prediction_explanation(trained_models):
    rf_pipe = trained_models["Random Forest"].pipeline
    lr_pipe = trained_models["Logistic Regression"].pipeline
    X_test = trained_models["Random Forest"].X_test

    sample_df = X_test.iloc[[0]]
    sample_series = X_test.iloc[0]

    # Test with DataFrame
    local_rf = explain_single_prediction(rf_pipe, sample_df)
    assert local_rf["predicted_class"] in {0, 1}
    assert 0.0 <= local_rf["attrition_probability"] <= 1.0
    assert "explanation" in local_rf
    assert len(local_rf["positive_contributors"]) + len(local_rf["negative_contributors"]) <= len(local_rf["feature_names"])

    # Test with Series
    local_series = explain_single_prediction(rf_pipe, sample_series)
    assert local_series["predicted_class"] == local_rf["predicted_class"]

    # Test with Logistic Regression
    local_lr = explain_single_prediction(lr_pipe, sample_df, background_df=X_test)
    assert local_lr["predicted_class"] in {0, 1}
    assert 0.0 <= local_lr["attrition_probability"] <= 1.0


@pytest.mark.parametrize("model_name", ["Random Forest", "Logistic Regression"])
def test_positive_class_reconstruction_uses_all_features(trained_models, model_name):
    """Base plus every transformed SHAP value reconstructs the native model output."""
    pipe = trained_models[model_name].pipeline
    X_test = trained_models[model_name].X_test
    sample = X_test.iloc[[0]]
    local = explain_single_prediction(pipe, sample, background_df=X_test)

    assert len(local["all_contributors"]) == len(local["feature_names"])
    sum_all = float(sum(c["shap_value"] for c in local["all_contributors"]))
    assert sum_all == pytest.approx(local["sum_all_shap_values"], abs=1e-10)
    assert local["base_value"] + sum_all == pytest.approx(
        local["reconstructed_output"], abs=1e-10
    )

    if model_name == "Random Forest":
        assert local["output_space"] == "probability"
        model_output = float(pipe.predict_proba(sample)[0, 1])
        assert local["reconstructed_probability"] == pytest.approx(model_output, abs=1e-8)
    else:
        assert local["output_space"] == "log-odds (margin)"
        transformed = pipe.named_steps["preprocess"].transform(sample)
        model_output = float(pipe.named_steps["model"].decision_function(transformed)[0])
        assert local["reconstructed_output"] == pytest.approx(model_output, abs=1e-8)
        sigmoid_output = 1.0 / (1.0 + np.exp(-model_output))
        assert local["reconstructed_probability"] == pytest.approx(sigmoid_output, abs=1e-8)
        assert local["reconstructed_probability"] == pytest.approx(
            float(pipe.predict_proba(sample)[0, 1]), abs=1e-8
        )


@pytest.mark.parametrize("model_name", ["Random Forest", "Logistic Regression"])
def test_categorical_names_align_with_reconstruction_values(trained_models, model_name):
    pipe = trained_models[model_name].pipeline
    X_test = trained_models[model_name].X_test
    local = explain_single_prediction(pipe, X_test.iloc[[0]], background_df=X_test)

    assert len(local["feature_names"]) == len(local["all_contributors"])
    assert len(set(local["feature_names"])) == len(local["feature_names"])
    assert any(name.startswith("department:") for name in local["feature_names"])
    assert any(name.startswith("salary:") for name in local["feature_names"])


@pytest.mark.parametrize("model_name", ["Random Forest", "Logistic Regression"])
def test_global_reconstruction_matches_native_model_output(trained_models, model_name):
    pipe = trained_models[model_name].pipeline
    X_test = trained_models[model_name].X_test
    result = compute_global_shap_explanations(
        pipe, X_test, background_df=X_test, max_samples=25, random_state=42
    )

    assert result.values.shape == (25, len(result.feature_names))
    reconstructed = result.reconstructed_outputs
    assert reconstructed is not None
    assert len(reconstructed) == 25

    if model_name == "Random Forest":
        model_output = pipe.predict_proba(result.raw_features)[:, 1]
        assert reconstructed == pytest.approx(model_output, abs=1e-8)
    else:
        transformed = pipe.named_steps["preprocess"].transform(result.raw_features)
        model_output = pipe.named_steps["model"].decision_function(transformed)
        assert reconstructed == pytest.approx(model_output, abs=1e-8)


def test_invalid_input_handling(trained_models):
    rf_pipe = trained_models["Random Forest"].pipeline
    X_test = trained_models["Random Forest"].X_test

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=X_test.columns)
    with pytest.raises(ValueError, match="Cannot compute SHAP explanations for an empty dataset"):
        compute_global_shap_explanations(rf_pipe, empty_df)

    # Multi-row input to explain_single_prediction
    with pytest.raises(ValueError, match="Expected exactly 1 row"):
        explain_single_prediction(rf_pipe, X_test.iloc[:2])

    # Wrong type to explain_single_prediction
    with pytest.raises(TypeError, match="must be a pandas DataFrame or Series"):
        explain_single_prediction(rf_pipe, [1, 2, 3])


def test_plots_generate_figures(trained_models):
    rf_pipe = trained_models["Random Forest"].pipeline
    X_test = trained_models["Random Forest"].X_test

    result = compute_global_shap_explanations(rf_pipe, X_test, max_samples=20)
    imp_df = get_shap_feature_importance(result)

    fig_summary = plot_shap_summary(result, max_display=5)
    assert isinstance(fig_summary, plt.Figure)
    plt.close(fig_summary)

    fig_bar = plot_shap_importance_bar(imp_df, top_n=5)
    assert isinstance(fig_bar, plt.Figure)
    plt.close(fig_bar)

    local = explain_single_prediction(rf_pipe, X_test.iloc[[0]])
    fig_wf = plot_shap_waterfall(local["explanation"], max_display=5)
    assert isinstance(fig_wf, plt.Figure)
    plt.close(fig_wf)


def test_reconstruction_property_random_forest(trained_models):
    """Verify that baseline + sum(all_shap_values) exactly reconstructs predicted probability for Random Forest."""
    rf_pipe = trained_models["Random Forest"].pipeline
    X_test = trained_models["Random Forest"].X_test

    for idx in [0, 5, 10, 42]:
        sample = X_test.iloc[[idx]]
        local = explain_single_prediction(rf_pipe, sample, background_df=X_test)
        
        base_val = local["base_value"]
        shap_sum = local["sum_all_shap_values"]
        reconstructed = base_val + shap_sum
        actual_prob = local["attrition_probability"]

        assert pytest.approx(actual_prob, abs=1e-6) == reconstructed
        assert pytest.approx(actual_prob, abs=1e-6) == local["reconstructed_probability"]
        assert local["output_space"] == "probability"


def test_reconstruction_property_logistic_regression(trained_models):
    """Verify that baseline + sum(all_shap_values) reconstructs log-odds margin and probability for Logistic Regression."""
    lr_pipe = trained_models["Logistic Regression"].pipeline
    X_test = trained_models["Logistic Regression"].X_test
    lr_model = lr_pipe.named_steps["model"]
    preprocessor = lr_pipe.named_steps["preprocess"]

    for idx in [0, 5, 10, 42]:
        sample = X_test.iloc[[idx]]
        local = explain_single_prediction(lr_pipe, sample, background_df=X_test)

        base_val = local["base_value"]
        shap_sum = local["sum_all_shap_values"]
        reconstructed_margin = base_val + shap_sum

        sample_trans = preprocessor.transform(sample)
        if hasattr(sample_trans, "toarray"):
            sample_trans = sample_trans.toarray()
        actual_margin = float(lr_model.decision_function(sample_trans)[0])

        assert pytest.approx(actual_margin, abs=1e-6) == reconstructed_margin
        reconstructed_prob = 1.0 / (1.0 + np.exp(-reconstructed_margin))
        actual_prob = local["attrition_probability"]
        assert pytest.approx(actual_prob, abs=1e-6) == reconstructed_prob
        assert pytest.approx(actual_prob, abs=1e-6) == local["reconstructed_probability"]
        assert local["output_space"] == "log-odds (margin)"


def test_global_batch_reconstruction_property(trained_models):
    """Verify that batch explanations preserve additivity across all instances."""
    rf_pipe = trained_models["Random Forest"].pipeline
    X_test = trained_models["Random Forest"].X_test

    batch_res = compute_global_shap_explanations(rf_pipe, X_test, max_samples=10, random_state=42)
    sample_trans = batch_res.transformed_values
    rf_model = rf_pipe.named_steps["model"]
    actual_probs = rf_model.predict_proba(sample_trans)[:, 1]

    # Row-wise sum of shap values + base value
    row_sums = np.sum(batch_res.explanation.values, axis=1) + batch_res.base_value
    np.testing.assert_allclose(row_sums, actual_probs, atol=1e-6)

