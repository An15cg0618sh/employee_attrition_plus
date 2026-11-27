"""SHAP Explainable AI module for Employee Attrition Prediction.

This module provides model-agnostic and model-specific SHAP explanations
for the scikit-learn models (Random Forest and Logistic Regression) trained
in the workforce analytics pipeline.

Responsible AI Note:
SHAP values describe statistical contributions to a model's mathematical output
relative to a baseline expectation. They reflect model behavior and data
correlations, NOT actual employee motives or real-world causal mechanisms.
They must NEVER be used to automate employment decisions, rankings, or terminations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


SUPPORTED_MODELS = (RandomForestClassifier, LogisticRegression)


@dataclass
class ShapExplanationResult:
    """Container for SHAP explanations of the positive class (quit = 1)."""
    model_name: str
    explanation: shap.Explanation
    feature_names: list[str]
    raw_features: pd.DataFrame
    transformed_values: np.ndarray
    base_value: float
    positive_class_index: int = 1
    output_scale: str = "probability"
    reconstructed_outputs: np.ndarray | None = None

    @property
    def values(self) -> np.ndarray:
        return self.explanation.values

    @property
    def transformed_features(self) -> np.ndarray:
        return self.transformed_values


def clean_feature_names(raw_names: list[str] | np.ndarray) -> list[str]:
    """Clean feature names produced by ColumnTransformer.

    Strips pipeline prefixes ('numeric__', 'categorical__') and formats
    one-hot encoded categorical columns cleanly (e.g. 'department: sales',
    'salary: low').
    """
    clean: list[str] = []
    for raw in raw_names:
        name = str(raw)
        if name.startswith("numeric__"):
            clean.append(name.replace("numeric__", ""))
        elif name.startswith("categorical__"):
            remainder = name.replace("categorical__", "")
            if remainder.startswith("department_"):
                val = remainder[len("department_"):]
                clean.append(f"department: {val}")
            elif remainder.startswith("salary_"):
                val = remainder[len("salary_"):]
                clean.append(f"salary: {val}")
            else:
                clean.append(remainder.replace("_", ": "))
        else:
            clean.append(name)
    return clean


def get_transformed_feature_names(pipeline_or_preprocessor: Pipeline | ColumnTransformer) -> list[str]:
    """Return clean feature names for the transformed feature space."""
    if isinstance(pipeline_or_preprocessor, Pipeline):
        if "preprocess" not in pipeline_or_preprocessor.named_steps:
            raise ValueError("Pipeline does not contain a 'preprocess' step.")
        preprocessor = pipeline_or_preprocessor.named_steps["preprocess"]
    elif isinstance(pipeline_or_preprocessor, ColumnTransformer):
        preprocessor = pipeline_or_preprocessor
    else:
        raise TypeError(
            f"Expected Pipeline or ColumnTransformer, got {type(pipeline_or_preprocessor).__name__}"
        )

    raw_names = preprocessor.get_feature_names_out()
    return clean_feature_names(raw_names)


def extract_preprocessor_and_model(pipeline: Pipeline) -> tuple[ColumnTransformer, Any]:
    """Extract and validate the preprocessor and estimator from a fitted pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected a scikit-learn Pipeline, got {type(pipeline).__name__}")

    if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError(
            "Pipeline must contain 'preprocess' and 'model' named steps. "
            f"Found: {list(pipeline.named_steps.keys())}"
        )

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    if not isinstance(model, SUPPORTED_MODELS):
        raise ValueError(
            f"Unsupported model type '{type(model).__name__}'. "
            f"Supported models are: {[m.__name__ for m in SUPPORTED_MODELS]}"
        )

    return preprocessor, model


def get_transformed_data_and_names(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Transform input DataFrame and return dense array and clean feature names."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Input X must be a pandas DataFrame, got {type(X).__name__}")

    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float64)

    raw_names = preprocessor.get_feature_names_out()
    feature_names = clean_feature_names(raw_names)
    return transformed, feature_names


def build_explainer(
    pipeline: Pipeline,
    background_df: pd.DataFrame | None = None,
    max_background_samples: int = 100,
    random_state: int = 42,
) -> tuple[Any, str]:
    """Build the appropriate SHAP explainer for the given pipeline model.

    Uses TreeExplainer for RandomForestClassifier and LinearExplainer with an
    Independent masker for LogisticRegression.
    """
    preprocessor, model = extract_preprocessor_and_model(pipeline)

    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        return explainer, "tree"

    if isinstance(model, LogisticRegression):
        if background_df is None or len(background_df) == 0:
            raise ValueError(
                "A background dataset is required to initialize LinearExplainer for Logistic Regression."
            )
        bg_sample = background_df
        if len(bg_sample) > max_background_samples:
            bg_sample = bg_sample.sample(n=max_background_samples, random_state=random_state)

        bg_trans, _ = get_transformed_data_and_names(preprocessor, bg_sample)
        masker = shap.maskers.Independent(data=bg_trans)
        explainer = shap.LinearExplainer(model, masker=masker, model_output="raw")
        return explainer, "linear"

    raise ValueError(f"No SHAP explainer configured for model type {type(model).__name__}")


def explain_samples(
    pipeline: Pipeline,
    X: pd.DataFrame,
    background: pd.DataFrame | None = None,
    max_samples: int = 200,
    max_background: int = 100,
    random_state: int = 42,
) -> ShapExplanationResult:
    """Compute SHAP explanations for input samples focusing on positive class (quit = 1)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Input X must be a pandas DataFrame, got {type(X).__name__}")

    if len(X) == 0:
        raise ValueError("Cannot compute SHAP explanations for an empty dataset.")

    preprocessor, model = extract_preprocessor_and_model(pipeline)
    model_name = type(model).__name__

    if background is None:
        background = X

    if len(X) > max_samples:
        explained_df = X.sample(n=max_samples, random_state=random_state).copy()
    else:
        explained_df = X.copy()

    X_trans, feature_names = get_transformed_data_and_names(preprocessor, explained_df)
    explainer, explainer_type = build_explainer(
        pipeline, background_df=background, max_background_samples=max_background, random_state=random_state
    )

    try:
        raw_exp = explainer(X_trans)
    except Exception as exc:
        raise RuntimeError(f"SHAP explanation calculation failed for {model_name}: {exc}") from exc

    if explainer_type == "tree":
        if len(raw_exp.shape) == 3 and raw_exp.shape[2] == 2:
            exp_pos = raw_exp[:, :, 1]
            base_values = np.asarray(raw_exp.base_values)
            if base_values.ndim == 2:
                base_val = float(np.mean(base_values[:, 1]))
            else:
                base_val = float(np.mean(base_values))
            output_scale = "probability"
        elif len(raw_exp.shape) == 2:
            exp_pos = raw_exp
            base_val = float(np.mean(np.asarray(raw_exp.base_values)))
            output_scale = "probability"
        else:
            raise ValueError(f"Unexpected shape from TreeExplainer: {raw_exp.shape}")
    else:
        exp_pos = raw_exp
        base_val = float(np.mean(raw_exp.base_values)) if hasattr(raw_exp, "base_values") else 0.0
        output_scale = "log-odds"

    exp_pos.feature_names = feature_names
    exp_pos.data = X_trans
    per_row_bases = np.asarray(exp_pos.base_values).reshape(-1)
    if len(per_row_bases) == 1 and len(exp_pos.values) > 1:
        per_row_bases = np.repeat(per_row_bases, len(exp_pos.values))
    reconstructed_outputs = per_row_bases + np.sum(exp_pos.values, axis=1)

    return ShapExplanationResult(
        model_name=model_name,
        explanation=exp_pos,
        feature_names=feature_names,
        raw_features=explained_df.reset_index(drop=True),
        transformed_values=X_trans,
        base_value=base_val,
        positive_class_index=1,
        output_scale=output_scale,
        reconstructed_outputs=reconstructed_outputs,
    )


def compute_global_shap_explanations(
    pipeline: Pipeline,
    X: pd.DataFrame,
    background_df: pd.DataFrame | None = None,
    max_samples: int = 200,
    random_state: int = 42,
) -> ShapExplanationResult:
    """Convenience alias for explain_samples."""
    return explain_samples(
        pipeline=pipeline,
        X=X,
        background=background_df,
        max_samples=max_samples,
        max_background=100,
        random_state=random_state,
    )


def get_shap_feature_importance(result: ShapExplanationResult) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across explained samples."""
    vals = result.explanation.values
    mean_abs = np.mean(np.abs(vals), axis=0)
    mean_shap = np.mean(vals, axis=0)

    df = pd.DataFrame({
        "feature": result.feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    total_importance = df["mean_abs_shap"].sum()
    if total_importance > 0:
        df["importance_share"] = df["mean_abs_shap"] / total_importance
    else:
        df["importance_share"] = 0.0

    return df


def get_feature_impact_summary(result: ShapExplanationResult, top_n: int = 6) -> list[dict[str, Any]]:
    """Compute direction of feature impact on predicted attrition risk.

    Analyzes correlation between feature value and SHAP contribution to
    summarize whether higher feature values generally associate with higher
    or lower model attrition output.

    Note: This is a statistical summary of model behaviour, NOT real-world causation.
    """
    importance_df = get_shap_feature_importance(result)
    top_features = importance_df.head(top_n)

    summaries = []
    vals = result.explanation.values
    data = result.transformed_values

    for _, row in top_features.iterrows():
        feat_name = row["feature"]
        idx = result.feature_names.index(feat_name)
        feat_vals = data[:, idx]
        shap_vals = vals[:, idx]

        std_feat = float(np.std(feat_vals))
        std_shap = float(np.std(shap_vals))

        if std_feat > 1e-8 and std_shap > 1e-8:
            corr = float(np.corrcoef(feat_vals, shap_vals)[0, 1])
        else:
            corr = 0.0

        if corr >= 0.20:
            impact_desc = "Higher values generally increase predicted attrition risk relative to baseline."
            tendency = "Positive impact (increases risk)"
        elif corr <= -0.20:
            impact_desc = "Higher values generally decrease predicted attrition risk relative to baseline."
            tendency = "Negative impact (decreases risk)"
        else:
            impact_desc = "Non-linear or mixed relationship with predicted attrition risk across employee profiles."
            tendency = "Mixed / non-linear impact"

        summaries.append({
            "feature": feat_name,
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "correlation": corr,
            "tendency": tendency,
            "impact_description": impact_desc,
        })

    return summaries


def explain_single_prediction(
    pipeline: Pipeline,
    sample: pd.DataFrame | pd.Series,
    background_df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute local SHAP explanation for a single employee record."""
    preprocessor, model = extract_preprocessor_and_model(pipeline)

    if isinstance(sample, pd.Series):
        sample_df = sample.to_frame().T
    elif isinstance(sample, pd.DataFrame):
        if len(sample) != 1:
            raise ValueError(f"Expected exactly 1 row for local explanation, got {len(sample)}")
        sample_df = sample.copy()
    else:
        raise TypeError(f"Sample must be a pandas DataFrame or Series, got {type(sample).__name__}")

    if "quit" in sample_df.columns:
        sample_features = sample_df.drop(columns=["quit"])
    else:
        sample_features = sample_df

    try:
        probabilities = pipeline.predict_proba(sample_features)[0]
    except Exception as exc:
        raise RuntimeError(f"Prediction failed on sample: {exc}") from exc

    prob_quit = float(probabilities[1])
    pred_class = int(prob_quit >= 0.5)

    X_trans, feature_names = get_transformed_data_and_names(preprocessor, sample_features)

    explainer, explainer_type = build_explainer(
        pipeline, background_df=background_df, max_background_samples=100, random_state=random_state
    )

    try:
        raw_exp = explainer(X_trans)
    except Exception as exc:
        raise RuntimeError(f"Local SHAP explanation failed: {exc}") from exc

    if explainer_type == "tree":
        if len(raw_exp.shape) == 3 and raw_exp.shape[2] == 2:
            single_exp = raw_exp[0, :, 1]
            base_values = np.asarray(raw_exp.base_values)
            if base_values.ndim == 2:
                base_val = float(base_values[0, 1])
            else:
                base_val = float(np.asarray(single_exp.base_values).reshape(-1)[0])
        elif len(raw_exp.shape) == 2:
            single_exp = raw_exp[0]
            base_val = float(np.asarray(single_exp.base_values).reshape(-1)[0])
        else:
            raise ValueError(f"Unexpected explanation shape from TreeExplainer: {raw_exp.shape}")
    else:
        single_exp = raw_exp[0]
        base_val = float(np.asarray(single_exp.base_values).reshape(-1)[0])

    single_exp.feature_names = feature_names
    single_exp.data = X_trans[0]

    contributors = []
    for name, shap_val, feat_val in zip(feature_names, single_exp.values, X_trans[0]):
        contributors.append({
            "feature": name,
            "shap_value": float(shap_val),
            "feature_value": float(feat_val),
        })

    pos_contributors = sorted(
        [c for c in contributors if c["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True,
    )
    neg_contributors = sorted(
        [c for c in contributors if c["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )

    output_space = "probability" if explainer_type == "tree" else "log-odds (margin)"
    sum_all_shap = float(np.sum(single_exp.values))
    reconstructed_output = base_val + sum_all_shap

    if output_space == "probability":
        reconstructed_prob = reconstructed_output
    else:
        reconstructed_prob = float(1.0 / (1.0 + np.exp(-reconstructed_output)))

    # Sum of features outside top 4 pos and top 4 neg
    displayed_feats = set([c["feature"] for c in pos_contributors[:4]] + [c["feature"] for c in neg_contributors[:4]])
    other_contributors = [c for c in contributors if c["feature"] not in displayed_feats]
    other_sum = float(sum(c["shap_value"] for c in other_contributors))

    return {
        "predicted_class": pred_class,
        "predicted_label": "Left (1)" if pred_class == 1 else "Stayed (0)",
        "attrition_probability": prob_quit,
        "base_value": base_val,
        "output_space": output_space,
        "sum_all_shap_values": sum_all_shap,
        "reconstructed_output": reconstructed_output,
        "reconstructed_probability": reconstructed_prob,
        "explanation": single_exp,
        "raw_record": sample_features.iloc[0].to_dict(),
        "feature_names": feature_names,
        "positive_contributors": pos_contributors,
        "negative_contributors": neg_contributors,
        "other_contributors_count": len(other_contributors),
        "other_contributors_sum": other_sum,
        "all_contributors": sorted(contributors, key=lambda x: abs(x["shap_value"]), reverse=True),
    }


def plot_shap_summary(
    result: ShapExplanationResult,
    max_display: int = 10,
    title: str = "SHAP Summary Plot (Impact on 'Quit = 1')",
) -> plt.Figure:
    """Generate and return a matplotlib Figure with the SHAP summary plot."""
    fig = plt.figure(figsize=(9, 5))
    shap.summary_plot(
        result.explanation.values,
        result.transformed_values,
        feature_names=result.feature_names,
        max_display=max_display,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    single_exp: shap.Explanation,
    max_display: int = 10,
    title: str = "Local SHAP Waterfall (Relative to Baseline)",
) -> plt.Figure:
    """Generate and return a matplotlib Figure with the SHAP waterfall plot."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(single_exp, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    return fig


def plot_shap_importance_bar(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top Features by Mean Absolute SHAP Value",
) -> plt.Figure:
    """Generate horizontal bar chart of feature importance."""
    top = importance_df.head(top_n).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, int(top_n * 0.4))))
    bars = ax.barh(top["feature"], top["mean_abs_shap"], color="#1f77b4", edgecolor="none")
    ax.set_xlabel("Mean |SHAP Value| (Average impact on model output magnitude)")
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (top["mean_abs_shap"].max() * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    return fig
