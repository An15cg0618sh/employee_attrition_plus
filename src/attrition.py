"""Reusable employee attrition modelling utilities.

This project is an educational workforce-analytics prototype. It must not be
used to make employment, promotion, termination, or other consequential
decisions about individuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "quit"
EXPECTED_COLUMNS = {
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company", "Work_accident",
    "quit", "promotion_last_5years", "department", "salary",
}
OPTIMIZED_MODEL_FILENAME = "optimized_best_model.joblib"
OPTIMIZED_REPORT_FILENAME = "optimized_training_report.json"


@dataclass
class TrainingResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    cv_metrics: dict[str, float]
    y_test: pd.Series
    probabilities: np.ndarray
    predictions: np.ndarray
    X_test: pd.DataFrame
    train_time_sec: float = 0.0
    predict_time_sec: float = 0.0


@dataclass
class OptimizationResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    best_params: dict[str, Any]
    best_cv_score: float
    cv_std: float
    optimization_time_sec: float
    predict_time_sec: float
    n_iter: int
    y_test: pd.Series
    probabilities: np.ndarray
    predictions: np.ndarray
    X_test: pd.DataFrame
    cv_metrics: dict[str, float] = field(default_factory=dict)


def validate_dataset_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame matches the expected attrition schema and target coding."""
    if data.empty:
        raise ValueError("The dataset contains no rows.")
    missing_columns = EXPECTED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if data[TARGET].isna().any():
        raise ValueError("The target column 'quit' contains missing values.")
    observed = set(data[TARGET].dropna().unique())
    if not observed.issubset({0, 1}):
        raise ValueError("The target column 'quit' must contain only 0 and 1.")
    return data


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the supplied CSV without silently changing column names."""
    data = pd.read_csv(path)
    return validate_dataset_frame(data)


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact duplicate records and report missing values.

    Missing feature values are imputed inside the model pipeline to avoid
    learning preprocessing statistics from the held-out test set.
    Deterministic exact-duplicate removal before splitting is safe because it
    uses no learned population statistics and prevents identical rows from
    leaking across train and test partitions.
    """
    before = len(data)
    cleaned = data.drop_duplicates().copy()
    report = {
        "rows_before": int(before),
        "duplicates_removed": int(before - len(cleaned)),
        "rows_after": int(len(cleaned)),
        "missing_feature_cells": int(cleaned.drop(columns=[TARGET]).isna().sum().sum()),
    }
    return cleaned, report


def split_dataset(
    data: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
    """Validate, deduplicate, and produce a deterministic stratified train/test split."""
    validate_dataset_frame(data)
    cleaned, cleaning_report = clean_dataset(data)
    X = cleaned.drop(columns=[TARGET])
    y = cleaned[TARGET].astype(int)

    if y.nunique() < 2 or int(y.value_counts().min()) < 4:
        raise ValueError(
            "Dataset must contain both classes (0 and 1) with at least 4 deduplicated "
            "records per class for stratified splitting and 3-fold cross-validation."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, cleaning_report


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def get_candidate_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
    }


def get_hyperparameter_search_spaces() -> dict[str, dict[str, list[Any]]]:
    """Return bounded hyperparameter search spaces for pipeline step 'model'."""
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 16, 24],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
    }


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray,
                      threshold: float = 0.5) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    return metrics, predictions


def _has_valid_metrics(metrics: dict[str, float]) -> bool:
    return all(np.isfinite(val) for val in metrics.values())


def train_and_compare(
    data: pd.DataFrame,
    random_state: int = 42,
    candidate_models: dict[str, Any] | None = None,
) -> tuple[dict[str, TrainingResult], pd.DataFrame, dict[str, int]]:
    X_train, X_test, y_train, y_test, cleaning_report = split_dataset(data, random_state=random_state)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    results: dict[str, TrainingResult] = {}
    rows = []
    models = candidate_models if candidate_models is not None else get_candidate_models()

    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(X_train)),
                ("model", model),
            ])
            t0 = time.perf_counter()
            cv_result = cross_validate(
                pipeline, X_train, y_train, cv=cv,
                scoring={"f1": "f1", "roc_auc": "roc_auc"},
                n_jobs=1, return_train_score=False,
                error_score=np.nan,
            )
            pipeline.fit(X_train, y_train)
            train_time_sec = float(time.perf_counter() - t0)

            t1 = time.perf_counter()
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predict_time_sec = float(time.perf_counter() - t1)

            metrics, predictions = calculate_metrics(y_test, probabilities, 0.5)
            cv_metrics = {
                "cv_f1_mean": float(np.nanmean(cv_result["test_f1"])),
                "cv_f1_std": float(np.nanstd(cv_result["test_f1"])),
                "cv_roc_auc_mean": float(np.nanmean(cv_result["test_roc_auc"])),
            }
            if not _has_valid_metrics(metrics) or not _has_valid_metrics(cv_metrics):
                continue

            results[name] = TrainingResult(
                name=name, pipeline=pipeline, metrics=metrics, cv_metrics=cv_metrics,
                y_test=y_test, probabilities=probabilities, predictions=predictions,
                X_test=X_test, train_time_sec=train_time_sec, predict_time_sec=predict_time_sec,
            )
            rows.append({
                "model": name,
                **metrics,
                **cv_metrics,
                "train_time_sec": train_time_sec,
                "predict_time_sec": predict_time_sec,
            })
        except Exception:
            continue

    if not results:
        raise RuntimeError("All candidate models failed during baseline training and evaluation.")

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["cv_f1_mean", "model"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return results, comparison, cleaning_report


def optimize_models(
    data: pd.DataFrame,
    random_state: int = 42,
    n_iter: int = 6,
    n_jobs: int = 1,
    cv_splits: int = 3,
    candidate_models: dict[str, Any] | None = None,
    search_spaces: dict[str, dict[str, list[Any]]] | None = None,
) -> tuple[dict[str, OptimizationResult], pd.DataFrame, dict[str, int], dict[str, str]]:
    """Run bounded RandomizedSearchCV strictly on training folds and evaluate once on X_test."""
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if n_jobs != 1:
        raise ValueError("n_jobs must be 1 for bounded optimization.")

    X_train, X_test, y_train, y_test, cleaning_report = split_dataset(data, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    models = candidate_models if candidate_models is not None else get_candidate_models()
    spaces = search_spaces if search_spaces is not None else get_hyperparameter_search_spaces()

    opt_results: dict[str, OptimizationResult] = {}
    failed_models: dict[str, str] = {}
    rows = []

    for name, model in models.items():
        if name not in spaces or not spaces[name]:
            failed_models[name] = "No hyperparameter search space configured."
            continue

        try:
            optimization_model = clone(model)
            parallel_params = {
                key: 1
                for key, value in optimization_model.get_params(deep=True).items()
                if (key == "n_jobs" or key.endswith("__n_jobs"))
                and value is not None
                and value != 1
            }
            if parallel_params:
                optimization_model.set_params(**parallel_params)

            pipeline = Pipeline([
                ("preprocess", build_preprocessor(X_train)),
                ("model", optimization_model),
            ])
            param_distributions = spaces[name]
            total_combos = int(np.prod([len(v) for v in param_distributions.values()]))
            effective_n_iter = min(int(n_iter), max(1, total_combos))

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                n_iter=effective_n_iter,
                scoring={"f1": "f1", "roc_auc": "roc_auc"},
                cv=cv,
                random_state=random_state,
                n_jobs=1,
                refit="f1",
                error_score=np.nan,
            )

            t0 = time.perf_counter()
            search.fit(X_train, y_train)
            opt_time_sec = float(time.perf_counter() - t0)

            best_pipeline = search.best_estimator_
            best_idx = int(search.best_index_)
            cv_std = float(search.cv_results_["std_test_f1"][best_idx])
            best_cv_score = float(search.best_score_)
            cv_roc_auc_mean = float(search.cv_results_["mean_test_roc_auc"][best_idx])

            t1 = time.perf_counter()
            probabilities = best_pipeline.predict_proba(X_test)[:, 1]
            predict_time_sec = float(time.perf_counter() - t1)

            metrics, predictions = calculate_metrics(y_test, probabilities, 0.5)
            if (
                not _has_valid_metrics(metrics)
                or not np.isfinite(best_cv_score)
                or not np.isfinite(cv_std)
                or not np.isfinite(cv_roc_auc_mean)
            ):
                failed_models[name] = "Optimization produced non-finite metrics."
                continue

            clean_params = {
                k.replace("model__", ""): v for k, v in search.best_params_.items()
            }
            cv_metrics = {
                "cv_f1_mean": best_cv_score,
                "cv_f1_std": cv_std,
                "cv_roc_auc_mean": cv_roc_auc_mean,
            }

            opt_results[name] = OptimizationResult(
                name=name,
                pipeline=best_pipeline,
                metrics=metrics,
                best_params=clean_params,
                best_cv_score=best_cv_score,
                cv_std=cv_std,
                optimization_time_sec=opt_time_sec,
                predict_time_sec=predict_time_sec,
                n_iter=effective_n_iter,
                y_test=y_test,
                probabilities=probabilities,
                predictions=predictions,
                X_test=X_test,
                cv_metrics=cv_metrics,
            )
            rows.append({
                "model": name,
                **metrics,
                "best_cv_f1": best_cv_score,
                "cv_f1_std": cv_std,
                "cv_roc_auc_mean": cv_roc_auc_mean,
                "optimization_time_sec": opt_time_sec,
                "predict_time_sec": predict_time_sec,
                "n_iter": effective_n_iter,
                "best_params": json.dumps(clean_params, sort_keys=True),
            })
        except Exception as exc:
            failed_models[name] = str(exc)

    if not opt_results:
        raise RuntimeError(f"All models failed optimization: {failed_models}")

    opt_comparison = (
        pd.DataFrame(rows)
        .sort_values(["best_cv_f1", "model"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return opt_results, opt_comparison, cleaning_report, failed_models


def compare_baseline_and_optimized(
    baseline_results: dict[str, TrainingResult],
    optimized_results: dict[str, OptimizationResult],
) -> pd.DataFrame:
    """Compare baseline and optimized metrics on the shared held-out test set."""
    rows = []
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    for name, opt_res in optimized_results.items():
        if name not in baseline_results:
            continue
        base_res = baseline_results[name]
        if not base_res.X_test.equals(opt_res.X_test) or not base_res.y_test.equals(opt_res.y_test):
            raise ValueError(f"Test split mismatch between baseline and optimized '{name}'.")

        row: dict[str, Any] = {
            "model": name,
        }
        metric_values = {
            **{
                key: (float(base_res.metrics[key]), float(opt_res.metrics[key]))
                for key in metric_keys
            },
            "cv_f1": (
                float(base_res.cv_metrics.get("cv_f1_mean", np.nan)),
                float(opt_res.cv_metrics.get("cv_f1_mean", opt_res.best_cv_score)),
            ),
            "cv_roc_auc": (
                float(base_res.cv_metrics.get("cv_roc_auc_mean", np.nan)),
                float(opt_res.cv_metrics.get("cv_roc_auc_mean", np.nan)),
            ),
        }
        row["baseline_cv_f1_std"] = float(
            base_res.cv_metrics.get("cv_f1_std", np.nan)
        )
        row["optimized_cv_f1_std"] = float(opt_res.cv_std)
        for key, (base_val, opt_val) in metric_values.items():
            abs_delta = opt_val - base_val
            rel_pct = (
                abs_delta / abs(base_val) * 100.0
                if np.isfinite(base_val) and base_val != 0
                else np.nan
            )
            row[f"baseline_{key}"] = base_val
            row[f"optimized_{key}"] = opt_val
            row[f"{key}_abs_delta"] = abs_delta
            row[f"{key}_rel_pct"] = rel_pct

        row["baseline_train_time_sec"] = float(base_res.train_time_sec)
        row["optimized_optimization_time_sec"] = float(opt_res.optimization_time_sec)
        row["baseline_predict_time_sec"] = float(base_res.predict_time_sec)
        row["optimized_predict_time_sec"] = float(opt_res.predict_time_sec)
        row["optimization_time_sec"] = float(opt_res.optimization_time_sec)
        row["best_params"] = json.dumps(opt_res.best_params, sort_keys=True)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["optimized_cv_f1", "model"], ascending=[False, True])
        .reset_index(drop=True)
    )


def threshold_metrics(result: TrainingResult | OptimizationResult, threshold: float) -> tuple[dict[str, float], np.ndarray]:
    return calculate_metrics(result.y_test, result.probabilities, threshold)


def get_permutation_importance(result: TrainingResult | OptimizationResult, repeats: int = 3) -> pd.DataFrame:
    """Feature importance on original input columns, not encoded dummy columns."""
    importance = permutation_importance(
        result.pipeline, result.X_test, result.y_test,
        n_repeats=repeats, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    return (
        pd.DataFrame({
            "feature": result.X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_model(result: TrainingResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, path)


def load_model(path: str | Path) -> Pipeline:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Model artifact not found: {target}")
    try:
        return joblib.load(target)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact '{target}': {exc}") from exc


def save_optimized_model(
    result: OptimizationResult,
    cleaning_report: dict[str, int],
    artifact_dir: str | Path = Path("artifacts"),
) -> tuple[Path, Path]:
    """Persist the optimized model and metadata separately from baseline artifacts."""
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / OPTIMIZED_MODEL_FILENAME
    report_path = target_dir / OPTIMIZED_REPORT_FILENAME

    joblib.dump(result.pipeline, model_path)
    metadata = {
        "selected_model": result.name,
        "selection_criterion": "best_cv_f1 (StratifiedKFold on X_train)",
        "best_params": result.best_params,
        "best_cv_f1": result.best_cv_score,
        "cv_f1_std": result.cv_std,
        "n_iter": result.n_iter,
        "optimization_time_sec": result.optimization_time_sec,
        "predict_time_sec": result.predict_time_sec,
        "metrics": result.metrics,
        "cv_metrics": result.cv_metrics,
        "data_quality": cleaning_report,
    }
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, report_path


def load_optimized_model(
    artifact_dir: str | Path = Path("artifacts"),
) -> tuple[Pipeline, dict[str, Any]]:
    """Load the persisted optimized model and its JSON metadata."""
    target_dir = Path(artifact_dir)
    model_path = target_dir / OPTIMIZED_MODEL_FILENAME
    report_path = target_dir / OPTIMIZED_REPORT_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"Optimized model artifact not found: {model_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Optimized report metadata not found: {report_path}")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Corrupted optimized model artifact '{model_path}': {exc}") from exc

    try:
        metadata = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Corrupted optimized metadata '{report_path}': {exc}") from exc

    return pipeline, metadata
