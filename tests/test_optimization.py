from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.attrition import (
    OPTIMIZED_MODEL_FILENAME,
    OPTIMIZED_REPORT_FILENAME,
    compare_baseline_and_optimized,
    load_dataset,
    load_model,
    load_optimized_model,
    optimize_models,
    save_optimized_model,
    split_dataset,
    train_and_compare,
)
from src.explainability import explain_single_prediction

DATA_PATH = Path(__file__).parents[1] / "data" / "employee_data.csv"


@pytest.fixture(scope="module")
def small_dataset():
    data = load_dataset(DATA_PATH)
    # Stratified small subset for fast, deterministic HPO tests
    pos = data[data["quit"] == 1].head(60)
    neg = data[data["quit"] == 0].head(140)
    return data.loc[pos.index.union(neg.index)].copy()


def test_shared_deterministic_split_and_no_overlap(small_dataset):
    X_train, X_test, y_train, y_test, _ = split_dataset(small_dataset, random_state=42)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    with pytest.raises(ValueError, match="n_jobs must be 1"):
        optimize_models(small_dataset, random_state=42, n_jobs=2)

    base_results, _, _ = train_and_compare(small_dataset, random_state=42)
    opt_results, _, _, failed = optimize_models(
        small_dataset, random_state=42, n_iter=2, n_jobs=1
    )
    assert not failed
    assert set(opt_results.keys()) == {"Logistic Regression", "Random Forest"}
    assert base_results["Random Forest"].pipeline.named_steps["model"].n_jobs == -1
    assert opt_results["Random Forest"].pipeline.named_steps["model"].n_jobs == 1

    for name in opt_results:
        assert base_results[name].y_test.index.equals(opt_results[name].y_test.index)
        assert base_results[name].X_test.equals(opt_results[name].X_test)
        assert base_results[name].train_time_sec >= 0.0
        assert base_results[name].predict_time_sec >= 0.0


def test_optimization_reproducibility_with_fixed_seed(small_dataset):
    res1, comp1, _, _ = optimize_models(small_dataset, random_state=42, n_iter=2, n_jobs=1)
    res2, comp2, _, _ = optimize_models(small_dataset, random_state=42, n_iter=2, n_jobs=1)

    for name in res1:
        assert res1[name].best_params == res2[name].best_params
        assert pytest.approx(res1[name].best_cv_score, abs=1e-12) == res2[name].best_cv_score
        np.testing.assert_allclose(res1[name].probabilities, res2[name].probabilities, atol=1e-12)
    assert comp1["model"].tolist() == comp2["model"].tolist()


def test_graceful_handling_of_failed_candidate_model(small_dataset):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        "Invalid LR": LogisticRegression(max_iter=500),
    }
    spaces = {
        "Logistic Regression": {"model__C": [0.1, 1.0]},
        "Invalid LR": {"model__C": [-5.0]},  # Invalid negative C fails all folds
    }
    opt_results, comp, _, failed = optimize_models(
        small_dataset,
        random_state=42,
        n_iter=2,
        n_jobs=1,
        candidate_models=models,
        search_spaces=spaces,
    )
    assert "Logistic Regression" in opt_results
    assert "Invalid LR" not in opt_results
    assert "Invalid LR" in failed
    assert len(comp) == 1


def test_baseline_vs_optimized_comparison_deltas(small_dataset):
    base_results, _, _ = train_and_compare(small_dataset, random_state=42)
    opt_results, _, _, _ = optimize_models(small_dataset, random_state=42, n_iter=2, n_jobs=1)
    summary = compare_baseline_and_optimized(base_results, opt_results)

    assert len(summary) == 2
    for _, row in summary.iterrows():
        expected_abs = row["optimized_f1"] - row["baseline_f1"]
        expected_rel = (expected_abs / row["baseline_f1"] * 100.0) if row["baseline_f1"] > 0 else 0.0
        assert pytest.approx(expected_abs, abs=1e-12) == row["f1_abs_delta"]
        assert pytest.approx(expected_rel, abs=1e-9) == row["f1_rel_pct"]
        assert row["optimization_time_sec"] >= 0.0


def test_save_and_load_optimized_model_isolated_from_baseline(small_dataset, tmp_path):
    opt_results, comp, cleaning_report, _ = optimize_models(
        small_dataset, random_state=42, n_iter=2, n_jobs=1
    )
    best_name = comp.iloc[0]["model"]
    best_opt = opt_results[best_name]

    model_path, report_path = save_optimized_model(best_opt, cleaning_report, artifact_dir=tmp_path)
    assert model_path.name == OPTIMIZED_MODEL_FILENAME
    assert report_path.name == OPTIMIZED_REPORT_FILENAME
    assert not (tmp_path / "best_model.joblib").exists()
    assert not (tmp_path / "training_report.json").exists()

    loaded_pipe, metadata = load_optimized_model(artifact_dir=tmp_path)
    p_orig = best_opt.pipeline.predict_proba(best_opt.X_test.head(10))[:, 1]
    p_load = loaded_pipe.predict_proba(best_opt.X_test.head(10))[:, 1]
    np.testing.assert_allclose(p_orig, p_load, atol=1e-12)
    assert metadata["selected_model"] == best_name
    assert metadata["best_params"] == best_opt.best_params
    assert pytest.approx(metadata["best_cv_f1"], abs=1e-12) == best_opt.best_cv_score


def test_missing_and_corrupted_artifact_handling(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.joblib")

    with pytest.raises(FileNotFoundError):
        load_optimized_model(artifact_dir=tmp_path)

    corrupt_model = tmp_path / OPTIMIZED_MODEL_FILENAME
    corrupt_report = tmp_path / OPTIMIZED_REPORT_FILENAME
    corrupt_model.write_bytes(b"not a valid joblib file")
    corrupt_report.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Corrupted optimized model artifact"):
        load_optimized_model(artifact_dir=tmp_path)


def test_optimized_models_compatible_with_shap(small_dataset):
    opt_results, _, _, _ = optimize_models(small_dataset, random_state=42, n_iter=2, n_jobs=1)
    for name, opt_res in opt_results.items():
        local = explain_single_prediction(
            opt_res.pipeline,
            opt_res.X_test.iloc[[0]],
            background_df=opt_res.X_test,
        )
        diff = abs(local["reconstructed_probability"] - local["attrition_probability"])
        assert diff < 1e-6, f"SHAP reconstruction failed on optimized {name}"
