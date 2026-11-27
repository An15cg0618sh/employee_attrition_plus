# Milestone 2.5 – Checkpoint 2 Reconciliation

**Date:** 2026-11-27  
**Decision:** **BLOCKED – historical test identity is not fully reconstructible**  
**Checkpoint 2 implementation sign-off:** Not granted  
**Checkpoint 3 dashboard integration:** Not started

## 1. Executive summary

The current pre-optimization suite has 54 collected tests and all 54 pass. The seven optimization tests also pass, and the complete suite has 61 passing tests. However, the Milestone 2.4 report records only an aggregate count of 53; it does not provide the complete historical test-name/assertion inventory. This workspace is not a Git repository, so the historical source diff cannot be recovered here. It is therefore not possible to certify that all 53 original test bodies and assertions are preserved.

The nested Random Forest parallelism issue was corrected without changing baseline training: optimization now clones candidate estimators and constrains nested `n_jobs` parameters to 1, while baseline Random Forest configuration remains unchanged. The current optimizer and tests are reviewed below, but these results do not resolve the historical test-identity blocker or constitute implementation sign-off.

## 2. Git status and history

Before editing, these commands were attempted against the workspace:

- `git rev-parse --show-toplevel`
- `git status --short --branch`
- `git log -10 --oneline --decorate`

Each returned `fatal: not a git repository`. No Git status, commit history, or historical file diff is available. The project root has no usable Git metadata, so test-body equality cannot be verified through Git.

The supplied remote, [employee_attrition_plus](https://github.com/An15cg0618sh/employee_attrition_plus), was also inspected. `git ls-remote --heads --tags` returned no refs, GitHub branch listing returned an empty list, and GitHub commit queries returned `409 Git Repository is empty`. It contains no history to compare.

The local session index records an `apply_patch` event for `tests/test_analytics.py` at `2026-11-27T13:39:45.727Z`. The earlier Milestone 2.5 QA report is timestamped `2026-11-27T13:23:58.178Z`, but the session index does not retain the patch hunk. This is evidence of a later file edit, not proof whether a test was added, renamed, or modified.

## 3. Historical test inventory reconciliation

The Milestone 2.4 QA report states:

- 49 tests before its focused additions.
- 53 tests afterward.
- Four additions described by behavior: Random Forest save/load probability round trip, combined-filter aggregate consistency, empty-export headers, and aggregate CSV payload round trip.

It does not list all 53 test names or their assertion bodies. A subsequent Milestone 2.5 audit report records 53 collected tests and names `test_tenure_grouping` as a failing baseline test, but also does not provide a complete collection listing. The prior implementation plan's per-module breakdown (11/18/13/11) does not match the current module counts and is not a historical test source.

| Historical name or description | Current test name | Module | Status | Evidence |
|---|---|---|---|---|
| Random Forest save/load predict-probability round trip | `test_saved_model_round_trip` | `tests/test_pipeline.py` | Preserved by behavior/name match; exact assertion diff unavailable | Milestone 2.4 QA report lists this addition; current test saves, reloads, and compares `predict_proba`. |
| Combined-filter aggregate export consistency | `test_filtered_aggregate_export_reflects_combined_filters` | `tests/test_workforce_analytics.py` | Preserved by behavior/name match; exact assertion diff unavailable | Milestone 2.4 QA report lists this addition; current test checks filtered counts, required columns, and absence of employee-level fields. |
| Empty filtered export required-header behavior | `test_empty_filtered_dataset_exports_required_headers` | `tests/test_workforce_analytics.py` | Preserved by behavior/name match; exact assertion diff unavailable | Milestone 2.4 QA report lists this addition; current test checks empty output and required headers. |
| Aggregate CSV payload round trip | `test_aggregate_export_csv_payload_round_trip` | `tests/test_workforce_analytics.py` | Preserved by behavior/name match; exact assertion diff unavailable | Milestone 2.4 QA report lists this addition; current test serializes and reloads CSV bytes and checks aggregates. |
| `test_tenure_grouping` | `test_tenure_grouping` | `tests/test_analytics.py` | Present under the same name; exact assertion diff unavailable | The later QA audit identifies this name in its 53-item collection and records its failure. Current test still asserts the expected `"Unknown"` labels. |
| Not enumerated in the historical QA reports | `test_tenure_grouping_missing_and_boundary_values` | `tests/test_analytics.py` | **Unverified**; possible post-audit addition or modification | Present in the current 54-test suite. A later file-level patch event exists, but its hunk is unavailable; the test's historical identity cannot be established. |
| Remaining historical identities (up to 48 if the five mappings above are distinct) | Unmapped current cases; exact count cannot be verified | All four pre-optimization modules | **Cannot determine** | Aggregate historical reports and absent Git history do not provide test names or old assertion bodies; the candidate boundary test's provenance is also unresolved. |

No historical test can be confirmed as deleted, renamed, or weakened from the available evidence. That is not evidence that none were deleted, renamed, or weakened. The four documented Milestone 2.4 additions and `test_tenure_grouping` are identifiable by report description/name, but their exact historical source bodies are not available for a diff.

### Current collected test inventory

Current per-module collection is:

| Module | Collected |
|---|---:|
| `tests/test_analytics.py` | 12 |
| `tests/test_explainability.py` | 18 |
| `tests/test_pipeline.py` | 11 |
| `tests/test_workforce_analytics.py` | 13 |
| **Pre-optimization modules** | **54** |
| `tests/test_optimization.py` | 7 |
| **Current total** | **61** |

The current named test functions were inspected:

- **Analytics (12):** `test_tenure_grouping`, `test_tenure_grouping_missing_and_boundary_values`, `test_department_attrition_calculations`, `test_salary_attrition_calculations`, `test_tenure_attrition_calculations`, `test_small_sample_safeguards`, `test_filter_workforce_data`, `test_empty_dataset_handling`, `test_missing_values_handling`, `test_satisfaction_performance_summary`, `test_aggregate_export_and_csv_generation`, `test_plotly_charts_generate_valid_figures`.
- **Explainability (18 collected cases):** `test_clean_feature_names`, `test_build_explainer_supported_models`, `test_explainer_unsupported_model_raises`, `test_pipeline_validation_errors`, `test_compute_global_shap_both_models`, `test_shap_feature_importance_and_impact`, `test_single_prediction_explanation`, `test_positive_class_reconstruction_uses_all_features` (2 parameter cases), `test_categorical_names_align_with_reconstruction_values` (2), `test_global_reconstruction_matches_native_model_output` (2), `test_invalid_input_handling`, `test_plots_generate_figures`, `test_reconstruction_property_random_forest`, `test_reconstruction_property_logistic_regression`, `test_global_batch_reconstruction_property`.
- **Pipeline (11):** `test_dataset_loads_and_target_is_binary`, `test_duplicate_cleaning_reports_removed_rows`, `test_saved_model_round_trip`, `test_app_uses_streamlit_compatible_shap_notation`, `test_training_pipeline_returns_both_models`, `test_shap_explainability_for_supported_models`, `test_shap_feature_names_align_with_transformed_data`, `test_shap_rejects_empty_and_invalid_input`, `test_validate_dataset_frame_rejects_invalid_schemas`, `test_training_rejects_tiny_dataset`, `test_threshold_metrics_and_unseen_categories`.
- **Workforce analytics (13):** `test_department_attrition_calculation`, `test_salary_attrition_calculation`, `test_tenure_grouping_handles_ranges_and_missing_values`, `test_deduplicated_source_of_truth_and_no_filter_counts`, `test_filters_update_filtered_dataset`, `test_empty_filtered_dataset`, `test_missing_categories_and_values_are_grouped_as_unknown`, `test_small_group_safeguard_prevents_unreliable_ranking`, `test_group_attrition_rejects_missing_target`, `test_aggregate_export_has_no_employee_identifiers`, `test_filtered_aggregate_export_reflects_combined_filters`, `test_empty_filtered_dataset_exports_required_headers`, `test_aggregate_export_csv_payload_round_trip`.
- **Optimization (7):** `test_shared_deterministic_split_and_no_overlap`, `test_optimization_reproducibility_with_fixed_seed`, `test_graceful_handling_of_failed_candidate_model`, `test_baseline_vs_optimized_comparison_deltas`, `test_save_and_load_optimized_model_isolated_from_baseline`, `test_missing_and_corrupted_artifact_handling`, `test_optimized_models_compatible_with_shap`.

The historical Milestone 2.4 count is **53**. The current non-optimization modules contain **54 passing tests**, including one test whose historical identity is unresolved. The optimization module contains **7 tests**, and current total collection is **61**. The count arithmetic does not establish historical assertion preservation.

## 4. Optimization parallelism verification

Before the correction, the default search wrapper used `RandomizedSearchCV(n_jobs=1)`, but the shared Random Forest candidate was configured with `n_jobs=-1`. That allowed estimator-level parallelism inside otherwise serial CV.

The correction in `src/attrition.py`:

- Preserves `get_candidate_models()` and baseline Random Forest `n_jobs=-1`.
- Clones candidate estimators for optimization, so optimization does not mutate the baseline candidate.
- Sets every non-`None`, non-1 nested `n_jobs` estimator parameter to 1 in the optimization clone.
- Fixes `RandomizedSearchCV(n_jobs=1)` and rejects an `optimize_models(n_jobs=...)` value other than 1.
- Leaves Logistic Regression's `n_jobs=None` unchanged; setting it to 1 generated 49 Scikit-learn FutureWarnings on an initial test run, so that warning-producing behavior was removed.

The existing shared-split optimization test now asserts baseline Random Forest remains `n_jobs=-1`, optimized Random Forest is `n_jobs=1`, and an outer request for `n_jobs=2` is rejected. Final optimization tests pass without warnings.

Boundedness and reproducibility observed in source:

- `optimize_models()` default `n_iter=6`, `random_state=42`, `cv_splits=3`.
- `RandomizedSearchCV` uses `StratifiedKFold(shuffle=True, random_state=42)` by default.
- Candidate spaces have 10 Logistic Regression combinations and 72 Random Forest combinations; `effective_n_iter` is capped by the number of combinations. The default therefore searches at most 6 candidates per model.
- Search fitting receives only `X_train`/`y_train`; held-out evaluation occurs after `search.fit`.

## 5. Existing optimization implementation audit

| Area | Finding |
|---|---|
| Shared split | `split_dataset()` validates, deterministically removes exact duplicates, then performs one stratified split with default seed 42. Both baseline and optimization call it with the same input/defaults. The optimization test compares `X_test` and `y_test` indices across both paths. |
| Preprocessing and leakage | Numeric imputation/scaling and categorical imputation/one-hot encoding are pipeline steps. CV/search fit the pipeline on training folds from `X_train`; `X_test` is used for evaluation after fitting. `drop_duplicates()` is deterministic and learns no population statistics; before-split deduplication prevents exact duplicate rows crossing the split. |
| Baseline training | `train_and_compare()` retains the baseline result/comparison flow, cross-validation, timing fields, and baseline estimator settings. Its broad `except Exception: continue` still suppresses per-model failure details; Ruff reports this as `BLE001`/`S112`. |
| Optimization | `optimize_models()` performs bounded randomized search, records CV score/parameters/timing, evaluates on the held-out split, and returns per-model failures. The parallelism correction is applied only to optimization clones. |
| Comparison | `compare_baseline_and_optimized()` calculates metric deltas and checks `y_test` indices. It does not itself compare `X_test` frame values; the shared deterministic split and existing optimization test currently verify those values in the standard path. A direct `X_test.equals()` guard would make the comparison helper more defensive. |
| Persistence | `save_optimized_model()` and `load_optimized_model()` use `optimized_best_model.joblib` and `optimized_training_report.json`; tests use `tmp_path`. Baseline persistence functions remain separate. |
| SHAP | Existing explainability accepts sklearn pipelines; the optimization suite tests local SHAP reconstruction for both optimized candidates. |
| Dashboard | `app.py` has no optimization/model-comparison integration (`rg` found no optimization or search references). Optimization controls, user-triggered execution, progress, charts, and best-parameter summary are not implemented. This is pending Checkpoint 3, not part of this reconciliation. |
| Artifacts | `artifacts/` contains only `best_model.joblib` and `training_report.json`; no optimized artifact was created. Before/after SHA-256 hashes match: model `90FB9226F8CECC95C11781ECA7119BC2405E38EB1FBA7312A36C81FAF11E6816`, report `E07230A78B26C44099A4D17EF19C9B9B4A9D82E42C92E1CC6CD72F7AA9B720FF`. |

## 6. Separate validation results

All commands used `.venv-new2\Scripts\python.exe`.

| Scope | Command | Result |
|---|---|---|
| Current pre-optimization modules | `python -m pytest -q tests/test_analytics.py tests/test_explainability.py tests/test_pipeline.py tests/test_workforce_analytics.py` | **54 passed** in 64.26s; 0 failures, skips, or warnings. |
| Optimization tests | `python -m pytest -q tests/test_optimization.py` | **7 passed** in 39.23s; 0 failures, skips, or warnings. |
| Full suite | `python -m pytest -q` | **61 passed** in 99.59s; 0 failures, skips, or warnings. |
| Collection | `python -m pytest --collect-only -q tests` | **61 tests collected**. |
| Compilation | `python -m py_compile src/attrition.py tests/test_optimization.py app.py` | Exit code 0. |

An earlier optimization-test run, before the `n_jobs=None` adjustment, passed 7 tests but emitted 49 Scikit-learn FutureWarnings. The final rerun after the adjustment passed without warnings.

### Current vs historical counts

| Count category | Result |
|---|---:|
| Historical Milestone 2.4 count documented by QA report | 53 |
| Historical tests individually corroborated by report name/description | 5 (4 documented Milestone 2.4 additions plus `test_tenure_grouping` in the later 53-item audit; assertion-body equality unverified) |
| Current pre-optimization cases passing | 54 |
| New optimization test cases | 7 |
| Current total cases passing | 61 |
| Historical assertion preservation established | **No; full source baseline unavailable** |

## 7. Ruff findings

Command: `.venv-new2\Scripts\python.exe -m ruff check src/ app.py tests/`

**Exit code 1; 48 findings.** Findings by rule:

| Rule | Count |
|---|---:|
| `BLE001` | 5 |
| `C408` | 26 |
| `I001` | 6 |
| `PERF102` | 1 |
| `RUF046` | 1 |
| `RUF059` | 7 |
| `S112` | 1 |
| `TRY004` | 1 |
| **Total** | **48** |

The same count and rule distribution were observed before and after the optimization parallelism correction. No Ruff rule was disabled or weakened. Existing findings include import ordering, blind exception handling, unnecessary `dict()` calls, and unused unpacked values. `src/attrition.py` retains `BLE001`/`S112` findings in baseline training and a broad exception handler in optimization.

## 8. Files modified

- `src/attrition.py` — isolate optimization estimator parallelism from baseline settings and enforce search-level `n_jobs=1`.
- `tests/test_optimization.py` — assert baseline/optimized Random Forest worker settings and reject `n_jobs=2`.
- `MILESTONE_2_5_IMPLEMENTATION_PLAN.md` — record the parallelism correction and retain the historical test inventory as the blocking condition.
- `MILESTONE_2_5_CHECKPOINT_2_RECONCILIATION.md` — this evidence report.

No original pre-optimization test file was edited in this task. No dependency or baseline artifact was changed.

## 9. Remaining blocker and requested decision

Checkpoint 2 cannot be signed off because historical identity and assertion preservation for all 53 original tests cannot be established from the available report and workspace. The current suite passes, but a passing current count is not proof of historical preservation.

The supplied repository did not resolve the missing history. Please provide:

1. An archived Milestone 2.4 test snapshot with all 53 test names and assertion bodies, or
2. A non-empty Git repository/commit containing the exact Milestone 2.4 test files.

Until that evidence is available, no claim is made that all 53 original tests are preserved. Checkpoint 3 dashboard integration remains blocked.
