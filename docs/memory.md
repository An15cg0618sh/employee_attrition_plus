# Engineering Memory

## Key Decisions

- Exact duplicate removal occurs before splitting to prevent identical records crossing train/test partitions.
- Preprocessing remains inside sklearn pipelines and is fitted only on training/CV folds.
- The baseline and optimized workflows share the same deterministic stratified split.
- Hyperparameter selection uses bounded RandomizedSearchCV on training folds only.
- Optimized artifacts use separate filenames and never overwrite baseline artifacts.
- M2.6 monitoring is aggregate-only and can operate without ground-truth labels.
- Monitoring results are deduplicated by reference, monitoring data, model identity, and threshold signature.
- Drift is descriptive and never presented as bias, causation, or guaranteed model failure.

## Lessons Learned

- Streamlit cache keys must include model identity and version to prevent stale SHAP explanations.
- Threshold metrics require result probabilities and y-test values, not retraining.
- Missing and single-class ground truth must be handled explicitly.
- PSI, KS, and Chi-square all have assumptions and limitations.
- A browser health check alone does not verify native file upload or a monitoring run.
- Local virtual environments and caches should not be committed.
- Milestone QA reports contain historical context but can become stale; canonical docs should reflect current code.

## Known Edge Cases

- Very small batches may produce unstable KS/Chi-square conclusions.
- Constant numeric features require safe PSI handling.
- Unseen categories are reported descriptively.
- Invalid or partial `quit` labels do not produce delayed performance metrics.
- Hosted ephemeral storage may lose monitoring history.
- Uploaded data is runtime-only and must not be treated as durable storage.

## Reference Notebook

- `notebooks/original_reference.ipynb` is retained as historical exploratory source material.
- It is not imported by application, training, or test code.
- Executable analytics behavior is implemented and tested in `src/analytics.py` and the other `src/` modules.
