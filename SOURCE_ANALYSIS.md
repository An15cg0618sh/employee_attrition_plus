# Source notebook analysis

The uploaded notebook already demonstrates exploratory plots, one-hot encoding,
a stratified 80/20 split, interactive Decision Tree and Random Forest training,
feature importance, and ROC-AUC visualizations.

## Issues addressed in this adaptation

1. The CSV has missing values in `promotion_last_5years`, `department`, and `salary`.
   The original notebook does not explicitly impute these values before model fitting.
2. The dataset contains exact duplicate rows. This adaptation reports and removes
   exact duplicates before splitting.
3. The target is imbalanced (2,000 positive and 9,582 negative rows in the supplied
   file), so accuracy alone is insufficient.
4. The original notebook primarily prints accuracy and plots ROC-AUC. This version
   adds precision, recall, F1, PR-AUC, confusion matrix, and cross-validation.
5. The original notebook uses older/deprecated or version-sensitive APIs and
   dependencies, including `pandas_profiling`, Yellowbrick, Graphviz, and
   `max_features='auto'`. The new app uses current scikit-learn pipeline patterns.
6. The original tree visualisation contains the class label typo `guit` instead of
   `quit`.
7. The original notebook's interactive Random Forest widget passes bootstrap
   values as strings (`'True'` / `'False'`) rather than booleans.

## Dataset inspection

- 11,582 rows and 10 columns.
- Target: `quit`.
- 2,000 positive target rows and 9,582 negative target rows.
- 3 missing cells in the supplied CSV.
- 9 exact duplicate rows.
- Positive target proportion: approximately 17.3%.

The adapted project is a new implementation around the supplied notebook and CSV;
it is not a line-by-line patch of the original notebook.
