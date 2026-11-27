# UI/UX and State Design

## Visual Direction

The UI uses an Enterprise HR Analytics SaaS aesthetic: dark surfaces, restrained blue/cyan accents, clear section hierarchy, metric cards, bordered forms, tabbed analytical detail, and status-coded monitoring.

Presentation helpers are centralized in `src/ui_theme.py`. This module is intentionally presentation-only and does not contain ML logic.

## Page Hierarchy

1. Hero and responsible-use banner.
2. Dataset overview.
3. Workforce analytics.
4. Model comparison and optimization.
5. Evaluation and SHAP.
6. Monitoring and data drift.
7. Data quality and responsible-use details.

The sidebar provides workspace navigation, dataset upload, and workforce filters.

## Component Rules

- Use metric cards for short numeric summaries.
- Use tabs to separate long analytical views.
- Use clear success, warning, and critical status banners.
- Use Plotly for interactive analytical charts and Matplotlib for existing model-evaluation and SHAP outputs.
- Keep downloads aggregate-only and labelled explicitly.
- Avoid individual-risk ranking or adverse-decision presentation.

## State Management

Baseline and optimized results remain in Streamlit session state under namespaced `m25_*` keys. Monitoring state uses `m26_*` keys and dataset/model/threshold fingerprints. Widget reruns read state and do not automatically retrain or re-run monitoring.

## API and Data Contracts

The employee CSV contract requires:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `quit`
- `promotion_last_5years`
- `department`
- `salary`

Monitoring batches use the nine feature columns and may optionally include binary `quit` labels.

## Responsive Behavior

The layout uses Streamlit columns and a max-width container. Narrow screens rely on Streamlit's responsive column stacking and compact tables/charts.
