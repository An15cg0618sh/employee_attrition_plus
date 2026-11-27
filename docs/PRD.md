# Product Requirements Document

## Product

Employee Attrition Prediction & Responsible Workforce Analytics with SHAP Explainable AI is an educational workforce analytics application for exploring historical attrition patterns, comparing classifiers, and explaining model behaviour.

## Problem

Organizations need to understand aggregate workforce patterns and model behaviour without presenting predictions as individual decisions or causal explanations. The product provides historical analysis, explainability, optimization, and drift monitoring while retaining human review and privacy safeguards.

## Goals

- Validate employee CSV schema and deduplicate employee records.
- Train and compare Random Forest and Logistic Regression baselines.
- Provide bounded, reproducible hyperparameter optimization.
- Explain global and individual model behaviour with SHAP.
- Provide workforce aggregate analytics with reliable-sample safeguards.
- Monitor numerical drift, categorical drift, data quality, predictions, and optional ground truth.
- Persist monitoring and optimized artifacts separately from baseline artifacts.
- Keep exports aggregate-only and prohibit individual employment decisions.

## User Flows

1. Upload or select the default employee CSV.
2. Review dataset overview, target distribution, and baseline model metrics.
3. Filter aggregate workforce data by department, salary, tenure, work accident, promotion, and minimum group size.
4. Select a model and threshold to evaluate ROC-AUC, PR-AUC, and threshold-dependent metrics.
5. Optionally run bounded optimization and select baseline or optimized model versions.
6. Inspect permutation importance and global/local SHAP explanations.
7. Upload a monitoring batch, run drift monitoring, and review/export aggregate reports.
8. Review responsible-use and data-quality notices.

## Shipped Milestones

- M2.1: Core data pipeline and baseline explainability.
- M2.2: Analytics, workforce filtering, and expanded testing.
- M2.3: Workforce consistency and dashboard integrations.
- M2.4: Production-safe exports and final baseline behavior.
- M2.5: Hyperparameter optimization, persistence, comparison, and version selection.
- M2.6: Monitoring, drift detection, status engine, and aggregate reporting.
- M2.7: Deployment configuration, environment validation, health, and documentation.

## Non-goals

- Employment, promotion, hiring, termination, or penalty recommendations.
- Individual employee ranking or automated adverse action.
- Causal inference or fairness certification.
- Replacement of HR/legal/privacy professional review.
- Real-time employee surveillance or externally persistent monitoring service.
