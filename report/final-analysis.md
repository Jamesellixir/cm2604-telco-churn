# Final evaluation, errors and responsible use

The selected model was frozen before test evaluation. All six frozen configurations and the majority-class baseline were fitted on training data and evaluated on the same 1,409 held-out customers. No test result changed a parameter or the 0.5 decision boundary.

## Selected model errors

Tuned random forest: test F1 0.6316, precision 0.5353, recall 0.7701, accuracy 0.7615, ROC-AUC 0.8428. It missed 86 churn cases and flagged 250 non-churn cases. False negatives could miss retention opportunities; false positives could waste outreach effort or impose unwanted contact. These consequences are plausible use-case risks, not observed business outcomes.

## Overfitting and comparison

- baseline: training F1 0.0000; test F1 0.0000; gap 0.0000.
- knn_default: training F1 0.6804; test F1 0.5476; gap 0.1328.
- knn_tuned: training F1 0.6313; test F1 0.5878; gap 0.0436.
- logistic_regression_default: training F1 0.5999; test F1 0.6040; gap -0.0041.
- logistic_regression_tuned: training F1 0.6331; test F1 0.6176; gap 0.0154.
- random_forest_default: training F1 0.9963; test F1 0.5370; gap 0.4594.
- random_forest_tuned: training F1 0.7372; test F1 0.6316; gap 0.1057.

Training scores are resubstitution scores, not independent validation. Large gaps suggest fitting training-specific patterns. Compare them with the earlier cross-validation summaries; a single held-out split does not establish stability on future populations. Higher test accuracy for another configuration does not override the predeclared F1-based selection. The test set must not become another tuning set.

## Subgroup audit

Rates below concern the selected model only. FNR is the fraction of actual churners missed; FPR is the fraction of non-churners flagged. Counts supply essential context.

| Attribute | Group | Customers | Churners | FNR | FPR |
|---|---|---:|---:|---:|---:|
| gender | Female | 687 | 193 | 0.2487 | 0.2368 |
| gender | Male | 722 | 181 | 0.2099 | 0.2458 |
| SeniorCitizen | 0 | 1187 | 276 | 0.2681 | 0.2141 |
| SeniorCitizen | 1 | 222 | 98 | 0.1224 | 0.4435 |

Differences describe this sample and do not establish discrimination or fairness. Groups overlap across attributes. Smaller groups have less stable rates; no causal inference or statistical significance is claimed. SeniorCitizen is a binary dataset field, not a detailed age measure. An audit of two attributes cannot detect every intersectional or proxy-related harm.

## Development choices and ethical risks

Raw data and customer-level predictions are excluded from the public repository; only aggregate outputs and an explicitly synthetic demonstration are exported. Customer identifiers were removed from inputs. Preprocessing was fitted inside training folds and test results did not influence selection. These practices reduce disclosure and evaluation-leakage risks but do not prove lawful provenance or consent. Dataset redistribution rights remain unresolved. No claim is made that the original collection satisfied any particular legal standard.

Gender and SeniorCitizen remain predictors as approved in the plan. Their use creates discrimination and proxy risks and needs justification before any operational use. Aggregate subgroup monitoring alone cannot make that use acceptable. A real organisation should assess necessity, consent/provenance, access controls and appropriate retention with its responsible reviewers. Do not use the model to deny service, impose prices or make irreversible decisions automatically. Human review, accessible explanations and a route to challenge a decision would be needed.

## Proposed monitoring and controls

This project is not deployed. Before deployment, establish baseline input distributions, missing-value rates, churn prevalence and subgroup error rates on representative recent data. Monitor incoming schema/unknown categories and delayed-label precision, recall, F1, calibration and subgroup FNR/FPR. Alert thresholds and review frequency must be agreed from business costs and sample sizes; no validated numerical alert limits are available here.

Keep versioned data/model configurations, evaluation evidence and access logs. Investigate drift or material subgroup deterioration with human review. Retraining must use a new controlled development dataset and an independent evaluation set, followed by approval. Preserve the previous approved version and a rollback procedure. Stop automated use when quality or data provenance cannot be established. Avoid unnecessary retention outreach and excessive retention of personal information.

## Limitations and demonstration

The static dataset may not represent current customers or another provider. No time-based validation, prospective deployment, business-impact experiment or fairness guarantee is available. Class imbalance, correlated service attributes, a single split and tuning-selection optimism limit interpretation. No resampling or extra feature engineering was introduced after test access. Future improvements require a new evaluation design rather than repeated tuning against this test set.

Run `python -m src.evaluate` from the repository root. It verifies the frozen hash and reproduces tables, figures and outputs/prediction_example.json. The example uses categorical training modes and specified numeric values; it is synthetic, not a customer record. Its probability is not a calibrated guarantee. Model binaries are not saved.
