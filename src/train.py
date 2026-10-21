"""Training-only cross-validation and majority-class baseline exports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import CSV_SHA256, load_data, split_data
from src.models import build_baseline


SCORING = {
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def fixed_folds():
    """Use this same shuffled, stratified five-fold design for every model."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_pipeline(pipeline, inputs, target):
    """Evaluate an unfitted pipeline on training data; retain folds for audits.

    Each fold clones and fits the entire pipeline. Never pass the held-out
    test partition to this function. Indices and estimators remain in memory.
    """
    if not inputs.index.equals(target.index) or not inputs.index.is_unique:
        raise ValueError("Inputs and target must have matching unique row indices")
    if set(target.unique()) != {0, 1} or target.value_counts().min() < 5:
        raise ValueError("Five-fold evaluation requires at least five rows in each binary class")
    return cross_validate(pipeline, inputs, target, cv=fixed_folds(), scoring=SCORING,
                          n_jobs=1, error_score="raise", return_estimator=True,
                          return_indices=True)


def summarise_cv(results):
    """Return fold scores and mean/population-standard-deviation summaries."""
    folds = pd.DataFrame({
        "fold": np.arange(1, len(results["fit_time"]) + 1),
        "training_rows": [len(indices) for indices in results["indices"]["train"]],
        "validation_rows": [len(indices) for indices in results["indices"]["test"]],
        "fit_seconds": results["fit_time"], "score_seconds": results["score_time"],
        **{name: results[f"test_{name}"] for name in SCORING},
    })
    summary = pd.DataFrame([
        {"metric": name, "mean": folds[name].mean(), "std": folds[name].std(ddof=0)}
        for name in [*SCORING, "fit_seconds", "score_seconds"]
    ])
    return folds, summary


def main():
    inputs, held_out, target, held_out_target = split_data(load_data())
    del held_out, held_out_target
    results = evaluate_pipeline(build_baseline(), inputs, target)
    folds, summary = summarise_cv(results)
    output = Path(__file__).resolve().parents[1] / "outputs" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "baseline_cv_folds.csv", index=False)
    summary.to_csv(output / "baseline_cv_summary.csv", index=False)
    config = {
        "model": "DummyClassifier", "strategy": "most_frequent",
        "numeric_scaling": False, "numeric_imputation": "median",
        "categorical_imputation": "most_frequent", "encoding": "one-hot, unknown ignored",
        "split_seed": 42, "test_fraction": 0.2, "cv_folds": 5,
        "cv_shuffle": True, "cv_seed": 42, "training_rows": len(inputs),
        "selection_metric": "f1", "positive_label": 1, "threshold": 0.5,
        "std_ddof": 0, "zero_division": 0, "scikit_learn_version": sklearn.__version__,
        "csv_sha256": CSV_SHA256,
        "timing_note": "Wall-clock times vary between runs; scores should reproduce.",
    }
    (output / "baseline_configuration.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("Saved training-only baseline results in outputs/tables/.")


if __name__ == "__main__":
    main()
