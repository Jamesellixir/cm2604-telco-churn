"""Training-only cross-validation, baseline and default-model exports."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.exceptions import ConvergenceWarning

from src.data import CSV_SHA256, load_data, split_data
from src.models import build_baseline, build_default_models, fixed_search_grids


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


def evaluate_defaults(inputs, target):
    """Evaluate the baseline and three defaults with identical folds and metrics."""
    models = {"baseline": build_baseline(), **build_default_models()}
    return {name: evaluate_pipeline(pipeline, inputs, target)
            for name, pipeline in models.items()}


def export_default_comparison(results):
    """Export only aggregate scores, times and the actual starting parameters."""
    fold_tables, summary_tables = [], []
    for name, scores in results.items():
        folds, summary = summarise_cv(scores)
        fold_tables.append(folds.assign(model=name))
        summary_tables.append(summary.assign(model=name))
    folds = pd.concat(fold_tables, ignore_index=True)
    summary = pd.concat(summary_tables, ignore_index=True)
    output = Path(__file__).resolve().parents[1] / "outputs" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "default_cv_folds.csv", index=False)
    summary.to_csv(output / "default_cv_summary.csv", index=False)
    models = {"baseline": build_baseline(), **build_default_models()}
    first = next(iter(results.values()))
    config = {
        "stage": "default comparison; no tuning or final selection",
        "csv_sha256": CSV_SHA256, "scikit_learn_version": sklearn.__version__,
        "split_seed": 42, "test_fraction": 0.2, "cv_folds": 5,
        "cv_shuffle": True, "cv_seed": 42, "positive_label": 1,
        "threshold": 0.5, "std_ddof": 0, "zero_division": 0,
        "training_rows": len(first["indices"]["train"][0]) + len(first["indices"]["test"][0]),
        "metrics": list(SCORING),
        "models": {name: {"class": type(pipe["model"]).__name__,
                           "parameters": pipe["model"].get_params(deep=False),
                           "numeric_scaling": name in {"knn", "logistic_regression"}}
                   for name, pipe in models.items()},
        "timing_note": "Wall-clock fit/score times vary; score values should reproduce.",
    }
    (output / "default_model_configuration.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return folds, summary


def build_search(name):
    """Search a complete pipeline, fitting preprocessing separately in each fold."""
    return GridSearchCV(build_default_models()[name], fixed_search_grids()[name],
                        scoring=SCORING, refit="f1", cv=fixed_folds(),
                        n_jobs=1, error_score="raise", return_train_score=False)


def search_result_table(search):
    """Keep every candidate's parameters, fold scores, ranks and timing summaries."""
    raw = search.cv_results_
    columns = {key: value for key, value in raw.items()
               if key.startswith(("split", "mean_", "std_", "rank_"))}
    table = pd.DataFrame(columns)
    table.insert(0, "candidate", range(len(table)))
    table.insert(1, "parameters_json", [json.dumps(params, sort_keys=True) for params in raw["params"]])
    return table


def choose_configuration(comparison):
    """Select among the six required defaults/tuned models using mean churn F1."""
    rows = comparison[comparison.metric.eq("f1")]
    expected = {(name, setting) for name in build_default_models()
                for setting in ["default", "tuned"]}
    if len(rows) != 6 or set(zip(rows.model, rows.setting)) != expected:
        raise ValueError("Selection requires one F1 row for each of six configurations")
    if not np.isfinite(rows["mean"]).all() or not rows["mean"].between(0, 1).all():
        raise ValueError("Selection scores must be finite F1 values in [0, 1]")
    priority = {"logistic_regression": 0, "knn": 1, "random_forest": 2}
    return min(rows.to_dict("records"), key=lambda row: (
        -row["mean"], priority[row["model"]], row["setting"] != "default"))


def freeze_configurations(comparison, best_parameters, training_index, path):
    """Save six reconstructible settings and the selected one; refuse changed freezes."""
    chosen = choose_configuration(comparison)
    configurations = []
    for name, default in build_default_models().items():
        for setting in ["default", "tuned"]:
            changes = {} if setting == "default" else best_parameters[name]
            model = default["model"]
            parameters = model.get_params(deep=False)
            parameters.update({key.removeprefix("model__"): value for key, value in changes.items()})
            score = comparison.loc[(comparison.model == name) & (comparison.setting == setting)
                                   & comparison.metric.eq("f1"), "mean"].iloc[0]
            configurations.append({"id": f"{name}_{setting}", "model": name,
                                   "setting": setting, "parameters": parameters,
                                   "numeric_scaling": name in {"knn", "logistic_regression"},
                                   "cv_f1": float(score)})
    frozen = {
        "schema_version": 1, "csv_sha256": CSV_SHA256,
        "scikit_learn_version": sklearn.__version__, "split_seed": 42, "test_fraction": 0.2,
        "training_rows": len(training_index),
        "training_index_sha256": hashlib.sha256(
            json.dumps(list(map(int, training_index))).encode()).hexdigest(),
        "cv_folds": 5, "cv_shuffle": True, "cv_seed": 42,
        "positive_label": 1, "threshold": 0.5,
        "preprocessing": "numeric fold median; optional StandardScaler; categorical fold mode and one-hot unknown ignore",
        "selected_id": f"{chosen['model']}_{chosen['setting']}",
        "selection_rule": "Highest mean training CV F1; exact ties: logistic regression, KNN, random forest; within a model prefer default",
        "search_ties": "First tied candidate in deterministic ParameterGrid order",
        "selection_score_warning": "Tuning scores helped choose parameters and may be optimistic; no held-out results used",
        "configurations": configurations,
    }
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != frozen:
            raise ValueError("Frozen configuration differs; retain it and obtain approval before changing it")
    else:
        path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


def run_tuning(inputs, target):
    """Run the bounded grids on training data and freeze choices before test access."""
    if not inputs.index.equals(target.index) or not inputs.index.is_unique:
        raise ValueError("Inputs and target must have matching unique row indices")
    if set(target.unique()) != {0, 1} or target.value_counts().min() < 5:
        raise ValueError("Five-fold tuning requires at least five rows in each binary class")
    output = Path(__file__).resolve().parents[1] / "outputs"
    tables = output / "tables"
    defaults = pd.read_csv(tables / "default_cv_summary.csv")
    config = json.loads((tables / "default_model_configuration.json").read_text(encoding="utf-8"))
    if (config["csv_sha256"] != CSV_SHA256 or config["training_rows"] != len(inputs)
            or config["cv_seed"] != 42 or config["cv_folds"] != 5
            or config["test_fraction"] != 0.2 or config["split_seed"] != 42
            or config["scikit_learn_version"] != sklearn.__version__):
        raise ValueError("Saved defaults do not match the approved training design")
    for name, pipeline in build_default_models().items():
        if config["models"][name]["parameters"] != pipeline["model"].get_params(deep=False):
            raise ValueError("Default model parameters changed; reproduce comparison before tuning")
    comparison = [defaults[defaults.model.ne("baseline")].assign(setting="default")]
    chosen_parameters, search_metadata = {}, {}
    for name in build_default_models():
        print(f"Searching {name} on five training folds...", flush=True)
        search = build_search(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(inputs, target)
        table = search_result_table(search)
        table.to_csv(tables / f"{name}_search_results.csv", index=False)
        chosen_parameters[name] = search.best_params_
        index = search.best_index_
        rows = [{"model": name, "setting": "tuned", "metric": metric,
                 "mean": float(search.cv_results_[f"mean_test_{metric}"][index]),
                 "std": float(search.cv_results_[f"std_test_{metric}"][index])}
                for metric in SCORING]
        rows.extend({"model": name, "setting": "tuned", "metric": label,
                     "mean": float(search.cv_results_[f"mean_{key}"][index]),
                     "std": float(search.cv_results_[f"std_{key}"][index])}
                    for label, key in [("fit_seconds", "fit_time"), ("score_seconds", "score_time")])
        comparison.append(pd.DataFrame(rows))
        search_metadata[name] = {"grid": fixed_search_grids()[name], "candidates": len(table),
                                 "best_index": int(index), "best_parameters": search.best_params_,
                                 "best_cv_f1": float(search.best_score_),
                                 "refit_seconds": float(search.refit_time_)}
        print(f"Finished {name}: mean CV F1={search.best_score_:.6f}", flush=True)
    comparison = pd.concat(comparison, ignore_index=True)
    comparison.to_csv(tables / "default_tuned_comparison.csv", index=False)
    (tables / "tuning_configuration.json").write_text(
        json.dumps(search_metadata, indent=2) + "\n", encoding="utf-8")
    frozen = freeze_configurations(comparison, chosen_parameters, inputs.index,
                                   output / "frozen_configuration.json")
    print(f"Frozen selection: {frozen['selected_id']}", flush=True)
    return comparison, frozen


def main():
    parser = argparse.ArgumentParser(description="Run training-only cross-validation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--defaults", action="store_true", help="Compare baseline and all three default models")
    group.add_argument("--tune", action="store_true", help="Run the approved grids and freeze validation choices")
    args = parser.parse_args()
    inputs, held_out, target, held_out_target = split_data(load_data())
    del held_out, held_out_target
    if args.tune:
        run_tuning(inputs, target)
        return
    if args.defaults:
        _, summary = export_default_comparison(evaluate_defaults(inputs, target))
        print(summary.pivot(index="model", columns="metric", values="mean").to_string())
        print("Saved training-only default comparison in outputs/tables/.")
        return
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
