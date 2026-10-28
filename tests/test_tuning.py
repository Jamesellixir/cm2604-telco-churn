"""Check bounded grids, per-fold fitting, selection ties and freeze protection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid

from src.models import fixed_search_grids
from src.train import (build_search, choose_configuration, fixed_folds,
                       freeze_configurations, search_result_table, run_tuning)
from test_baseline import training_fixture


def comparison_fixture():
    return pd.DataFrame([{"model": name, "setting": setting, "metric": "f1", "mean": .6, "std": .1}
                         for name in fixed_search_grids() for setting in ["default", "tuned"]])


class TuningTests(unittest.TestCase):
    def test_tuning_rejects_misaligned_or_nonbinary_labels_before_search(self):
        X, y = training_fixture()
        for bad in [y.iloc[::-1], y.replace(1, 2), y * 0]:
            with self.assertRaises(ValueError):
                run_tuning(X, bad)

    def test_grids_match_bounded_plan(self):
        grids = fixed_search_grids()
        self.assertEqual({name: len(ParameterGrid(grid)) for name, grid in grids.items()},
                         {"knn": 16, "logistic_regression": 8, "random_forest": 16})
        self.assertEqual(grids["knn"]["model__n_neighbors"], [5, 11, 21, 31])
        self.assertEqual(grids["knn"]["model__weights"], ["uniform", "distance"])
        self.assertEqual(grids["knn"]["model__p"], [1, 2])
        self.assertEqual(grids["logistic_regression"]["model__C"], [.01, .1, 1, 10])
        self.assertEqual(grids["random_forest"]["model__max_depth"], [None, 10])
        self.assertEqual(grids["random_forest"]["model__min_samples_leaf"], [1, 5])
        for name in grids:
            search = build_search(name)
            self.assertEqual(search.refit, "f1")
            self.assertEqual(search.cv.n_splits, 5)
            self.assertEqual(search.cv.random_state, 42)
            self.assertEqual(search.error_score, "raise")

    def test_search_refits_preprocessing_in_every_training_fold(self):
        X, y = training_fixture()
        X["tenure"] = np.arange(len(X), dtype=float)
        search = build_search("knn").set_params(param_grid={"model__n_neighbors": [3, 5]})
        observed = []
        original = SimpleImputer.fit

        def record_fit(imputer, values, target=None, **kwargs):
            if values.shape[1] == 3:
                observed.append(set(values.iloc[:, 0].astype(int)))
            return original(imputer, values, target, **kwargs)

        with patch.object(SimpleImputer, "fit", new=record_fit):
            search.fit(X, y)
        expected = [set(training) for training, _ in fixed_folds().split(X, y)]
        self.assertEqual(observed, expected + expected + [set(range(len(X)))])
        table = search_result_table(search)
        self.assertEqual(len(table), 2)
        self.assertEqual(set(json.loads(table.iloc[0].parameters_json)), {"model__n_neighbors"})
        for _, row in table.iterrows():
            scores = [row[f"split{i}_test_f1"] for i in range(5)]
            self.assertAlmostEqual(row.mean_test_f1, np.mean(scores))
            self.assertAlmostEqual(row.std_test_f1, np.std(scores))
        self.assertAlmostEqual(search.best_score_, table.mean_test_f1.max())

    def test_selection_uses_scores_then_declared_ties(self):
        comparison = comparison_fixture()
        chosen = choose_configuration(comparison)
        self.assertEqual((chosen["model"], chosen["setting"]), ("logistic_regression", "default"))
        comparison.loc[comparison.model.eq("knn") & comparison.setting.eq("tuned"), "mean"] = .7
        self.assertEqual(choose_configuration(comparison)["model"], "knn")
        comparison.loc[0, "mean"] = np.nan
        with self.assertRaises(ValueError):
            choose_configuration(comparison)
        with self.assertRaises(ValueError):
            choose_configuration(comparison_fixture().iloc[:5])

    def test_freeze_roundtrip_and_changed_choice_protection(self):
        parameters = {"knn": {"model__n_neighbors": 11},
                      "logistic_regression": {"model__C": .1},
                      "random_forest": {"model__max_depth": 10}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            original = freeze_configurations(comparison_fixture(), parameters, range(40), path)
            snapshot = path.read_bytes()
            self.assertEqual(len(original["configurations"]), 6)
            repeated = freeze_configurations(comparison_fixture(), parameters, range(40), path)
            self.assertEqual(repeated, original)
            self.assertEqual(path.read_bytes(), snapshot)
            changed = comparison_fixture()
            changed.loc[0, "mean"] = .9
            with self.assertRaisesRegex(ValueError, "Frozen configuration differs"):
                freeze_configurations(changed, parameters, range(40), path)
            self.assertEqual(path.read_bytes(), snapshot)
