"""Verify the baseline mathematically and audit fitted estimators per fold."""

import unittest

import numpy as np
import pandas as pd

from src.models import build_baseline
from src.preprocessing import build_preprocessor, convert_numeric
from src.data import NUMERIC_COLUMNS
from src.train import evaluate_pipeline, fixed_folds, summarise_cv
from test_preprocessing import predictor_fixture


def training_fixture():
    inputs = pd.concat([predictor_fixture()] * 10, ignore_index=True)
    inputs["TotalCharges"] = [np.nan if i % 7 == 0 else float(i * i) for i in range(40)]
    inputs["Contract"] = [f"synthetic-category-{i}" for i in range(40)]
    target = pd.Series([int(i % 4 == 0) for i in range(40)], name="Churn")
    return inputs, target


class BaselineTests(unittest.TestCase):
    def test_constant_predictions_match_hand_calculated_metrics(self):
        inputs, target = training_fixture()
        pipeline = build_baseline()
        results = evaluate_pipeline(pipeline, inputs, target)
        self.assertFalse(hasattr(pipeline["preprocess"], "transformers_"))
        folds, summary = summarise_cv(results)
        self.assertEqual(len(folds), 5)
        expected = {"accuracy": .75, "balanced_accuracy": .5, "f1": 0,
                    "precision": 0, "recall": 0, "roc_auc": .5, "average_precision": .25}
        for metric, value in expected.items():
            np.testing.assert_allclose(folds[metric], value)
            self.assertAlmostEqual(summary.set_index("metric").loc[metric, "mean"], value)
            self.assertAlmostEqual(summary.set_index("metric").loc[metric, "std"], 0)
        seen = []
        for estimator, training, validation in zip(results["estimator"],
                results["indices"]["train"], results["indices"]["test"]):
            self.assertFalse(set(training) & set(validation))
            seen.extend(validation)
            predictions = estimator.predict(inputs.iloc[validation])
            probability = estimator.predict_proba(inputs.iloc[validation])
            np.testing.assert_array_equal(predictions, np.zeros(len(validation)))
            np.testing.assert_array_equal(probability[:, 1], np.zeros(len(validation)))
            np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(sorted(seen), list(range(len(inputs))))

    def test_fold_medians_and_categories_exclude_validation(self):
        inputs, target = training_fixture()
        pipeline = build_baseline().set_params(preprocess=build_preprocessor(scaled=True))
        results = evaluate_pipeline(pipeline, inputs, target)
        fitted_ids = []
        differs_from_full_data = []
        for estimator, training, validation in zip(results["estimator"],
                results["indices"]["train"], results["indices"]["test"]):
            preprocess = estimator["preprocess"]
            fitted_ids.append(id(preprocess))
            actual = preprocess.named_transformers_["numeric"]["impute"].statistics_[2]
            expected = np.nanmedian(inputs.iloc[training]["TotalCharges"].astype(float))
            self.assertEqual(actual, expected)
            differs_from_full_data.append(actual != np.nanmedian(inputs["TotalCharges"].astype(float)))
            numeric = preprocess.named_transformers_["numeric"]
            fold_values = convert_numeric(inputs.iloc[training][NUMERIC_COLUMNS])
            expected_mean = fold_values.fillna(fold_values.median()).mean().to_numpy()
            np.testing.assert_allclose(numeric["scale"].mean_, expected_mean)
            categorical = preprocess.named_transformers_["categorical"]
            column = list(categorical["convert"].feature_names_in_).index("Contract")
            fitted_categories = set(categorical["encode"].categories_[column])
            self.assertEqual(fitted_categories, set(inputs.iloc[training]["Contract"]))
            self.assertFalse(fitted_categories & set(inputs.iloc[validation]["Contract"]))
        self.assertEqual(len(set(fitted_ids)), 5)
        self.assertTrue(any(differs_from_full_data))

    def test_fixed_folds_and_target_checks(self):
        inputs, target = training_fixture()
        for first, second in zip(fixed_folds().split(inputs, target), fixed_folds().split(inputs, target)):
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)
        for bad in [target.iloc[::-1], target.replace(1, 2), pd.Series([0] * 40)]:
            with self.assertRaises(ValueError):
                evaluate_pipeline(build_baseline(), inputs, bad)
