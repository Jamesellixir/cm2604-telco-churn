"""Check default model behaviour and shared cross-validation on synthetic data."""

import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from src.models import build_default_models
from src.train import evaluate_defaults, summarise_cv
from test_baseline import training_fixture


class DefaultModelTests(unittest.TestCase):
    def test_plan_settings_and_unfitted_independent_pipelines(self):
        models = build_default_models()
        self.assertEqual(set(models), {"knn", "logistic_regression", "random_forest"})
        self.assertEqual(models["knn"]["model"].n_neighbors, 5)
        self.assertEqual(models["knn"]["model"].weights, "uniform")
        self.assertEqual(models["knn"]["model"].p, 2)
        self.assertEqual(models["logistic_regression"]["model"].max_iter, 2000)
        self.assertEqual(models["logistic_regression"]["model"].C, 1)
        self.assertEqual(models["random_forest"]["model"].n_estimators, 100)
        self.assertEqual(models["random_forest"]["model"].random_state, 42)
        self.assertEqual(len({id(p["preprocess"]) for p in models.values()}), 3)
        X, _ = training_fixture()
        for pipeline in models.values():
            with self.assertRaises(NotFittedError):
                pipeline.predict(X)

    def test_shared_folds_finite_scores_and_probabilities(self):
        X, y = training_fixture()
        results = evaluate_defaults(X, y)
        reference = results["baseline"]["indices"]
        for name, result in results.items():
            folds, summary = summarise_cv(result)
            self.assertEqual(len(folds), 5)
            self.assertTrue(np.isfinite(summary[["mean", "std"]].to_numpy()).all())
            for training, validation, expected_train, expected_val, model in zip(
                    result["indices"]["train"], result["indices"]["test"],
                    reference["train"], reference["test"], result["estimator"]):
                np.testing.assert_array_equal(training, expected_train)
                np.testing.assert_array_equal(validation, expected_val)
                probabilities = model.predict_proba(X.iloc[validation])
                self.assertEqual(probabilities.shape, (len(validation), 2))
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                numeric = model["preprocess"].named_transformers_["numeric"]
                self.assertEqual("scale" in numeric.named_steps,
                                 name in {"knn", "logistic_regression"})

    def test_prediction_rules_against_independent_calculations(self):
        X, y = training_fixture()
        for name, pipeline in build_default_models().items():
            pipeline.fit(X, y)
            inputs = pipeline["preprocess"].transform(X.iloc[:3])
            probabilities = pipeline.predict_proba(X.iloc[:3])[:, 1]
            model = pipeline["model"]
            if name == "knn":
                training = pipeline["preprocess"].transform(X)
                distances = ((inputs[:, None, :] - training[None, :, :]) ** 2).sum(axis=2)
                neighbours = np.argsort(distances, axis=1)[:, :5]
                expected = np.array([y.iloc[indices].mean() for indices in neighbours])
            elif name == "logistic_regression":
                logits = inputs @ model.coef_[0] + model.intercept_[0]
                expected = 1 / (1 + np.exp(-logits))
            else:
                expected = np.mean([tree.predict_proba(inputs)[:, 1]
                                    for tree in model.estimators_], axis=0)
            np.testing.assert_allclose(probabilities, expected)
