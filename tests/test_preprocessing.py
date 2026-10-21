"""Hand-checkable preprocessing and validation-data separation checks."""

import unittest

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.data import NUMERIC_COLUMNS
from src.preprocessing import build_preprocessor, convert_numeric
from test_data import fixture


def predictor_fixture():
    row = fixture().iloc[0].drop(labels=["customerID", "Churn"]).to_dict()
    frame = pd.DataFrame([row.copy() for _ in range(4)], dtype=object)
    frame["tenure"] = ["1", "3", " ", "5"]
    frame["MonthlyCharges"] = ["10", "30", None, "50"]
    frame["TotalCharges"] = ["10", "90", pd.NA, "250"]
    frame["Contract"] = ["One year", "One year", "Month-to-month", None]
    frame["OnlineSecurity"] = ["Yes", "No internet service", "No", "Yes"]
    return frame


class PreprocessingTests(unittest.TestCase):
    def test_numeric_conversion_and_imputation(self):
        frame = predictor_fixture()
        original = frame.copy(deep=True)
        transformer = build_preprocessor(scaled=False)
        self.assertFalse(hasattr(transformer, "transformers_"))
        values = transformer.fit_transform(frame)
        np.testing.assert_allclose(values[:, :3], [[1, 10, 10], [3, 30, 90],
                                                  [3, 30, 90], [5, 50, 250]])
        pd.testing.assert_frame_equal(frame, original)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(len(transformer.get_feature_names_out()), values.shape[1])

    def test_scaler_uses_imputed_training_values(self):
        transformer = build_preprocessor(scaled=True)
        transformer.fit(predictor_fixture())
        numeric = transformer.named_transformers_["numeric"]
        np.testing.assert_allclose(numeric["impute"].statistics_, [3, 30, 90])
        np.testing.assert_allclose(numeric["scale"].mean_, [3, 30, 110])
        np.testing.assert_allclose(numeric["scale"].var_, [2, 200, 7600])
        validation = predictor_fixture().iloc[:1].copy()
        validation.loc[0, NUMERIC_COLUMNS] = ["1000", "10000", " "]
        numeric_values = numeric.transform(validation[NUMERIC_COLUMNS])
        self.assertAlmostEqual(numeric_values[0, 0], (1000 - 3) / np.sqrt(2))
        self.assertAlmostEqual(numeric_values[0, 2], (90 - 110) / np.sqrt(7600))
        np.testing.assert_allclose(numeric["scale"].mean_, [3, 30, 110])
        np.testing.assert_allclose(numeric["impute"].statistics_, [3, 30, 90])

    def test_unseen_and_missing_categories_preserve_service_labels(self):
        transformer = build_preprocessor(scaled=False).fit(predictor_fixture())
        names = list(transformer.get_feature_names_out())
        self.assertIn("categorical__OnlineSecurity_No internet service", names)
        validation = predictor_fixture().iloc[:2].copy()
        validation.loc[0, "Contract"] = "Weekly"
        validation.loc[1, "Contract"] = pd.NA
        values = transformer.transform(validation)
        contract_columns = [i for i, name in enumerate(names) if name.startswith("categorical__Contract_")]
        self.assertEqual(values[0, contract_columns].sum(), 0)
        self.assertEqual(values[1, names.index("categorical__Contract_One year")], 1)
        self.assertEqual(values.shape[1], len(names))
        self.assertTrue(np.isfinite(values).all())

    def test_all_missing_training_numeric_column_is_retained(self):
        frame = predictor_fixture()
        frame["TotalCharges"] = np.nan
        transformer = build_preprocessor(scaled=False).fit(frame)
        self.assertIn("numeric__TotalCharges", transformer.get_feature_names_out())
        self.assertTrue((transformer.transform(frame)[:, 2] == 0).all())

    def test_malformed_numeric_and_unfitted_transform_fail(self):
        for value in ["not-a-number", "inf", "-1"]:
            frame = predictor_fixture()[NUMERIC_COLUMNS]
            frame.loc[0, "TotalCharges"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                convert_numeric(frame)
        with self.assertRaises(NotFittedError):
            build_preprocessor().transform(predictor_fixture())
