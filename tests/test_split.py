"""Check disjoint partitions, target alignment and stable stratification."""

import unittest

import pandas as pd

from src.data import split_data
from test_data import fixture


def split_fixture():
    frame = pd.concat([fixture()] * 50, ignore_index=True)
    frame["customerID"] = [f"sample-{n}" for n in range(len(frame))]
    frame["Churn"] = ["Yes" if n % 4 == 0 else "No" for n in range(len(frame))]
    frame.index = range(1000, 1100)
    return frame


class SplitTests(unittest.TestCase):
    def test_fixed_disjoint_stratified_and_aligned(self):
        frame = split_fixture()
        original = frame.copy(deep=True)
        train, test, y_train, y_test = split_data(frame)
        self.assertEqual((len(train), len(test)), (80, 20))
        self.assertFalse(set(train.index) & set(test.index))
        self.assertEqual(set(train.index) | set(test.index), set(frame.index))
        self.assertEqual(y_train.sum(), 20)
        self.assertEqual(y_test.sum(), 5)
        for inputs, target in [(train, y_train), (test, y_test)]:
            self.assertEqual(inputs.shape[1], 19)
            self.assertNotIn("customerID", inputs)
            self.assertNotIn("Churn", inputs)
            pd.testing.assert_series_equal(
                target, frame.loc[inputs.index, "Churn"].map({"No": 0, "Yes": 1}))
        for first, second in zip((train, test, y_train, y_test), split_data(frame)):
            self.assertTrue(first.equals(second))
        pd.testing.assert_frame_equal(frame, original)

    def test_invalid_class_counts_and_indices(self):
        one_class = split_fixture().assign(Churn="No")
        one_rare = one_class.copy()
        one_rare.loc[1000, "Churn"] = "Yes"
        duplicate_index = split_fixture()
        duplicate_index.index = [0] * len(duplicate_index)
        for frame in [one_class, one_rare, duplicate_index, fixture()]:
            with self.subTest(rows=len(frame)), self.assertRaises(ValueError):
                split_data(frame)


if __name__ == "__main__":
    unittest.main()
