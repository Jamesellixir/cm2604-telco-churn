"""Small synthetic fixtures exercise validation without publishing customer data."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data import CATEGORIES, load_data, main, separate_inputs_target, validate_frame


def fixture():
    row = {name: sorted(values)[0] for name, values in CATEGORIES.items()}
    row.update(customerID="sample-1", Churn="Yes", tenure="0",
               MonthlyCharges="20.5", TotalCharges=" ")
    other = dict(row, customerID="sample-2", Churn="No", tenure="2", TotalCharges="41")
    return pd.DataFrame([row, other])


class DataTests(unittest.TestCase):
    def test_conversion_and_target(self):
        frame = fixture()
        clean = validate_frame(frame)
        self.assertTrue(pd.isna(clean.loc[0, "TotalCharges"]))
        self.assertEqual(clean.loc[1, "TotalCharges"], 41.0)
        self.assertEqual(frame.loc[0, "TotalCharges"], " ")
        x, y = separate_inputs_target(clean)
        self.assertEqual(x.shape, (2, 19))
        self.assertNotIn("customerID", x)
        self.assertNotIn("Churn", x)
        self.assertEqual(y.tolist(), [1, 0])

    def test_missing_extra_duplicate_columns_and_empty(self):
        f = fixture()
        for bad in [f.drop(columns="gender"), f.assign(extra="bad"), f.iloc[:0],
                    pd.concat([f, f[["gender"]]], axis=1)]:
            with self.subTest(columns=list(bad.columns)), self.assertRaises(ValueError):
                validate_frame(bad)

    def test_invalid_values(self):
        cases = [("customerID", "sample-2"), ("customerID", " "),
                 ("customerID", None), ("Churn", "Maybe"), ("Churn", ""),
                 ("gender", "Unknown"), ("SeniorCitizen", "2"),
                 ("TotalCharges", "bad"), ("TotalCharges", "inf"),
                 ("TotalCharges", "-1"), ("MonthlyCharges", ""),
                 ("MonthlyCharges", "NaN"), ("tenure", "-1"), ("tenure", "1.5")]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                frame = fixture()
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_frame(frame)

    def test_csv_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            fixture().to_csv(path, index=False)
            loaded = load_data(path, verify_checksum=False)
            self.assertEqual(len(loaded), 2)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(path)
            path.write_text("gender,gender\nMale,Male\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_data(path, verify_checksum=False)
            path.write_text("a,b\n1,2,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "field count"):
                load_data(path, verify_checksum=False)
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                load_data(path, verify_checksum=False)

    def test_missing_path_and_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent.csv"
            with self.assertRaisesRegex(ValueError, "not found"):
                load_data(path)
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                self.assertEqual(main(["--path", str(path)]), 1)
            self.assertIn("Data error:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
