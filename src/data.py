"""Validate the source CSV and expose model inputs without customer identifiers."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CSV_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "source" / CSV_NAME
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORIES = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {"0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{name: {"Yes", "No", "No internet service"} for name in
       ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"]},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"},
}
EXPECTED_COLUMNS = {"customerID", "Churn", *NUMERIC_COLUMNS, *CATEGORIES}


def validate_frame(frame):
    """Validate a frame, returning a copy with numeric columns converted.

    Blank TotalCharges becomes NaN. Imputation is deferred to model pipelines.
    """
    if frame.columns.duplicated().any() or set(frame.columns) != EXPECTED_COLUMNS:
        missing = sorted(EXPECTED_COLUMNS - set(frame.columns))
        extra = sorted(set(frame.columns) - EXPECTED_COLUMNS)
        raise ValueError(f"Invalid columns; missing={missing}, unexpected={extra}; headers must be unique")
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")
    result = frame.copy()
    ids = result["customerID"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError("customerID must not be missing or blank")
    if ids.str.strip().duplicated().any():
        raise ValueError("customerID must be unique")
    for name, allowed in {**CATEGORIES, "Churn": {"No", "Yes"}}.items():
        if not result[name].isin(allowed).all():
            raise ValueError(f"{name} contains missing or invalid labels")
    for name in NUMERIC_COLUMNS:
        text = result[name].astype("string")
        blank = text.isna() | text.str.strip().eq("")
        if name != "TotalCharges" and blank.any():
            raise ValueError(f"{name} must not be missing or blank")
        try:
            values = pd.to_numeric(text.mask(blank), errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} contains malformed numeric values") from exc
        present = values.dropna()
        if not np.isfinite(present).all() or (present < 0).any():
            raise ValueError(f"{name} must contain finite non-negative values")
        if name == "tenure" and (present % 1 != 0).any():
            raise ValueError("tenure must contain whole months")
        result[name] = values
    return result


def load_data(path=DEFAULT_PATH, *, verify_checksum=True):
    """Load the expected dataset. Disable checksum only for local test fixtures."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"Dataset not found: {path}. Follow README.md dataset instructions.") from exc
    if verify_checksum and hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("CSV checksum mismatch; obtain the original dataset described in README.md")
    try:
        text = raw.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text), strict=True))
        if not rows or len(set(rows[0])) != len(rows[0]):
            raise ValueError("CSV must have unique column headers")
        if any(len(row) != len(rows[0]) for row in rows[1:]):
            raise ValueError("CSV rows must have the same field count as the header")
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Dataset must be a valid UTF-8 CSV with a header") from exc
    return validate_frame(frame)


def separate_inputs_target(frame):
    """Return the 19 predictors and binary target; retain row indices."""
    validated = validate_frame(frame)
    return (validated.drop(columns=["customerID", "Churn"]),
            validated["Churn"].map({"No": 0, "Yes": 1}).astype(int))


def split_data(frame):
    """Return X_train, X_test, y_train, y_test using the fixed coursework split.

    Keep original CSV row order and indices for reproducibility. No preprocessing
    is fitted here. The test partition is reserved for final model evaluation.
    """
    from sklearn.model_selection import train_test_split

    if not frame.index.is_unique:
        raise ValueError("Row indices must be unique before splitting")
    inputs, target = separate_inputs_target(frame)
    if set(target.unique()) != {0, 1} or target.value_counts().min() < 2:
        raise ValueError("Stratified splitting requires at least two rows in each target class")
    try:
        return train_test_split(inputs, target, test_size=0.2,
                                random_state=42, shuffle=True, stratify=target)
    except ValueError as exc:
        raise ValueError("Dataset is too small for the fixed stratified 80/20 split") from exc


def inspect_data(frame):
    """Return structural counts only; detailed exploration belongs to training data."""
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "predictors": len(frame.columns) - 2,
        "target_counts": dict(Counter(frame["Churn"])),
        "missing_total_charges": int(frame["TotalCharges"].isna().sum()),
        "duplicate_customer_ids": int(frame["customerID"].duplicated().sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate the original Telco CSV and print structural counts")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    try:
        frame = load_data(args.path)
    except (ValueError, OSError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(inspect_data(frame), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
