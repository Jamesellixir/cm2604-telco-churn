"""Unfitted preprocessing for the fixed 19 customer predictors."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORIES, NUMERIC_COLUMNS


def convert_numeric(values):
    """Convert numeric text, mapping blanks to NaN without learning statistics."""
    result = values.copy()
    for column in result.columns:
        text = result[column].astype("string").str.strip()
        text = text.mask(text.isna() | text.eq(""))
        try:
            numbers = pd.to_numeric(text, errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} contains malformed numeric values") from exc
        present = numbers.dropna()
        if not np.isfinite(present).all() or (present < 0).any():
            raise ValueError(f"{column} must contain finite non-negative values")
        if column == "tenure" and (present % 1 != 0).any():
            raise ValueError("tenure must contain whole months")
        result[column] = numbers
    return result


def convert_categories(values):
    """Use strings for categories and NumPy NaN for missing or blank cells.

    Preserve service categories and allow new labels at transform time. The
    original source CSV still passes the stricter validation in data.py.
    """
    result = values.copy()
    for column in result.columns:
        text = result[column].astype("string").str.strip()
        result[column] = text.astype(object)
        result.loc[text.isna() | text.eq(""), column] = np.nan
    return result


def build_preprocessor(*, scaled=True):
    """Construct an unfitted transformer; fit it inside each model pipeline.

    KNN/logistic regression use scaled numeric values. Random forest and the
    dummy baseline use the unscaled branch. Categorical indicators are unscaled.
    """
    numeric_steps = [
        ("convert", FunctionTransformer(convert_numeric, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scaled:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("convert", FunctionTransformer(convert_categories, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
        ("categorical", categorical, list(CATEGORIES)),
    ], remainder="drop")
