"""Construct unfitted model pipelines for the coursework."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_baseline():
    """Always predict the majority class learned from the current training fold."""
    return Pipeline([
        ("preprocess", build_preprocessor(scaled=False)),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])


def build_default_models():
    """Construct the three starting configurations fixed in the coursework plan."""
    return {
        "knn": Pipeline([
            ("preprocess", build_preprocessor(scaled=True)),
            ("model", KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2)),
        ]),
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor(scaled=True)),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                         class_weight=None)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor(scaled=False)),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=None,
                                            random_state=42, n_jobs=1)),
        ]),
    }
