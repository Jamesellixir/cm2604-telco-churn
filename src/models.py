"""Construct unfitted model pipelines for the coursework."""

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_baseline():
    """Always predict the majority class learned from the current training fold."""
    return Pipeline([
        ("preprocess", build_preprocessor(scaled=False)),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])
