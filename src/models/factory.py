"""Model factory for the required baseline and ML models."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_model(model_name: str):
    """Create one of the supported models."""
    name = model_name.lower()
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def supported_models() -> list[str]:
    """Return the exact required models for the project."""
    return ["dummy", "logistic_regression", "random_forest"]
