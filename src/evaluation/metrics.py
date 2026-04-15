"""Metrics, reports, and plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.utils.io import ensure_directory


def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    """Return the core evaluation metrics required for the assignment."""
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probabilities))
    else:
        roc_auc = None

    return {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }


def cross_validate_model(model: Pipeline, X_train, y_train) -> dict:
    """Compute cross-validation means and standard deviations."""
    scoring = ["precision", "recall", "f1", "roc_auc"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
    summary = {}
    for key, values in results.items():
        if key.startswith("test_"):
            metric_name = key.replace("test_", "")
            summary[metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
    return summary


def save_confusion_matrix(model: Pipeline, X_test, y_test, path: str | Path) -> None:
    """Save a confusion matrix heatmap."""
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)
    ensure_directory(Path(path).parent)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_roc_curve(model: Pipeline, X_test, y_test, path: str | Path) -> None:
    """Save a ROC curve if the model supports probabilities."""
    if not hasattr(model, "predict_proba"):
        return

    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)
    ensure_directory(Path(path).parent)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_model_comparison(results_df: pd.DataFrame, path: str | Path) -> None:
    """Save a model comparison chart."""
    ensure_directory(Path(path).parent)
    melted = results_df.melt(id_vars="model_name", value_vars=["precision", "recall", "f1_score", "roc_auc"])
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="model_name", y="value", hue="variable")
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_feature_importance_plot(model: Pipeline, feature_names: list[str], path: str | Path) -> list[str]:
    """Save feature importance or coefficient plot for the final model."""
    final_estimator = model.named_steps["model"]
    top_features: list[str] = []

    if hasattr(final_estimator, "feature_importances_"):
        importances = final_estimator.feature_importances_
    elif hasattr(final_estimator, "coef_"):
        importances = abs(final_estimator.coef_[0])
    else:
        return top_features

    feature_importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    top_features = feature_importance_df["feature"].tolist()
    ensure_directory(Path(path).parent)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="importance", y="feature", orient="h")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return top_features


def save_eda_figures(df: pd.DataFrame, target_column: str, output_dir: str | Path) -> None:
    """Generate a starter set of EDA figures for the report."""
    output_dir = ensure_directory(output_dir)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    missing_counts = df.isna().sum().sort_values(ascending=False).head(15)
    sns.barplot(x=missing_counts.index, y=missing_counts.values)
    plt.title("Top Missing Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png")
    plt.close()

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_columns) >= 2:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_columns].corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()

    for index, column in enumerate(numeric_columns[:3], start=1):
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f"Distribution: {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"distribution_{index}_{column}.png")
        plt.close()
