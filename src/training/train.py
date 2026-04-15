"""Model training CLI for identity theft detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ProjectConfig
from src.evaluation.metrics import (
    cross_validate_model,
    evaluate_model,
    save_confusion_matrix,
    save_eda_figures,
    save_feature_importance_plot,
    save_model_comparison,
    save_roc_curve,
)
from src.models.factory import create_model, supported_models
from src.preprocessing.data_loader import load_data, validate_target_column
from src.preprocessing.pipeline import build_preprocessor, prepare_dataset_bundle, prepare_splits
from src.utils.io import ensure_directory, write_json


def train_model(model_name: str, X_train, y_train, top_k_features: int = 20) -> Pipeline:
    """Fit a pipeline for a given model name."""
    preprocessor = build_preprocessor(X_train, top_k_features=top_k_features)
    model = create_model(model_name)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def infer_feature_names(model: Pipeline, X_train: pd.DataFrame) -> list[str]:
    """Recover transformed feature names when possible."""
    prep_block = model.named_steps["preprocessor"]
    selector = None
    transformer = prep_block

    if hasattr(prep_block, "named_steps"):
        transformer = prep_block.named_steps["preprocessor"]
        selector = prep_block.named_steps.get("selector")

    try:
        names = transformer.get_feature_names_out().tolist()
    except Exception:
        names = X_train.columns.tolist()

    if selector is not None and hasattr(selector, "get_support"):
        mask = selector.get_support()
        names = [name for name, keep in zip(names, mask) if keep]
    return names


def build_prediction_preview(model: Pipeline, X_test: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    """Create a small CSV-friendly preview for demo purposes."""
    preview = X_test.head(10).copy()
    probabilities = model.predict_proba(preview)[:, 1] if hasattr(model, "predict_proba") else [0.0] * len(preview)
    predictions = model.predict(preview)
    preview["fraud_probability"] = probabilities
    preview["prediction"] = predictions
    preview["prediction_reason"] = (
        "Top model signals: " + ", ".join(top_features[:3])
        if top_features
        else "Based on fitted model patterns."
    )
    return preview


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train an identity theft detection model.")
    parser.add_argument("--data", required=True, help="Path to the raw CSV dataset.")
    parser.add_argument("--config", default="config/project_config.json", help="Path to JSON config.")
    parser.add_argument("--target", default=None, help="Override the target column name.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model artifacts.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for processed CSVs.")
    parser.add_argument("--top-k-features", type=int, default=None, help="Number of selected features.")
    parser.add_argument("--test-size", type=float, default=None, help="Test split size.")
    parser.add_argument("--validation-size", type=float, default=None, help="Validation split size.")
    return parser.parse_args()


def main() -> None:
    """Train models, save metrics, and export report-ready artifacts."""
    args = parse_args()
    config = ProjectConfig.from_json(args.config) if Path(args.config).exists() else ProjectConfig()

    if args.target:
        config.target_column = args.target
    if args.top_k_features is not None:
        config.top_k_features = args.top_k_features
    if args.test_size is not None:
        config.test_size = args.test_size
    if args.validation_size is not None:
        config.validation_size = args.validation_size

    output_dir = ensure_directory(args.output_dir)
    processed_dir = ensure_directory(args.processed_dir)
    figures_dir = ensure_directory(output_dir / "figures")

    raw_df = load_data(args.data)
    validate_target_column(raw_df, config.target_column)
    save_eda_figures(raw_df, config.target_column, figures_dir)

    train_df, val_df, test_df = prepare_splits(raw_df, config)
    train_df.to_csv(processed_dir / "train_split.csv", index=False)
    val_df.to_csv(processed_dir / "validation_split.csv", index=False)
    test_df.to_csv(processed_dir / "test_split.csv", index=False)

    bundle = prepare_dataset_bundle(raw_df, config)

    leaderboard_rows = []
    best_model = None
    best_model_name = None
    best_score = -1.0
    best_cv = {}

    for model_name in supported_models():
        pipeline = train_model(model_name, bundle.X_train, bundle.y_train, config.top_k_features)
        metrics = evaluate_model(pipeline, bundle.X_test, bundle.y_test)
        cv_metrics = cross_validate_model(pipeline, bundle.X_train, bundle.y_train)
        score = (metrics["f1_score"] + metrics["recall"]) / 2

        leaderboard_rows.append(
            {
                "model_name": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"] if metrics["roc_auc"] is not None else 0.0,
                "cv_f1_mean": cv_metrics["f1"]["mean"],
                "cv_f1_std": cv_metrics["f1"]["std"],
            }
        )

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_model_name = model_name
            best_cv = cv_metrics

    if best_model is None or best_model_name is None:
        raise RuntimeError("Training did not produce a valid best model.")

    results_df = pd.DataFrame(leaderboard_rows).sort_values(by="f1_score", ascending=False)
    results_df.to_csv(output_dir / "model_leaderboard.csv", index=False)
    save_model_comparison(results_df, figures_dir / "model_comparison.png")

    best_metrics = evaluate_model(best_model, bundle.X_test, bundle.y_test)
    save_confusion_matrix(best_model, bundle.X_test, bundle.y_test, figures_dir / "confusion_matrix.png")
    save_roc_curve(best_model, bundle.X_test, bundle.y_test, figures_dir / "roc_curve.png")

    feature_names = infer_feature_names(best_model, bundle.X_train)
    top_features = save_feature_importance_plot(best_model, feature_names, figures_dir / "feature_importance.png")

    sample_predictions = build_prediction_preview(best_model, bundle.X_test, top_features)
    sample_predictions.to_csv(output_dir / "sample_predictions.csv", index=False)
    sample_predictions.drop(
        columns=["fraud_probability", "prediction", "prediction_reason"]
    ).to_csv(processed_dir / "sample_predictions_input.csv", index=False)

    metadata = {
        "best_model_name": best_model_name,
        "target_column": config.target_column,
        "top_features": top_features,
        "train_shape": list(bundle.X_train.shape),
        "validation_shape": list(bundle.X_val.shape),
        "test_shape": list(bundle.X_test.shape),
        "columns_seen_during_training": bundle.X_train.columns.tolist(),
        "cross_validation": best_cv,
    }

    write_json(
        output_dir / "metrics.json",
        {"best_model": best_model_name, "test_metrics": best_metrics, "leaderboard": leaderboard_rows},
    )
    write_json(output_dir / "model_metadata.json", metadata)
    joblib.dump(best_model, output_dir / "best_model.joblib")

    summary = {
        "status": "success",
        "best_model": best_model_name,
        "artifacts_dir": str(output_dir),
        "processed_dir": str(processed_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
