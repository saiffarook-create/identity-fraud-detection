"""Batch prediction CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def predict_records(input_df: pd.DataFrame, model, top_features: list[str] | None = None) -> pd.DataFrame:
    """Add predictions and probabilities to an input dataframe."""
    result = input_df.copy()
    probabilities = model.predict_proba(input_df)[:, 1] if hasattr(model, "predict_proba") else [0.0] * len(input_df)
    predictions = model.predict(input_df)
    result["fraud_probability"] = probabilities
    result["prediction"] = predictions
    if top_features:
        result["prediction_reason"] = "Top model signals: " + ", ".join(top_features[:3])
    else:
        result["prediction_reason"] = "Based on fitted model patterns."
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run identity theft predictions.")
    parser.add_argument("--input", required=True, help="CSV file to score.")
    parser.add_argument("--model", default="artifacts/best_model.joblib", help="Saved model path.")
    parser.add_argument("--metadata", default="artifacts/model_metadata.json", help="Metadata JSON path.")
    parser.add_argument("--output", default="artifacts/predictions.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    """Load the fitted pipeline and score a CSV file."""
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)
    metadata_path = Path(args.metadata)

    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    input_df = pd.read_csv(input_path)
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    top_features = metadata.get("top_features", [])
    output_df = predict_records(input_df, model, top_features)
    output_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
