"""Unit tests using synthetic data."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.features.engineering import normalize_target
from src.inference.predict import predict_records
from src.preprocessing.data_loader import load_data
from src.preprocessing.pipeline import build_preprocessor, prepare_features_and_target, prepare_splits
from src.training.train import train_model

TEST_TMP_ROOT = Path("tests/.tmp")


def make_synthetic_dataset(rows: int = 60) -> pd.DataFrame:
    """Create a small synthetic dataset for repeatable tests."""
    data = []
    for index in range(rows):
        label = 1 if index % 5 == 0 else 0
        data.append(
            {
                "user_id": f"user_{index % 10}",
                "event_timestamp": f"2026-01-{(index % 28) + 1:02d} 0{index % 9}:00:00",
                "device_type": "mobile" if index % 3 == 0 else "desktop",
                "location": "Cairo" if index % 4 == 0 else "Alex",
                "transaction_amount": 900 + index * 13 if label else 50 + index * 2,
                "failed_login_attempts": 5 if label else index % 2,
                "session_duration": 2 if label else 12 + (index % 4),
                "is_fraud": label,
            }
        )
    return pd.DataFrame(data)


class PipelineTests(unittest.TestCase):
    """Core functional tests for data prep, training, and inference."""

    def setUp(self) -> None:
        self.df = make_synthetic_dataset()
        self.config = ProjectConfig()
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def test_load_data_reads_csv(self) -> None:
        csv_path = TEST_TMP_ROOT / "sample.csv"
        self.df.to_csv(csv_path, index=False)
        loaded = load_data(str(csv_path))
        self.assertEqual(len(loaded), len(self.df))

    def test_normalize_target_is_binary(self) -> None:
        series = pd.Series(["fraud", "legitimate", "account_takeover", "normal"])
        normalized = normalize_target(series)
        self.assertEqual(normalized.tolist(), [1, 0, 1, 0])

    def test_prepare_splits_preserves_all_rows(self) -> None:
        train_df, val_df, test_df = prepare_splits(self.df, self.config)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.df))
        self.assertGreater(len(train_df), len(test_df))

    def test_build_preprocessor_returns_transformer(self) -> None:
        X, _ = prepare_features_and_target(self.df, self.config)
        transformer = build_preprocessor(X, top_k_features=10)
        self.assertIsNotNone(transformer)

    def test_train_model_runs(self) -> None:
        X, y = prepare_features_and_target(self.df, self.config)
        model = train_model("logistic_regression", X, y, top_k_features=10)
        predictions = model.predict(X.head(5))
        self.assertEqual(len(predictions), 5)

    def test_predict_records_formats_output(self) -> None:
        X, y = prepare_features_and_target(self.df, self.config)
        model = train_model("random_forest", X, y, top_k_features=10)
        output = predict_records(X.head(10), model, ["transaction_amount", "failed_login_attempts"])
        self.assertIn("fraud_probability", output.columns)
        self.assertIn("prediction", output.columns)
        self.assertIn("prediction_reason", output.columns)

    def test_config_json_roundtrip(self) -> None:
        config_path = TEST_TMP_ROOT / "config.json"
        config_path.write_text(json.dumps({"target_column": "is_fraud"}), encoding="utf-8")
        loaded = ProjectConfig.from_json(config_path)
        self.assertEqual(loaded.target_column, "is_fraud")


if __name__ == "__main__":
    unittest.main()
