"""CLI for inspecting a real fraud dataset before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.preprocessing.data_loader import load_data


TARGET_HINTS = ("fraud", "label", "target", "class", "is_fraud", "account_takeover")
USER_HINTS = ("user", "customer", "account", "member", "client")
TIME_HINTS = ("time", "date", "timestamp", "event")
DEVICE_HINTS = ("device", "browser", "platform", "agent")
LOCATION_HINTS = ("location", "country", "city", "region", "ip")
AMOUNT_HINTS = ("amount", "value", "transaction", "payment", "balance")
LOGIN_HINTS = ("login", "failed", "attempt", "auth")


def pick_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    """Return the first column whose name contains one of the hint tokens."""
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        for lowered_name, original_name in lowered.items():
            if hint in lowered_name:
                return original_name
    return None


def build_summary(df: pd.DataFrame) -> dict:
    """Create a compact dataset profile."""
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in columns if column not in numeric_columns]

    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(columns)),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "suggested_columns": {
            "target_column": pick_column(columns, TARGET_HINTS),
            "user_id_column": pick_column(columns, USER_HINTS),
            "timestamp_column": pick_column(columns, TIME_HINTS),
            "device_column": pick_column(columns, DEVICE_HINTS),
            "location_column": pick_column(columns, LOCATION_HINTS),
            "amount_column": pick_column(columns, AMOUNT_HINTS),
            "failed_login_column": pick_column(columns, LOGIN_HINTS),
        },
    }

    target_column = summary["suggested_columns"]["target_column"]
    if target_column and target_column in df.columns:
        try:
            summary["target_distribution"] = df[target_column].value_counts(dropna=False).to_dict()
        except Exception:
            summary["target_distribution"] = {}
    else:
        summary["target_distribution"] = {}

    return summary


def build_config_stub(summary: dict) -> dict:
    """Create a starter config file from the detected columns."""
    suggestions = summary["suggested_columns"]
    return {
        "target_column": suggestions["target_column"] or "CHANGE_ME",
        "id_columns": [],
        "drop_columns": [],
        "user_id_column": suggestions["user_id_column"],
        "timestamp_column": suggestions["timestamp_column"],
        "device_column": suggestions["device_column"],
        "location_column": suggestions["location_column"],
        "amount_column": suggestions["amount_column"],
        "failed_login_column": suggestions["failed_login_column"],
        "top_k_features": 20,
        "test_size": 0.15,
        "validation_size": 0.15,
        "random_state": 42
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect a fraud dataset and suggest a config.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--output", default="artifacts/dataset_profile.json", help="Path to save the profile JSON.")
    parser.add_argument(
        "--config-output",
        default="config/detected_dataset_config.json",
        help="Path to save the suggested config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Inspect a dataset and save a starter config."""
    args = parse_args()
    df = load_data(args.data)
    summary = build_summary(df)
    config_stub = build_config_stub(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config_output_path = Path(args.config_output)
    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    config_output_path.write_text(json.dumps(config_stub, indent=2), encoding="utf-8")

    print(json.dumps({"profile": str(output_path), "config": str(config_output_path)}, indent=2))


if __name__ == "__main__":
    main()
