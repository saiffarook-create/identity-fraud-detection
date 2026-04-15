"""Raw data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV input files are supported in this project.")
    dataframe = pd.read_csv(csv_path)
    unnamed_ratio = sum(str(column).startswith("Unnamed:") for column in dataframe.columns) / max(len(dataframe.columns), 1)
    first_column = str(dataframe.columns[0]).lstrip("\ufeff") if len(dataframe.columns) else ""

    # Some Kaggle CSVs include a comment-style banner before the real header.
    if unnamed_ratio > 0.5 or first_column.startswith("#"):
        dataframe = pd.read_csv(csv_path, comment="#")
    return dataframe


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Validate that the selected target column exists."""
    if target_column not in df.columns:
        available = ", ".join(df.columns[:20])
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns include: {available}"
        )
