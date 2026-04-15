"""Beginner-friendly feature engineering for tabular fraud data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert a target series to binary 0/1 labels."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).clip(lower=0, upper=1)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "legitimate": "0",
                "normal": "0",
                "benign": "0",
                "false": "0",
                "no": "0",
                "fraud": "1",
                "fraudulent": "1",
                "identity_fraud": "1",
                "account_takeover": "1",
                "attack": "1",
                "true": "1",
                "yes": "1",
            }
        )
    )
    return normalized.isin(
        {"1", "fraud", "fraudulent", "identity_fraud", "account_takeover", "attack"}
    ).astype(int)


def engineer_features(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Add a few safe, explainable features when source columns exist."""
    engineered = df.copy()

    if config.amount_column and config.amount_column in engineered.columns:
        amount_col = config.amount_column
        engineered[f"{amount_col}_is_high_value"] = (
            engineered[amount_col].fillna(0) > engineered[amount_col].fillna(0).median()
        ).astype(int)

    if config.failed_login_column and config.failed_login_column in engineered.columns:
        fail_col = config.failed_login_column
        engineered[f"{fail_col}_flag"] = (engineered[fail_col].fillna(0) > 3).astype(int)

    if config.user_id_column and config.user_id_column in engineered.columns:
        user_col = config.user_id_column
        engineered["user_event_count"] = engineered.groupby(user_col)[user_col].transform("count")

        if (
            config.amount_column
            and config.amount_column in engineered.columns
            and pd.api.types.is_numeric_dtype(engineered[config.amount_column])
        ):
            user_mean = engineered.groupby(user_col)[config.amount_column].transform("mean")
            engineered["amount_vs_user_mean"] = (
                engineered[config.amount_column].fillna(0) - user_mean.fillna(0)
            )

    if (
        config.device_column
        and config.location_column
        and config.device_column in engineered.columns
        and config.location_column in engineered.columns
    ):
        combo = (
            engineered[config.device_column].astype(str).fillna("unknown")
            + "::"
            + engineered[config.location_column].astype(str).fillna("unknown")
        )
        combo_counts = combo.value_counts(dropna=False)
        engineered["rare_device_location_flag"] = combo.map(combo_counts).fillna(0).lt(3).astype(int)

    if config.timestamp_column and config.timestamp_column in engineered.columns:
        parsed = pd.to_datetime(
            engineered[config.timestamp_column],
            errors="coerce",
            dayfirst=True,
        )
        if parsed.notna().any():
            engineered["event_hour"] = parsed.dt.hour.fillna(-1).astype(int)
            engineered["event_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(int)
            engineered["is_night_event"] = engineered["event_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    for column in engineered.columns:
        if pd.api.types.is_numeric_dtype(engineered[column]):
            engineered[column] = engineered[column].replace([np.inf, -np.inf], np.nan)

    return engineered
