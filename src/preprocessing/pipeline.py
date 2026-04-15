"""Preprocessing, splitting, and feature selection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ProjectConfig
from src.features.engineering import engineer_features, normalize_target


@dataclass(slots=True)
class DatasetBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def prepare_features_and_target(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the dataframe, derive features, and return X/y."""
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' does not exist.")

    working = df.drop_duplicates().copy()
    y = normalize_target(working[config.target_column])

    drop_cols = set(config.drop_columns + config.id_columns + [config.target_column])
    existing_drop_cols = [col for col in drop_cols if col in working.columns]
    X = working.drop(columns=existing_drop_cols)
    X = engineer_features(X, config)
    return X, y


def prepare_splits(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test splits with target preserved."""
    train_df, temp_df = train_test_split(
        df,
        test_size=config.test_size + config.validation_size,
        stratify=normalize_target(df[config.target_column]),
        random_state=config.random_state,
    )

    relative_validation_size = config.validation_size / (config.test_size + config.validation_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_validation_size,
        stratify=normalize_target(temp_df[config.target_column]),
        random_state=config.random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame, top_k_features: int = 20):
    """Build a reusable preprocessing block for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    base_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    if top_k_features <= 0:
        return base_transformer

    selector_k = "all" if top_k_features >= max(len(X.columns), 1) else top_k_features

    return Pipeline(
        steps=[
            ("preprocessor", base_transformer),
            ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        ]
    )


def prepare_dataset_bundle(df: pd.DataFrame, config: ProjectConfig) -> DatasetBundle:
    """Generate clean X/y splits."""
    train_df, val_df, test_df = prepare_splits(df, config)
    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)
    X_test, y_test = prepare_features_and_target(test_df, config)
    return DatasetBundle(X_train, X_val, X_test, y_train, y_val, y_test)
