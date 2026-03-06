"""
ml_engine.preprocessing
========================
Automatic preprocessing pipeline builder for sklearn.

Builds a ``ColumnTransformer`` that handles:
  - Numeric: median imputation + StandardScaler
  - Categorical: mode imputation + OneHotEncoding
  - Auto-drops ID-like columns (high uniqueness ratio)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger("ml_engine.preprocessing")

# Columns with uniqueness ratio above this threshold are considered ID-like
_ID_UNIQUENESS_THRESHOLD = 0.9
_MAX_TRAINING_ROWS = 50_000


def _is_id_column(series: pd.Series, n_rows: int) -> bool:
    """Return True if the column looks like an ID / primary-key."""
    name_lower = series.name.lower().strip().replace(" ", "_")
    if name_lower in ("id", "index", "row_id", "row_number", "pk", "key",
                      "unique_id", "serial", "serial_no", "sr_no", "sno"):
        return True
    if series.nunique(dropna=True) / max(n_rows, 1) > _ID_UNIQUENESS_THRESHOLD:
        if series.dtype == "object":
            return True
    return False


def identify_feature_columns(
    df: pd.DataFrame,
    target_column: str | None,
) -> Dict[str, List[str]]:
    """
    Classify columns into numeric, categorical, or drop lists.

    Returns dict with keys: numeric, categorical, drop.
    """
    numeric: List[str] = []
    categorical: List[str] = []
    drop: List[str] = []
    n_rows = len(df)

    for col in df.columns:
        if col == target_column:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            drop.append(col)
            continue
        if _is_id_column(df[col], n_rows):
            drop.append(col)
            log.info("  Dropping ID-like column: %s", col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            # Only keep categoricals with reasonable cardinality
            if df[col].nunique(dropna=True) <= 50:
                categorical.append(col)
            else:
                drop.append(col)
                log.info("  Dropping high-cardinality categorical: %s (%d unique)",
                         col, df[col].nunique())
        else:
            drop.append(col)

    return {"numeric": numeric, "categorical": categorical, "drop": drop}


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str | None,
) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    """
    Build a sklearn ColumnTransformer for *df*.

    Returns
    -------
    (preprocessor, feature_info)
        preprocessor  : fitted-ready ColumnTransformer
        feature_info  : dict with numeric / categorical / drop lists
    """
    feature_info = identify_feature_columns(df, target_column)
    num_cols = feature_info["numeric"]
    cat_cols = feature_info["categorical"]

    transformers = []

    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    log.info("Preprocessing: %d numeric, %d categorical, %d dropped",
             len(num_cols), len(cat_cols), len(feature_info["drop"]))

    return preprocessor, feature_info


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, Dict[str, Any]]:
    """
    Full data preparation: sample, split features/target, build preprocessor.

    Returns (X, y, preprocessor, info_dict).
    """
    # Sample if too large
    sampled = False
    if len(df) > _MAX_TRAINING_ROWS:
        df = df.sample(_MAX_TRAINING_ROWS, random_state=42).reset_index(drop=True)
        sampled = True
        log.info("Sampled %d rows for training", _MAX_TRAINING_ROWS)

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    y = df[target_column].copy()

    # Encode target for classification if it's string
    label_map = None
    if problem_type == "classification" and y.dtype == "object":
        labels = sorted(y.unique())
        label_map = {lab: i for i, lab in enumerate(labels)}
        y = y.map(label_map)

    preprocessor, feature_info = build_preprocessing_pipeline(df, target_column)

    X = df[feature_info["numeric"] + feature_info["categorical"]]

    info = {
        "sampled": sampled,
        "n_rows": len(df),
        "feature_info": feature_info,
        "label_map": label_map,
    }

    return X, y, preprocessor, info
