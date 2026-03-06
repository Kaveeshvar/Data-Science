"""
ml_engine.training
===================
Auto baseline model training with cross-validation.

Classification: LogisticRegression & RandomForestClassifier (best by ROC-AUC).
Regression:     LinearRegression & RandomForestRegressor (best by R²).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from .preprocessing import prepare_data

log = logging.getLogger("ml_engine.training")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def _get_feature_importance(pipeline: Pipeline, feature_names: list) -> Dict[str, float]:
    """Extract feature importance from the final estimator in the pipeline."""
    model = pipeline.named_steps["model"]

    # Get transformed feature names
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        transformed_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        transformed_names = [f"feature_{i}" for i in range(len(feature_names))]

    importances: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        for name, imp in zip(transformed_names, raw):
            importances[name] = round(float(imp), 6)
    elif hasattr(model, "coef_"):
        coef = np.atleast_2d(model.coef_)
        # For binary classification coef_ is (1, n_features); take abs mean across classes
        mean_abs = np.abs(coef).mean(axis=0)
        for name, imp in zip(transformed_names, mean_abs):
            importances[name] = round(float(imp), 6)

    # Sort desc
    importances = dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True))
    return importances


def _signed_importances(pipeline: Pipeline) -> Dict[str, float]:
    """Get signed coefficient values (for linear models). Returns empty for tree models."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        transformed_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        return {}

    if hasattr(model, "coef_"):
        coef = np.atleast_2d(model.coef_)
        # average across classes for multi-class
        mean_coef = coef.mean(axis=0)
        result = {}
        for name, val in zip(transformed_names, mean_coef):
            result[name] = round(float(val), 6)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    return {}


def train_model(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> Dict[str, Any]:
    """
    Train baseline models and select the best via cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str
    problem_type : "classification" | "regression"

    Returns
    -------
    dict with keys:
        model            : trained sklearn Pipeline
        metrics          : dict of metric_name → value
        feature_importance : dict of feature_name → importance
        signed_importances : dict of feature_name → signed coefficient (linear only)
        model_type       : str  (name of best model)
        problem_type     : str
        target_column    : str
        label_map        : dict | None
        feature_info     : dict
        cv_results       : list of per-model results
        preprocessing_info : dict
    """
    log.info("Preparing data for training (%s → '%s')", problem_type, target_column)
    X, y, preprocessor, prep_info = prepare_data(df, target_column, problem_type)

    if len(X) < 20:
        raise ValueError(f"Not enough rows to train ({len(X)}). Need at least 20.")

    feature_names = list(X.columns)

    # ── Define candidate models ──────────────────────────────
    if problem_type == "classification":
        candidates = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=42,
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1,
            ),
        }
        scoring = "roc_auc_ovr_weighted"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        candidates = {
            "LinearRegression": LinearRegression(n_jobs=-1),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1,
            ),
        }
        scoring = "r2"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Cross-validate each candidate ────────────────────────
    cv_results = []
    best_name = None
    best_score = -np.inf

    for name, estimator in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            cv_results.append({
                "name": name,
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
                "scores": [round(float(s), 4) for s in scores],
            })
            log.info("  %s — CV %s: %.4f (±%.4f)", name, scoring, mean_score, std_score)
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
        except Exception as e:
            log.warning("  %s — CV failed: %s", name, e)
            cv_results.append({"name": name, "mean_score": None, "error": str(e)})

    if best_name is None:
        raise RuntimeError("All candidate models failed during cross-validation.")

    # ── Retrain best model on full data ──────────────────────
    best_estimator = candidates[best_name]
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_estimator),
    ])
    final_pipeline.fit(X, y)
    log.info("Best model: %s (CV score: %.4f)", best_name, best_score)

    # ── Feature importance ───────────────────────────────────
    importance = _get_feature_importance(final_pipeline, feature_names)
    signed = _signed_importances(final_pipeline)

    # ── Collect metrics summary ──────────────────────────────
    metrics: Dict[str, Any] = {
        "cv_metric": scoring,
        "cv_score": round(best_score, 4),
    }

    return {
        "model": final_pipeline,
        "metrics": metrics,
        "feature_importance": importance,
        "signed_importances": signed,
        "model_type": best_name,
        "problem_type": problem_type,
        "target_column": target_column,
        "label_map": prep_info.get("label_map"),
        "feature_info": prep_info.get("feature_info", {}),
        "cv_results": cv_results,
        "preprocessing_info": prep_info,
        "X": X,
        "y": y,
    }
