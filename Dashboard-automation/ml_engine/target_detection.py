"""
ml_engine.target_detection
===========================
Automatic target column detection for supervised learning.

Exposes ``detect_target(df)`` which returns a dict with:
  - target_column: str | None
  - problem_type: "classification" | "regression" | None
  - confidence: float (0–1)
  - reasoning: str
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("ml_engine.target_detection")

# ── Priority target names (case-insensitive, stripped of _ and spaces) ──

_PRIORITY_NAMES = {
    "target", "label", "response", "churn", "default", "outcome",
    "survived", "diagnosis", "fraud", "spam", "sentiment", "class",
    "y", "result", "status", "approved", "price", "salary",
    "revenue", "amount",
}

_CLASSIFICATION_BIAS_NAMES = {
    "target", "label", "churn", "default", "outcome", "survived",
    "diagnosis", "fraud", "spam", "sentiment", "class", "approved",
    "status", "result",
}

_REGRESSION_BIAS_NAMES = {
    "price", "salary", "revenue", "amount",
}


def _normalise(name: str) -> str:
    return name.lower().strip().replace("_", "").replace(" ", "")


def detect_target(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect the most likely target column in *df*.

    Returns
    -------
    dict
        target_column : str | None
        problem_type  : "classification" | "regression" | None
        confidence    : float (0.0 – 1.0)
        reasoning     : str
    """
    candidates: list[Dict[str, Any]] = []

    for col in df.columns:
        norm = _normalise(col)
        nunique = int(df[col].nunique(dropna=True))
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_cat = df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype)
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        missing_pct = float(df[col].isna().mean())

        # Skip datetime columns — never a target
        if is_datetime:
            continue

        # Skip ID-like columns (high-uniqueness string cols)
        if is_cat and nunique > max(0.5 * len(df), 100):
            continue

        # Skip columns with very high missing rate
        if missing_pct > 0.5:
            continue

        score = 0.0
        problem_type: Optional[str] = None
        reasons: list[str] = []

        # ── Name-based scoring ──────────────────────────────
        if norm in _PRIORITY_NAMES:
            score += 0.50
            reasons.append(f"column name '{col}' matches known target pattern")
            if norm in _CLASSIFICATION_BIAS_NAMES:
                problem_type = "classification"
            elif norm in _REGRESSION_BIAS_NAMES:
                problem_type = "regression"

        # ── Cardinality-based scoring ───────────────────────
        if nunique == 2:
            score += 0.25
            if problem_type is None:
                problem_type = "classification"
            reasons.append("binary column (2 unique values)")
        elif is_cat and nunique <= 10:
            score += 0.18
            if problem_type is None:
                problem_type = "classification"
            reasons.append(f"low-cardinality categorical ({nunique} unique)")
        elif is_numeric and nunique <= 10:
            score += 0.15
            if problem_type is None:
                problem_type = "classification"
            reasons.append(f"low-cardinality numeric ({nunique} unique)")
        elif is_numeric and nunique > 20:
            score += 0.05
            if problem_type is None:
                problem_type = "regression"
            reasons.append(f"high-cardinality numeric ({nunique} unique)")

        # ── Position bias: last column often is the target ──
        if col == df.columns[-1]:
            score += 0.10
            reasons.append("last column (commonly used as target)")

        # penalise if _first_ column (often an index)
        if col == df.columns[0]:
            score -= 0.10

        if score > 0:
            candidates.append({
                "column": col,
                "score": round(min(score, 1.0), 3),
                "problem_type": problem_type,
                "reasons": reasons,
            })

    if not candidates:
        log.info("No target candidate found.")
        return {
            "target_column": None,
            "problem_type": None,
            "confidence": 0.0,
            "reasoning": "No column matched target-detection heuristics.",
        }

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    # If best score is too low, still return but with low confidence
    confidence = best["score"]

    log.info(
        "Target detected: '%s' (%s, confidence=%.2f)",
        best["column"], best["problem_type"], confidence,
    )

    return {
        "target_column": best["column"],
        "problem_type": best["problem_type"],
        "confidence": confidence,
        "reasoning": "; ".join(best["reasons"]),
        "_candidates": candidates[:5],  # expose top-5 for debugging
    }
