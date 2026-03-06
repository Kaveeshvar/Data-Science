"""
ml_engine.export
=================
Export trained model artefacts:
  - model.pkl
  - preprocessing_pipeline.pkl
  - model_metadata.json
  - model.onnx  (optional, with skl2onnx)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("ml_engine.export")


def export_model(
    train_result: Dict[str, Any],
    eval_result: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Optional[str]]:
    """
    Save model artefacts to *output_dir*.

    Returns dict mapping artefact name → file path (or None on failure).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Optional[str]] = {}

    pipeline = train_result["model"]
    feature_info = train_result.get("feature_info", {})
    label_map = train_result.get("label_map")

    # ── model.pkl ────────────────────────────────────────────
    model_path = output_dir / "model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        paths["model_pkl"] = str(model_path)
        log.info("Saved model.pkl → %s", model_path)
    except Exception as e:
        log.error("Failed to save model.pkl: %s", e)
        paths["model_pkl"] = None

    # ── preprocessing_pipeline.pkl ───────────────────────────
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            pp_path = output_dir / "preprocessing_pipeline.pkl"
            with open(pp_path, "wb") as f:
                pickle.dump(preprocessor, f)
            paths["preprocessing_pkl"] = str(pp_path)
            log.info("Saved preprocessing_pipeline.pkl → %s", pp_path)
    except Exception as e:
        log.error("Failed to save preprocessing_pipeline.pkl: %s", e)
        paths["preprocessing_pkl"] = None

    # ── model_metadata.json ──────────────────────────────────
    metadata = _build_metadata(train_result, eval_result, feature_info, label_map)
    meta_path = output_dir / "model_metadata.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        paths["metadata_json"] = str(meta_path)
        log.info("Saved model_metadata.json → %s", meta_path)
    except Exception as e:
        log.error("Failed to save model_metadata.json: %s", e)
        paths["metadata_json"] = None

    # ── model.onnx  (optional) ───────────────────────────────
    onnx_path = _try_export_onnx(pipeline, train_result, output_dir)
    paths["onnx"] = onnx_path

    return paths


def _build_metadata(
    train_result: Dict[str, Any],
    eval_result: Dict[str, Any],
    feature_info: dict,
    label_map: Optional[dict],
) -> dict:
    inv_map = {v: k for k, v in label_map.items()} if label_map else None
    return {
        "target_column": train_result.get("target_column"),
        "problem_type": train_result.get("problem_type"),
        "model_type": train_result.get("model_type"),
        "features": {
            "numeric": feature_info.get("numeric", []),
            "categorical": feature_info.get("categorical", []),
        },
        "label_map": label_map,
        "inverse_label_map": inv_map,
        "training_metrics": train_result.get("metrics", {}),
        "evaluation_metrics": eval_result.get("metrics", {}),
        "cv_results": train_result.get("cv_results", []),
        "feature_importance": dict(list(train_result.get("feature_importance", {}).items())[:30]),
        "n_training_rows": train_result.get("preprocessing_info", {}).get("n_rows"),
        "sampled": train_result.get("preprocessing_info", {}).get("sampled", False),
    }


def _try_export_onnx(
    pipeline,
    train_result: Dict[str, Any],
    output_dir: Path,
) -> Optional[str]:
    """
    Attempt ONNX export. Returns path on success, None on failure.
    
    Gracefully handles model types that may not be ONNX-compatible by falling back
    to model.pkl usage (still works great for Python predictions).
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        X: pd.DataFrame = train_result["X"]
        feature_info = train_result.get("feature_info", {})
        num_cols = feature_info.get("numeric", [])
        cat_cols = feature_info.get("categorical", [])
        model_type = train_result.get("model_type", "unknown")

        # Build initial types matching actual column names and types
        initial_types = []
        for col in num_cols:
            initial_types.append((col, FloatTensorType([None, 1])))
        for col in cat_cols:
            initial_types.append((col, StringTensorType([None, 1])))

        if not initial_types:
            log.info("No features for ONNX export — skipping.")
            return None

        onnx_model = convert_sklearn(pipeline, initial_types=initial_types, target_opset=15)
        onnx_path = output_dir / "model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info("Saved model.onnx → %s (%s)", onnx_path, model_type)
        return str(onnx_path)

    except ImportError:
        log.info("skl2onnx not installed — ONNX export disabled. Using model.pkl instead.")
        return None
    except Exception as e:
        model_type = train_result.get("model_type", "unknown")
        log.warning(
            "ONNX export failed for %s: %s. Model will use .pkl format instead (Python predictions work fine).",
            model_type, 
            str(e)[:80]
        )
        return None


def get_inference_metadata(
    train_result: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build metadata dict for embedding into HTML for in-browser prediction.
    Includes feature descriptions, defaults, categories, etc.
    """
    feature_info = train_result.get("feature_info", {})
    target_col = train_result.get("target_column")
    label_map = train_result.get("label_map")
    inv_map = {v: k for k, v in label_map.items()} if label_map else None

    features_meta: list = []

    for col in feature_info.get("numeric", []):
        s = df[col].dropna()
        features_meta.append({
            "name": col,
            "type": "numeric",
            "default": round(float(s.median()), 4) if len(s) > 0 else 0,
            "min": round(float(s.min()), 4) if len(s) > 0 else 0,
            "max": round(float(s.max()), 4) if len(s) > 0 else 1,
            "mean": round(float(s.mean()), 4) if len(s) > 0 else 0,
        })

    for col in feature_info.get("categorical", []):
        s = df[col].dropna()
        cats = s.value_counts().head(20).index.tolist()
        mode_val = str(s.mode().iloc[0]) if len(s) > 0 else (cats[0] if cats else "")
        features_meta.append({
            "name": col,
            "type": "categorical",
            "categories": [str(c) for c in cats],
            "default": mode_val,
        })

    return {
        "target": target_col,
        "problemType": train_result.get("problem_type"),
        "modelType": train_result.get("model_type"),
        "features": features_meta,
        "labelMap": inv_map,
        "numericFeatures": feature_info.get("numeric", []),
        "categoricalFeatures": feature_info.get("categorical", []),
    }
