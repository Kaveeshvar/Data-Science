"""
ml_engine.evaluation
=====================
Model evaluation: compute detailed metrics and generate Plotly charts.

Classification → Confusion Matrix, ROC Curve, Precision-Recall, AUC, class imbalance warning.
Regression     → R², MAE, RMSE, residual distribution, predicted-vs-actual scatter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

log = logging.getLogger("ml_engine.evaluation")


def evaluate_model(
    train_result: Dict[str, Any],
    theme_id: str = "midnight",
) -> Dict[str, Any]:
    """
    Evaluate the trained model and return metrics + Plotly chart HTML strings.

    Parameters
    ----------
    train_result : dict from ``training.train_model``
    theme_id     : dashboard theme for chart styling

    Returns
    -------
    dict with keys:
        metrics     : detailed metric dict
        charts_html : list[str]  (Plotly HTML fragments)
        warnings    : list[str]
    """
    pipeline: Pipeline = train_result["model"]
    problem_type: str = train_result["problem_type"]
    X: pd.DataFrame = train_result["X"]
    y: pd.Series = train_result["y"]
    label_map = train_result.get("label_map")

    # Train/test split for evaluation charts
    stratify = y if problem_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify,
        )
    except ValueError:
        # Fallback without stratify if classes have too few members
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

    y_pred = pipeline.predict(X_test)

    charts: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {"problem_type": problem_type}

    if problem_type == "classification":
        metrics, charts, warnings = _eval_classification(
            pipeline, X_test, y_test, y_pred, label_map, theme_id,
        )
    else:
        metrics, charts, warnings = _eval_regression(
            y_test, y_pred, theme_id,
        )

    return {"metrics": metrics, "charts_html": charts, "warnings": warnings}


# ═══════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def _eval_classification(
    pipeline, X_test, y_test, y_pred, label_map, theme_id,
):
    from plotly.subplots import make_subplots

    metrics: Dict[str, Any] = {"problem_type": "classification"}
    charts: List[str] = []
    warnings: List[str] = []

    classes = sorted(y_test.unique())
    n_classes = len(classes)
    inv_map = {v: k for k, v in label_map.items()} if label_map else {}
    class_labels = [str(inv_map.get(c, c)) for c in classes]

    # Accuracy & F1
    metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
    metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

    # Class imbalance check
    vc = y_test.value_counts(normalize=True)
    if float(vc.min()) < 0.1:
        minority = inv_map.get(vc.idxmin(), vc.idxmin())
        warnings.append(
            f"Class imbalance detected: '{minority}' represents only {vc.min()*100:.1f}% of test data. "
            "Metrics may be misleading — consider SMOTE or class weights."
        )

    # ── Confusion Matrix ─────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=class_labels, y=class_labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig_cm.update_layout(
        title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
        height=400, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    charts.append(fig_cm.to_html(full_html=False, include_plotlyjs=False))

    # ── ROC Curve ────────────────────────────────────────────
    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc_val = float(auc(fpr, tpr))
                metrics["roc_auc"] = round(roc_auc_val, 4)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                    name=f"AUC = {roc_auc_val:.3f}", line=dict(width=2.5)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="gray"), name="Random", showlegend=False))
                fig_roc.update_layout(
                    title=f"ROC Curve (AUC = {roc_auc_val:.3f})",
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    height=400, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                charts.append(fig_roc.to_html(full_html=False, include_plotlyjs=False))

                # ── Precision-Recall Curve ───────────────────
                prec, rec, _ = precision_recall_curve(y_test, y_proba[:, 1])
                pr_auc = float(auc(rec, prec))
                metrics["pr_auc"] = round(pr_auc, 4)

                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                    name=f"PR AUC = {pr_auc:.3f}", line=dict(width=2.5)))
                fig_pr.update_layout(
                    title=f"Precision-Recall Curve (AUC = {pr_auc:.3f})",
                    xaxis_title="Recall", yaxis_title="Precision",
                    height=400, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                charts.append(fig_pr.to_html(full_html=False, include_plotlyjs=False))
            else:
                # Multi-class: compute per-class OVR ROC
                try:
                    roc_auc_val = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
                    metrics["roc_auc"] = round(roc_auc_val, 4)
                except Exception:
                    pass
    except Exception as e:
        log.warning("ROC/PR curve generation failed: %s", e)

    return metrics, charts, warnings


# ═══════════════════════════════════════════════════════════════════
#  REGRESSION
# ═══════════════════════════════════════════════════════════════════

def _eval_regression(y_test, y_pred, theme_id):
    metrics: Dict[str, Any] = {"problem_type": "regression"}
    charts: List[str] = []
    warnings: List[str] = []

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    metrics["r2"] = round(r2, 4)
    metrics["mae"] = round(mae, 4)
    metrics["rmse"] = round(rmse, 4)

    if r2 < 0:
        warnings.append("Negative R² — model performs worse than predicting the mean.")

    # ── Predicted vs Actual Scatter ──────────────────────────
    fig_pa = go.Figure()
    fig_pa.add_trace(go.Scatter(
        x=y_test.values, y=y_pred, mode="markers",
        marker=dict(opacity=0.5, size=5),
        name="Predictions",
    ))
    mn = min(float(y_test.min()), float(min(y_pred)))
    mx = max(float(y_test.max()), float(max(y_pred)))
    fig_pa.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(dash="dash", color="gray"), name="Perfect", showlegend=False,
    ))
    fig_pa.update_layout(
        title=f"Predicted vs Actual (R² = {r2:.3f})",
        xaxis_title="Actual", yaxis_title="Predicted",
        height=400, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    charts.append(fig_pa.to_html(full_html=False, include_plotlyjs=False))

    # ── Residual Distribution ────────────────────────────────
    residuals = y_test.values - y_pred
    fig_res = go.Figure()
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=50, opacity=0.8,
        marker_color="#818cf8",
    ))
    fig_res.update_layout(
        title="Residual Distribution",
        xaxis_title="Residual (Actual − Predicted)", yaxis_title="Count",
        height=400, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    charts.append(fig_res.to_html(full_html=False, include_plotlyjs=False))

    return metrics, charts, warnings
