"""
ml_engine.anomaly
==================
Enhanced anomaly detection using IsolationForest.

Supplements (does NOT replace) the existing IQR/Z-score outlier analysis.

Produces:
  - Anomaly score distribution chart
  - Top-10 anomalous rows table
  - Anomaly explanation text
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("ml_engine.anomaly")

_MAX_ROWS_ANOMALY = 50_000


def run_anomaly_detection(
    df: pd.DataFrame,
    numeric_cols: List[str],
    theme_id: str = "midnight",
) -> Optional[Dict[str, Any]]:
    """
    Run IsolationForest-based anomaly detection.

    Returns dict with:
        n_anomalies      : int
        pct_anomalies    : float
        top_anomalies    : DataFrame (top-10 most anomalous rows)
        score_stats      : dict
        charts_html      : list[str]
        explanation       : str
    """
    if len(numeric_cols) < 2:
        log.info("Anomaly detection skipped — fewer than 2 numeric columns.")
        return None

    usable = [c for c in numeric_cols if df[c].isna().mean() < 0.5]
    if len(usable) < 2:
        return None

    work = df.copy()
    X = work[usable].copy()

    # Sample if large
    sampled = False
    if len(X) > _MAX_ROWS_ANOMALY:
        idx = np.random.RandomState(42).choice(len(X), _MAX_ROWS_ANOMALY, replace=False)
        X = X.iloc[idx].copy()
        work = work.iloc[idx].copy()
        sampled = True

    # Fill NaN with median for IsolationForest
    X = X.fillna(X.median())

    if len(X) < 20:
        return None

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── IsolationForest ─────────────────────────────────────
    iso = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X_scaled)         # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X_scaled)  # lower = more anomalous

    work = work.reset_index(drop=True)
    work["_anomaly_label"] = preds
    work["_anomaly_score"] = scores

    n_anomalies = int((preds == -1).sum())
    pct_anomalies = round(n_anomalies / len(work) * 100, 2)

    log.info("Anomalies: %d / %d (%.1f%%)", n_anomalies, len(work), pct_anomalies)

    # ── Top-10 most anomalous rows ──────────────────────────
    top10 = work.nsmallest(10, "_anomaly_score")
    # keep only original columns + score
    display_cols = [c for c in df.columns if c in top10.columns][:8]
    top10_display = top10[display_cols + ["_anomaly_score"]].copy()
    top10_display = top10_display.rename(columns={"_anomaly_score": "Anomaly Score"})

    # ── Charts ──────────────────────────────────────────────
    charts: List[str] = []

    # 1) Score distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=scores, nbinsx=60, opacity=0.8,
        marker_color="#818cf8", name="All",
    ))
    if n_anomalies > 0:
        anomaly_scores = scores[preds == -1]
        fig_dist.add_trace(go.Histogram(
            x=anomaly_scores, nbinsx=30, opacity=0.7,
            marker_color="#f87171", name="Anomalies",
        ))
    fig_dist.update_layout(
        title=f"Anomaly Score Distribution ({n_anomalies} anomalies detected)",
        xaxis_title="Anomaly Score (lower = more anomalous)",
        yaxis_title="Count",
        barmode="overlay",
        height=400, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    charts.append(fig_dist.to_html(full_html=False, include_plotlyjs=False))

    # 2) If ≥2 usable cols, scatter plot of top 2 features coloured by anomaly
    if len(usable) >= 2:
        c1, c2 = usable[0], usable[1]
        plot_df = work[[c1, c2, "_anomaly_label"]].copy()
        plot_df["Type"] = plot_df["_anomaly_label"].map({1: "Normal", -1: "Anomaly"})
        fig_scat = go.Figure()
        for label_val, name, color, size, opacity in [
            (1, "Normal", "#818cf8", 4, 0.3),
            (-1, "Anomaly", "#f87171", 8, 0.9),
        ]:
            mask = plot_df["_anomaly_label"] == label_val
            fig_scat.add_trace(go.Scatter(
                x=plot_df.loc[mask, c1], y=plot_df.loc[mask, c2],
                mode="markers", name=name,
                marker=dict(color=color, size=size, opacity=opacity),
            ))
        fig_scat.update_layout(
            title=f"Anomalies: {c1} vs {c2}",
            xaxis_title=c1, yaxis_title=c2,
            height=400, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        charts.append(fig_scat.to_html(full_html=False, include_plotlyjs=False))

    # ── Explanation ─────────────────────────────────────────
    explanation = (
        f"IsolationForest identified {n_anomalies:,} anomalous records ({pct_anomalies:.1f}% of "
        f"{'sampled ' if sampled else ''}{len(work):,} rows). "
        f"These are data points that deviate significantly from the majority pattern across "
        f"{len(usable)} numeric features. "
        f"Review the top anomalous rows below for potential data quality issues, fraud, "
        f"or genuinely rare events."
    )

    return {
        "n_anomalies": n_anomalies,
        "pct_anomalies": pct_anomalies,
        "top_anomalies": top10_display,
        "score_stats": {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
        },
        "charts_html": charts,
        "explanation": explanation,
    }
