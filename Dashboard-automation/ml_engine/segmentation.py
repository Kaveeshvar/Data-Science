"""
ml_engine.segmentation
=======================
Customer / data segmentation via KMeans clustering.

Triggered when:
  - No target is detected  OR
  - Dataset has ≥ 5 numeric columns

Finds optimal K (2–6) via silhouette score, then generates:
  - Cluster size bar chart
  - Cluster feature profile table
  - Radar plot per cluster
  - Auto-generated cluster explanations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("ml_engine.segmentation")

_MAX_ROWS_SEGMENT = 30_000
_K_RANGE = range(2, 7)  # 2–6
_MAX_RADAR_FEATURES = 8


def should_run_segmentation(
    df: pd.DataFrame,
    target_detected: bool,
    numeric_cols: List[str],
) -> bool:
    """Decide whether segmentation should run."""
    if not target_detected and len(numeric_cols) >= 3:
        return True
    if len(numeric_cols) >= 5:
        return True
    return False


def run_segmentation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    theme_id: str = "midnight",
) -> Optional[Dict[str, Any]]:
    """
    Run KMeans segmentation.

    Returns dict with:
        n_clusters      : int
        silhouette       : float
        cluster_sizes    : dict
        profiles         : DataFrame (cluster × feature means)
        charts_html      : list[str]
        explanations     : list[str]
    """
    if len(numeric_cols) < 2:
        log.info("Segmentation skipped — fewer than 2 numeric columns.")
        return None

    # Select usable columns (drop those with too many NaNs)
    usable = [c for c in numeric_cols if df[c].isna().mean() < 0.5]
    if len(usable) < 2:
        return None

    work = df[usable].dropna().copy()
    if len(work) < 20:
        log.info("Segmentation skipped — too few rows after dropping NaN (%d).", len(work))
        return None

    if len(work) > _MAX_ROWS_SEGMENT:
        work = work.sample(_MAX_ROWS_SEGMENT, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work)

    # ── Find optimal K ──────────────────────────────────────
    best_k = 2
    best_sil = -1.0
    sil_scores: Dict[int, float] = {}

    for k in _K_RANGE:
        if k >= len(work):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled))))
        sil_scores[k] = round(sil, 4)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    log.info("Optimal K=%d (silhouette=%.3f)", best_k, best_sil)

    # ── Final fit ───────────────────────────────────────────
    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42, max_iter=300)
    work["_cluster"] = km_final.fit_predict(X_scaled)

    cluster_sizes = work["_cluster"].value_counts().sort_index().to_dict()
    profiles = work.groupby("_cluster")[usable].mean().round(3)

    # ── Charts ──────────────────────────────────────────────
    charts: List[str] = []

    # 1) Cluster size bar
    fig_size = go.Figure(go.Bar(
        x=[f"Cluster {i}" for i in sorted(cluster_sizes.keys())],
        y=[cluster_sizes[i] for i in sorted(cluster_sizes.keys())],
        marker_color=["#818cf8", "#22d3ee", "#f472b6", "#facc15", "#34d399", "#fb923c"][:best_k],
        text=[cluster_sizes[i] for i in sorted(cluster_sizes.keys())],
        textposition="outside",
    ))
    fig_size.update_layout(
        title="Cluster Sizes", height=350,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Count",
    )
    charts.append(fig_size.to_html(full_html=False, include_plotlyjs=False))

    # 2) Radar plot per cluster
    radar_feats = usable[:_MAX_RADAR_FEATURES]
    if len(radar_feats) >= 3:
        # Normalise profiles to 0-1 for radar
        prof_norm = profiles[radar_feats].copy()
        for c in radar_feats:
            rang = prof_norm[c].max() - prof_norm[c].min()
            if rang > 0:
                prof_norm[c] = (prof_norm[c] - prof_norm[c].min()) / rang
            else:
                prof_norm[c] = 0.5

        fig_radar = go.Figure()
        colors = ["#818cf8", "#22d3ee", "#f472b6", "#facc15", "#34d399", "#fb923c"]
        for idx, row in prof_norm.iterrows():
            vals = row[radar_feats].tolist() + [row[radar_feats[0]]]  # close polygon
            cats = [c[:20] for c in radar_feats] + [radar_feats[0][:20]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                name=f"Cluster {idx}",
                line_color=colors[int(idx) % len(colors)],
                opacity=0.65,
            ))
        fig_radar.update_layout(
            title="Cluster Feature Profiles (Radar)",
            height=450, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,.08)"),
            ),
        )
        charts.append(fig_radar.to_html(full_html=False, include_plotlyjs=False))

    # ── Explanations ────────────────────────────────────────
    explanations = _generate_cluster_explanations(profiles, cluster_sizes, usable)

    return {
        "n_clusters": best_k,
        "silhouette": round(best_sil, 4),
        "silhouette_scores": sil_scores,
        "cluster_sizes": cluster_sizes,
        "profiles": profiles,
        "charts_html": charts,
        "explanations": explanations,
    }


def _generate_cluster_explanations(
    profiles: pd.DataFrame,
    sizes: Dict[int, int],
    features: List[str],
) -> List[str]:
    """Generate a plain-English explanation per cluster."""
    total = sum(sizes.values())
    explanations = []
    global_means = profiles.mean()

    for idx, row in profiles.iterrows():
        pct = sizes[idx] / total * 100
        # find top 2 features where this cluster is notably above/below average
        diffs = ((row - global_means) / global_means.replace(0, 1)).sort_values()
        low_feat = diffs.index[0] if len(diffs) > 0 else "N/A"
        low_val = diffs.iloc[0] if len(diffs) > 0 else 0
        high_feat = diffs.index[-1] if len(diffs) > 0 else "N/A"
        high_val = diffs.iloc[-1] if len(diffs) > 0 else 0

        desc = (
            f"Cluster {idx} ({sizes[idx]:,} records, {pct:.1f}% of data): "
            f"Highest relative '{high_feat}' (+{high_val*100:.0f}% vs avg)"
        )
        if low_val < -0.05:
            desc += f", lowest relative '{low_feat}' ({low_val*100:.0f}% vs avg)"
        desc += "."
        explanations.append(desc)

    return explanations
