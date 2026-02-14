# Import typing for clear signatures.
from typing import Any, Dict, List, Tuple

# Import numpy for stats computations.
import numpy as np

# Import strict converter from transforms to reuse validation logic.
from transforms import _as_2d_float_array, ArrayLike


# Compute summary statistics per feature for quick sanity and drift baselines.
def summary_stats(X: ArrayLike, *, quantiles: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> Dict[str, Any]:
    # Validate and convert input to strict 2D float array.
    arr = _as_2d_float_array(X, name="X")
    # Compute per-feature mean.
    mean = arr.mean(axis=0)
    # Compute per-feature std.
    std = arr.std(axis=0)
    # Compute per-feature min.
    minv = arr.min(axis=0)
    # Compute per-feature max.
    maxv = arr.max(axis=0)
    # Compute requested quantiles per feature.
    qs = np.quantile(arr, quantiles, axis=0)
    # Return JSON-safe dict with lists.
    return {
        # Save feature count.
        "n_features": int(arr.shape[1]),
        # Save mean list.
        "mean": mean.tolist(),
        # Save std list.
        "std": std.tolist(),
        # Save min list.
        "min": minv.tolist(),
        # Save max list.
        "max": maxv.tolist(),
        # Save quantiles config.
        "quantiles": quantiles,
        # Save quantiles matrix as list-of-lists.
        "quantile_values": qs.tolist(),
    }


# Compare train stats to production stats and flag suspicious drift.
def drift_report(train_stats: Dict[str, Any], prod_stats: Dict[str, Any], *, z_threshold: float = 3.0) -> Dict[str, Any]:
    # Ensure feature counts match.
    if int(train_stats["n_features"]) != int(prod_stats["n_features"]):
        # Raise error for incompatible comparisons.
        raise ValueError("train_stats and prod_stats n_features mismatch")
    # Convert lists back into numpy for math.
    train_mean = np.asarray(train_stats["mean"], dtype=np.float64)
    # Convert lists back into numpy for math.
    train_std = np.asarray(train_stats["std"], dtype=np.float64)
    # Convert lists back into numpy for math.
    prod_mean = np.asarray(prod_stats["mean"], dtype=np.float64)
    # Convert lists back into numpy for math.
    prod_std = np.asarray(prod_stats["std"], dtype=np.float64)
    # Avoid division by zero in z-score by replacing tiny stds with 1.
    safe_std = np.where(train_std < 1e-12, 1.0, train_std)
    # Compute mean shift in units of train std.
    mean_z = (prod_mean - train_mean) / safe_std
    # Compute std ratio as a drift indicator.
    std_ratio = prod_std / np.where(train_std < 1e-12, 1.0, train_std)
    # Flag features with mean shift beyond threshold.
    mean_shift_flags = (np.abs(mean_z) >= float(z_threshold))
    # Return report dict.
    return {
        # Provide per-feature mean z-scores.
        "mean_z": mean_z.tolist(),
        # Provide per-feature std ratios.
        "std_ratio": std_ratio.tolist(),
        # Flags for suspicious mean drift.
        "mean_shift_flags": mean_shift_flags.tolist(),
        # Provide threshold used.
        "z_threshold": float(z_threshold),
    }


# Compute Population Stability Index (PSI) for one numeric feature.
def psi_1d(train: np.ndarray, prod: np.ndarray, *, bins: int = 10, eps: float = 1e-6) -> float:
    # Ensure arrays are float64.
    tr = np.asarray(train, dtype=np.float64)
    # Ensure arrays are float64.
    pr = np.asarray(prod, dtype=np.float64)
    # Reject NaN/Inf defensively.
    if not np.isfinite(tr).all() or not np.isfinite(pr).all():
        raise ValueError("psi input contains NaN/Inf")
    # Compute bin edges using train quantiles (stable baseline bins).
    edges = np.quantile(tr, np.linspace(0.0, 1.0, bins + 1))
    # Ensure edges are strictly increasing by adding tiny jitter where needed.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    # Compute train histogram counts.
    tr_counts, _ = np.histogram(tr, bins=edges)
    # Compute prod histogram counts.
    pr_counts, _ = np.histogram(pr, bins=edges)
    # Convert to proportions.
    tr_p = tr_counts / max(tr_counts.sum(), 1)
    # Convert to proportions.
    pr_p = pr_counts / max(pr_counts.sum(), 1)
    # Smooth zeros to avoid log(0).
    tr_p = np.clip(tr_p, eps, 1.0)
    # Smooth zeros to avoid log(0).
    pr_p = np.clip(pr_p, eps, 1.0)
    # Compute PSI sum.
    value = float(np.sum((pr_p - tr_p) * np.log(pr_p / tr_p)))
    # Return PSI value.
    return value


# Compute PSI per feature for two datasets.
def psi(train_X: ArrayLike, prod_X: ArrayLike, *, bins: int = 10) -> List[float]:
    # Validate and convert both inputs.
    tr = _as_2d_float_array(train_X, name="train_X")
    # Validate and convert both inputs.
    pr = _as_2d_float_array(prod_X, name="prod_X")
    # Enforce feature count match.
    if tr.shape[1] != pr.shape[1]:
        raise ValueError("train_X and prod_X feature mismatch")
    # Compute PSI per feature.
    values = []
    # Iterate each column.
    for j in range(tr.shape[1]):
        values.append(psi_1d(tr[:, j], pr[:, j], bins=bins))
    # Return list.
    return values
