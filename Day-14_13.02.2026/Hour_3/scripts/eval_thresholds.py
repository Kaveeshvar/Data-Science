# Import os so we can handle file paths.
import os
# Import json so we can read/write artifacts.
import json
# Import csv so we can write sweep outputs.
import csv
# Import datetime for metadata.
from datetime import datetime, timezone
# Import numpy for math.
import numpy as np

# Compute ROC curve points given labels and scores.
def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    # Sort by descending score for ROC construction.
    order = np.argsort(-y_score)
    # Apply ordering to true labels.
    y = y_true[order]
    # Apply ordering to scores.
    s = y_score[order]
    # Count positives.
    P = np.sum(y == 1)
    # Count negatives.
    N = np.sum(y == 0)
    # Initialize true positives.
    tp = 0
    # Initialize false positives.
    fp = 0
    # Track points list.
    points = []
    # Track last score for threshold changes.
    last_score = None
    # Iterate through sorted scores.
    for yi, si in zip(y, s):
        # If we hit a new threshold, record the ROC point.
        if last_score is None or si != last_score:
            # Compute TPR safely.
            tpr = tp / P if P > 0 else 0.0
            # Compute FPR safely.
            fpr = fp / N if N > 0 else 0.0
            # Append point.
            points.append((fpr, tpr, float(si)))
            # Update last score.
            last_score = si
        # Update counts based on label.
        if yi == 1:
            tp += 1
        else:
            fp += 1
    # Add final point at end.
    tpr = tp / P if P > 0 else 0.0
    fpr = fp / N if N > 0 else 0.0
    points.append((fpr, tpr, -np.inf))
    # Return list of points.
    return points

# Compute AUC via trapezoidal rule on ROC points.
def auc_from_points(points):
    # Sort points by FPR ascending.
    pts = sorted(points, key=lambda x: x[0])
    # Initialize area.
    area = 0.0
    # Iterate adjacent pairs.
    for (x1, y1, _), (x2, y2, _) in zip(pts[:-1], pts[1:]):
        # Add trapezoid area.
        area += (x2 - x1) * (y1 + y2) / 2.0
    # Return area.
    return float(area)

# Compute precision, recall, f1 for a threshold.
def prf_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    # Predict labels based on threshold.
    y_pred = (y_score >= thr).astype(int)
    # True positives count.
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    # False positives count.
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    # False negatives count.
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    # Precision computation.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall computation.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1 computation.
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    # Return tuple.
    return float(precision), float(recall), float(f1)

# Main function to run evaluation.
def main():
    # Compute project root directory.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Build artifact paths.
    model_path = os.path.join(root, "model_artifacts", "logreg_v1.json")
    policy_path = os.path.join(root, "model_artifacts", "threshold_policy_v1.json")
    # Output files.
    roc_csv_path = os.path.join(root, "model_artifacts", "roc.csv")
    sweep_csv_path = os.path.join(root, "model_artifacts", "threshold_sweep.csv")

    # Load model artifact.
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    # Extract model params.
    w = np.array(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    mu = np.array(model["mu"], dtype=np.float64)
    sigma = np.array(model["sigma"], dtype=np.float64)
    sigma = np.where(sigma == 0.0, 1.0, sigma)

    # --- Demo validation set (replace with real data later) ---
    # Create a deterministic synthetic dataset for offline evaluation.
    rng = np.random.default_rng(123)
    X = rng.normal(size=(500, len(w)))
    y = (rng.random(500) > 0.7).astype(int)

    # Standardize using artifact stats.
    Xs = (X - mu) / sigma
    # Compute logits.
    z = Xs.dot(w) + b
    # Compute probabilities using stable logic vectorized.
    y_score = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    # Compute ROC points and AUC.
    points = roc_curve_points(y, y_score)
    auc = auc_from_points(points)

    # Write ROC points to CSV.
    with open(roc_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fpr, tpr, thr in points:
            writer.writerow([fpr, tpr, thr])

    # Sweep thresholds and pick best F1.
    best_thr = 0.5
    best_f1 = -1.0

    # Write sweep file.
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "precision", "recall", "f1"])
        for thr in np.linspace(0.05, 0.95, 19):
            precision, recall, f1 = prf_at_threshold(y, y_score, float(thr))
            writer.writerow([float(thr), precision, recall, f1])
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

    # Build policy artifact.
    policy = {
        "policy_version": "1.0",
        "threshold": best_thr,
        "metric_used": "f1",
        "validation_auc": auc,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d")
    }

    # Write threshold policy JSON.
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    # Print summary for CLI visibility.
    print(f"AUC={auc:.4f} | best_thr={best_thr:.2f} | best_f1={best_f1:.4f}")

# Run main when executed as a script.
if __name__ == "__main__":
    main()
