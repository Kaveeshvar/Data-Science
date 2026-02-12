import json
import numpy as np
from pathlib import Path

def export_linear_regression_artifact(
    out_path: str,
    w: np.ndarray,
    b: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    model_version: str = "1.0",
    schema_version: str = "1.0",
):
    # Ensure correct shapes
    w = np.asarray(w, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)

    # Basic consistency checks (fail fast)
    if not (len(w) == len(mu) == len(sigma)):
        raise ValueError("w, mu, sigma must have the same length (n_features).")

    if not np.isfinite(w).all() or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("Artifact contains NaN/Inf in w/mu/sigma — fix training output.")

    # Prevent division-by-zero at inference: training should have handled this too
    if (sigma == 0).any():
        raise ValueError("sigma contains zeros. Add epsilon during training and re-export.")

    artifact = {
        "model_version": model_version,
        "schema_version": schema_version,
        "n_features": int(len(w)),
        "weights": w.tolist(),
        "bias": float(b),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"✅ Exported model artifact → {out_file.resolve()}")

# ---------- Example usage ----------
if __name__ == "__main__":
    # Replace these with your trained values
    w = np.array([ -0.80655042, -12.10480359,  26.09540471 , 13.0463588 , -18.4264794 ])
    b = 149.06999999999988
    mu = np.array([10, 20, 30, 40, 50])
    sigma = np.array([2, 4, 5, 8, 10])

    export_linear_regression_artifact(
        out_path="model_artifacts/linear_regression_v1.json",
        w=w,
        b=b,
        mu=mu,
        sigma=sigma,
    )
