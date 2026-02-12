import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


class ModelArtifactError(Exception):
    """Raised when the model artifact is missing fields or is invalid."""
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_key(obj: Dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise ModelArtifactError(f"Missing required field: '{key}'")
    return obj[key]


@dataclass(frozen=True)
class LinearRegressionModel:
    model_version: str
    schema_version: str
    n_features: int
    w: np.ndarray
    b: float
    mu: np.ndarray
    sigma: np.ndarray
    artifact_sha256: str

    @staticmethod
    def load_from_json(path: str) -> "LinearRegressionModel":
        p = Path(path)
        if not p.exists():
            raise ModelArtifactError(f"Artifact not found at: {p.resolve()}")

        sha = _sha256_file(p)

        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelArtifactError(f"Artifact JSON is invalid: {e}") from e

        model_version = str(_require_key(obj, "model_version"))
        schema_version = str(_require_key(obj, "schema_version"))
        n_features = int(_require_key(obj, "n_features"))

        weights = _require_key(obj, "weights")
        bias = float(_require_key(obj, "bias"))
        mu = _require_key(obj, "mu")
        sigma = _require_key(obj, "sigma")

        w = np.asarray(weights, dtype=float).reshape(-1)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(-1)

        # --- Validation ---
        if n_features <= 0:
            raise ModelArtifactError("n_features must be > 0")

        if not (len(w) == len(mu) == len(sigma) == n_features):
            raise ModelArtifactError(
                f"Length mismatch: n_features={n_features}, "
                f"len(w)={len(w)}, len(mu)={len(mu)}, len(sigma)={len(sigma)}"
            )

        if not np.isfinite(w).all():
            raise ModelArtifactError("weights contain NaN/Inf")

        if not np.isfinite(mu).all():
            raise ModelArtifactError("mu contains NaN/Inf")

        if not np.isfinite(sigma).all():
            raise ModelArtifactError("sigma contains NaN/Inf")

        if not np.isfinite(bias):
            raise ModelArtifactError("bias is NaN/Inf")

        if (sigma == 0).any():
            raise ModelArtifactError("sigma contains 0 — would cause divide-by-zero at inference")

        return LinearRegressionModel(
            model_version=model_version,
            schema_version=schema_version,
            n_features=n_features,
            w=w,
            b=bias,
            mu=mu,
            sigma=sigma,
            artifact_sha256=sha,
        )

    def predict(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=float).reshape(-1)

        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")

        if not np.isfinite(x).all():
            raise ValueError("features contain NaN/Inf")

        # IMPORTANT: standardize with training mu/sigma (never recompute)
        x_std = (x - self.mu) / self.sigma

        y = float(x_std @ self.w + self.b)
        if not np.isfinite(y):
            raise ValueError("prediction became NaN/Inf (numerical instability)")

        return y
