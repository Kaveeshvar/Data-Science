# Enable forward references for type hints in older Python versions.
from __future__ import annotations

# Import json-safe typing primitives.
from typing import Any, Dict, List, Optional, Sequence, Union

# Import numpy for numerical operations.
import numpy as np


# Define accepted array-like inputs for convenience.
ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


# Convert input into a strict 2D float64 numpy array with strong validation.
def _as_2d_float_array(X: ArrayLike, *, name: str = "X") -> np.ndarray:
    # Convert to float64 numpy array for stable computation.
    arr = np.asarray(X, dtype=np.float64)
    # If input is 1D, treat it as one sample with many features.
    if arr.ndim == 1:
        # Reshape to (1, n_features).
        arr = arr.reshape(1, -1)
    # Reject non-2D inputs to prevent silent shape bugs.
    if arr.ndim != 2:
        # Raise a clear error with the observed dimension.
        raise ValueError(f"{name} must be 2D (n_samples, n_features); got ndim={arr.ndim}")
    # Reject NaN/Inf immediately (security + correctness).
    if not np.isfinite(arr).all():
        # Raise a clear error so callers can handle cleanly.
        raise ValueError(f"{name} contains NaN or Inf")
    # Return validated array.
    return arr


# Minimal transformer interface.
class Transformer:
    # Fit learns parameters from training data only.
    def fit(self, X: ArrayLike) -> "Transformer":
        # Base interface requires implementation.
        raise NotImplementedError

    # Transform applies stored parameters to any data.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Base interface requires implementation.
        raise NotImplementedError

    # Serialize internal parameters to JSON-safe dict.
    def to_dict(self) -> Dict[str, Any]:
        # Base interface requires implementation.
        raise NotImplementedError

    # Deserialize from dict produced by to_dict.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transformer":
        # Base interface requires implementation.
        raise NotImplementedError


# StandardScaler: (x - mu) / sigma with sigma=0 handled safely.
class StandardScalerSafe(Transformer):
    # Initialize scaler with epsilon to handle constant columns.
    def __init__(self, eps: float = 1e-12) -> None:
        # Store epsilon for stability.
        self.eps = float(eps)
        # Mean vector learned from train.
        self.mu_: Optional[np.ndarray] = None
        # Std vector learned from train.
        self.sigma_: Optional[np.ndarray] = None
        # Feature count to enforce shape consistency.
        self.n_features_: Optional[int] = None

    # Fit mean/std on training data only.
    def fit(self, X: ArrayLike) -> "StandardScalerSafe":
        # Validate and convert input.
        arr = _as_2d_float_array(X, name="X")
        # Store feature count.
        self.n_features_ = int(arr.shape[1])
        # Compute train mean per column.
        self.mu_ = arr.mean(axis=0)
        # Compute train std per column.
        self.sigma_ = arr.std(axis=0)
        # Replace near-zero std with 1.0 so constant columns become 0 after scaling.
        self.sigma_ = np.where(self.sigma_ < self.eps, 1.0, self.sigma_)
        # Return self for chaining.
        return self

    # Transform using stored mu/sigma.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Ensure we have fit parameters.
        if self.mu_ is None or self.sigma_ is None or self.n_features_ is None:
            # Raise a clear error for misuse.
            raise RuntimeError("StandardScalerSafe is not fit yet")
        # Validate and convert input.
        arr = _as_2d_float_array(X, name="X")
        # Enforce consistent feature count.
        if int(arr.shape[1]) != int(self.n_features_):
            # Raise a clear mismatch error.
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_}")
        # Apply scaling using train-fitted parameters only.
        out = (arr - self.mu_) / self.sigma_
        # Return transformed array.
        return out

    # Serialize scaler parameters to JSON-safe dict.
    def to_dict(self) -> Dict[str, Any]:
        # Ensure fit occurred.
        if self.mu_ is None or self.sigma_ is None or self.n_features_ is None:
            # Raise a clear error for misuse.
            raise RuntimeError("StandardScalerSafe is not fit yet")
        # Return JSON-safe dictionary.
        return {
            # Type tag for safe loading.
            "type": "StandardScalerSafe",
            # Save epsilon.
            "eps": float(self.eps),
            # Save number of features.
            "n_features": int(self.n_features_),
            # Save mean as list.
            "mu": self.mu_.tolist(),
            # Save std as list.
            "sigma": self.sigma_.tolist(),
        }

    # Deserialize scaler from dict.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardScalerSafe":
        # Validate type tag.
        if d.get("type") != "StandardScalerSafe":
            # Raise mismatch error.
            raise ValueError("Dictionary type mismatch for StandardScalerSafe")
        # Create scaler with stored epsilon.
        obj = cls(eps=float(d.get("eps", 1e-12)))
        # Restore feature count.
        obj.n_features_ = int(d["n_features"])
        # Restore mean vector.
        obj.mu_ = np.asarray(d["mu"], dtype=np.float64)
        # Restore std vector.
        obj.sigma_ = np.asarray(d["sigma"], dtype=np.float64)
        # Return reconstructed scaler.
        return obj


# RobustScaler: (x - median) / IQR with IQR=0 handled safely.
class RobustScalerSafe(Transformer):
    # Initialize with quantile bounds and epsilon for stability.
    def __init__(self, q_low: float = 25.0, q_high: float = 75.0, eps: float = 1e-12) -> None:
        # Store lower quantile.
        self.q_low = float(q_low)
        # Store upper quantile.
        self.q_high = float(q_high)
        # Store epsilon for stability.
        self.eps = float(eps)
        # Store median vector.
        self.median_: Optional[np.ndarray] = None
        # Store IQR vector.
        self.iqr_: Optional[np.ndarray] = None
        # Store feature count.
        self.n_features_: Optional[int] = None

    # Fit median and IQR on training data only.
    def fit(self, X: ArrayLike) -> "RobustScalerSafe":
        # Validate and convert input.
        arr = _as_2d_float_array(X, name="X")
        # Store feature count.
        self.n_features_ = int(arr.shape[1])
        # Compute median per column.
        self.median_ = np.median(arr, axis=0)
        # Compute q_low per column.
        ql = np.percentile(arr, self.q_low, axis=0)
        # Compute q_high per column.
        qh = np.percentile(arr, self.q_high, axis=0)
        # Compute IQR.
        self.iqr_ = qh - ql
        # Replace near-zero IQR with 1.0 to stabilize constant-ish columns.
        self.iqr_ = np.where(self.iqr_ < self.eps, 1.0, self.iqr_)
        # Return self.
        return self

    # Transform using stored median and IQR.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Ensure fit occurred.
        if self.median_ is None or self.iqr_ is None or self.n_features_ is None:
            # Raise clear error.
            raise RuntimeError("RobustScalerSafe is not fit yet")
        # Validate and convert.
        arr = _as_2d_float_array(X, name="X")
        # Enforce shape.
        if int(arr.shape[1]) != int(self.n_features_):
            # Raise mismatch.
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_}")
        # Apply robust scaling.
        out = (arr - self.median_) / self.iqr_
        # Return output.
        return out

    # Serialize robust scaler.
    def to_dict(self) -> Dict[str, Any]:
        # Ensure fit occurred.
        if self.median_ is None or self.iqr_ is None or self.n_features_ is None:
            # Raise clear error.
            raise RuntimeError("RobustScalerSafe is not fit yet")
        # Return JSON-safe dict.
        return {
            # Type tag.
            "type": "RobustScalerSafe",
            # Quantile config.
            "q_low": float(self.q_low),
            # Quantile config.
            "q_high": float(self.q_high),
            # Epsilon config.
            "eps": float(self.eps),
            # Feature count.
            "n_features": int(self.n_features_),
            # Median list.
            "median": self.median_.tolist(),
            # IQR list.
            "iqr": self.iqr_.tolist(),
        }

    # Deserialize robust scaler.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobustScalerSafe":
        # Validate type.
        if d.get("type") != "RobustScalerSafe":
            # Raise mismatch.
            raise ValueError("Dictionary type mismatch for RobustScalerSafe")
        # Create object with stored config.
        obj = cls(q_low=float(d["q_low"]), q_high=float(d["q_high"]), eps=float(d.get("eps", 1e-12)))
        # Restore feature count.
        obj.n_features_ = int(d["n_features"])
        # Restore median.
        obj.median_ = np.asarray(d["median"], dtype=np.float64)
        # Restore IQR.
        obj.iqr_ = np.asarray(d["iqr"], dtype=np.float64)
        # Return object.
        return obj


# MinMax scaler: (x - min) / (max - min) with optional clipping + clip counter.
class MinMaxSafe(Transformer):
    # Initialize with clipping option.
    def __init__(self, clip: bool = False, eps: float = 1e-12) -> None:
        # Store whether to clip transformed values into [0, 1].
        self.clip = bool(clip)
        # Store epsilon to avoid division by zero.
        self.eps = float(eps)
        # Store min per column.
        self.min_: Optional[np.ndarray] = None
        # Store max per column.
        self.max_: Optional[np.ndarray] = None
        # Store scale per column (max-min adjusted).
        self.scale_: Optional[np.ndarray] = None
        # Store feature count.
        self.n_features_: Optional[int] = None
        # Count how many values were clipped (runtime signal).
        self.clip_count_: int = 0

    # Fit min/max on training data only.
    def fit(self, X: ArrayLike) -> "MinMaxSafe":
        # Validate and convert.
        arr = _as_2d_float_array(X, name="X")
        # Store feature count.
        self.n_features_ = int(arr.shape[1])
        # Compute min per column on train.
        self.min_ = arr.min(axis=0)
        # Compute max per column on train.
        self.max_ = arr.max(axis=0)
        # Compute scale.
        scale = self.max_ - self.min_
        # Replace near-zero scale with 1.0 to stabilize constant columns.
        self.scale_ = np.where(scale < self.eps, 1.0, scale)
        # Reset clip counter at fit time.
        self.clip_count_ = 0
        # Return self.
        return self

    # Transform using stored min/max.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Ensure fit occurred.
        if self.min_ is None or self.max_ is None or self.scale_ is None or self.n_features_ is None:
            # Raise clear error.
            raise RuntimeError("MinMaxSafe is not fit yet")
        # Validate and convert.
        arr = _as_2d_float_array(X, name="X")
        # Enforce consistent shape.
        if int(arr.shape[1]) != int(self.n_features_):
            # Raise mismatch.
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_}")
        # Apply min-max scaling.
        out = (arr - self.min_) / self.scale_
        # If clipping is enabled, clip and count clipped entries.
        if self.clip:
            # Count values below 0.
            below = out < 0.0
            # Count values above 1.
            above = out > 1.0
            # Increase clip counter by number of out-of-range values.
            self.clip_count_ += int(np.sum(below) + np.sum(above))
            # Clip into [0, 1].
            out = np.clip(out, 0.0, 1.0)
        # Return transformed.
        return out

    # Serialize minmax scaler.
    def to_dict(self) -> Dict[str, Any]:
        # Ensure fit occurred.
        if self.min_ is None or self.max_ is None or self.scale_ is None or self.n_features_ is None:
            # Raise error.
            raise RuntimeError("MinMaxSafe is not fit yet")
        # Return JSON-safe dict.
        return {
            # Type tag.
            "type": "MinMaxSafe",
            # Clipping config.
            "clip": bool(self.clip),
            # Epsilon config.
            "eps": float(self.eps),
            # Feature count.
            "n_features": int(self.n_features_),
            # Min list.
            "min": self.min_.tolist(),
            # Max list.
            "max": self.max_.tolist(),
            # Scale list.
            "scale": self.scale_.tolist(),
        }

    # Deserialize minmax scaler.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MinMaxSafe":
        # Validate type tag.
        if d.get("type") != "MinMaxSafe":
            # Raise mismatch.
            raise ValueError("Dictionary type mismatch for MinMaxSafe")
        # Create object with config.
        obj = cls(clip=bool(d["clip"]), eps=float(d.get("eps", 1e-12)))
        # Restore feature count.
        obj.n_features_ = int(d["n_features"])
        # Restore min.
        obj.min_ = np.asarray(d["min"], dtype=np.float64)
        # Restore max.
        obj.max_ = np.asarray(d["max"], dtype=np.float64)
        # Restore scale.
        obj.scale_ = np.asarray(d["scale"], dtype=np.float64)
        # Initialize clip counter.
        obj.clip_count_ = 0
        # Return object.
        return obj


# Log1p transform: log(1 + x) on selected columns, with strict negativity policy.
class Log1pSafe(Transformer):
    # Initialize with selected columns and optional shifting.
    def __init__(self, columns: Optional[List[int]] = None, allow_shift: bool = False, shift_value: float = 0.0) -> None:
        # Store which columns to transform.
        self.columns = columns
        # Store whether shifting negatives is allowed.
        self.allow_shift = bool(allow_shift)
        # Store shift amount (only used if allow_shift=True).
        self.shift_value = float(shift_value)
        # Store feature count after fit.
        self.n_features_: Optional[int] = None

    # Fit validates configuration and checks train data compatibility.
    def fit(self, X: ArrayLike) -> "Log1pSafe":
        # Validate and convert input.
        arr = _as_2d_float_array(X, name="X")
        # Store feature count.
        self.n_features_ = int(arr.shape[1])
        # Resolve which columns to use.
        cols = self._resolve_columns(self.n_features_)
        # If shift is not allowed, reject any negative values in selected cols.
        if not self.allow_shift:
            # Check for negatives.
            if (arr[:, cols] < 0.0).any():
                # Reject because log1p for negatives is not safe/meaningful by default.
                raise ValueError("Log1pSafe rejects negatives unless allow_shift=True")
        # If shift is allowed, validate shift makes values >= 0.
        else:
            # Ensure shift_value is not negative (nonsense).
            if self.shift_value < 0.0:
                # Reject bad config.
                raise ValueError("shift_value must be >= 0")
            # Check whether shifting would still leave negatives.
            if (arr[:, cols] + self.shift_value < 0.0).any():
                # Reject because even after shift we'd have invalid domain.
                raise ValueError("allow_shift=True but shift_value is insufficient to remove negatives")
        # Return self.
        return self

    # Transform applies log1p with chosen negativity policy.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Ensure fit occurred.
        if self.n_features_ is None:
            # Raise error.
            raise RuntimeError("Log1pSafe is not fit yet")
        # Validate and convert.
        arr = _as_2d_float_array(X, name="X")
        # Enforce shape.
        if int(arr.shape[1]) != int(self.n_features_):
            # Raise mismatch.
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_}")
        # Resolve columns.
        cols = self._resolve_columns(self.n_features_)
        # Copy array to avoid mutating caller input.
        out = arr.copy()
        # If shift is not allowed, reject negatives at transform time too.
        if not self.allow_shift:
            # Reject if negatives present.
            if (out[:, cols] < 0.0).any():
                raise ValueError("Log1pSafe rejects negatives unless allow_shift=True")
            # Apply log1p to selected columns.
            out[:, cols] = np.log1p(out[:, cols])
        # If shift is allowed, apply shift then log1p.
        else:
            # Ensure shifted values remain valid.
            if (out[:, cols] + self.shift_value < 0.0).any():
                raise ValueError("Shifted values still negative; increase shift_value")
            # Apply shift and log1p.
            out[:, cols] = np.log1p(out[:, cols] + self.shift_value)
        # Return output.
        return out

    # Resolve selected columns into a validated numpy index array.
    def _resolve_columns(self, n_features: int) -> np.ndarray:
        # If no columns specified, transform all columns.
        if self.columns is None:
            # Return all indices.
            return np.arange(n_features, dtype=int)
        # Convert list to numpy int array.
        cols = np.asarray(self.columns, dtype=int)
        # Validate bounds.
        if (cols < 0).any() or (cols >= n_features).any():
            raise ValueError("Log1pSafe columns out of bounds")
        # Validate uniqueness.
        if len(np.unique(cols)) != len(cols):
            raise ValueError("Log1pSafe columns must be unique")
        # Return validated cols.
        return cols

    # Serialize log1p transform.
    def to_dict(self) -> Dict[str, Any]:
        # Ensure fit occurred.
        if self.n_features_ is None:
            raise RuntimeError("Log1pSafe is not fit yet")
        # Return JSON-safe dict.
        return {
            # Type tag.
            "type": "Log1pSafe",
            # Columns config.
            "columns": self.columns,
            # Shift policy.
            "allow_shift": bool(self.allow_shift),
            # Shift value.
            "shift_value": float(self.shift_value),
            # Feature count.
            "n_features": int(self.n_features_),
        }

    # Deserialize log1p transform.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Log1pSafe":
        # Validate type.
        if d.get("type") != "Log1pSafe":
            raise ValueError("Dictionary type mismatch for Log1pSafe")
        # Create instance with config.
        obj = cls(
            columns=d.get("columns"),
            allow_shift=bool(d.get("allow_shift", False)),
            shift_value=float(d.get("shift_value", 0.0)),
        )
        # Restore feature count.
        obj.n_features_ = int(d["n_features"])
        # Return object.
        return obj


# Pipeline chains multiple transforms and supports JSON artifact roundtrip.
class Pipeline(Transformer):
    # Initialize pipeline with ordered steps.
    def __init__(self, steps: List[Transformer]) -> None:
        # Store steps.
        self.steps = steps

    # Fit sequentially (fit and transform) on training data only.
    def fit(self, X: ArrayLike) -> "Pipeline":
        # Validate input once at pipeline entry.
        cur = _as_2d_float_array(X, name="X")
        # Fit each step and pass transformed output to next.
        for step in self.steps:
            # Fit step on current train representation.
            step.fit(cur)
            # Transform current representation for next step.
            cur = step.transform(cur)
        # Return self.
        return self

    # Transform sequentially using stored parameters.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Validate input once at pipeline entry.
        cur = _as_2d_float_array(X, name="X")
        # Apply each step in order.
        for step in self.steps:
            # Transform with stored parameters only.
            cur = step.transform(cur)
        # Return final transformed output.
        return cur

    # Serialize pipeline and all steps.
    def to_dict(self) -> Dict[str, Any]:
        # Serialize each step.
        steps_dicts = [step.to_dict() for step in self.steps]
        # Return JSON-safe dict.
        return {
            # Root type tag.
            "type": "Pipeline",
            # Steps list.
            "steps": steps_dicts,
        }

    # Deserialize pipeline from dict.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pipeline":
        # Validate root type.
        if d.get("type") != "Pipeline":
            raise ValueError("Dictionary type mismatch for Pipeline")
        # Build steps list.
        steps: List[Transformer] = []
        # Deserialize each step based on its type field.
        for sd in d["steps"]:
            # Read step type tag.
            t = sd.get("type")
            # Dispatch to correct loader.
            if t == "StandardScalerSafe":
                steps.append(StandardScalerSafe.from_dict(sd))
            elif t == "RobustScalerSafe":
                steps.append(RobustScalerSafe.from_dict(sd))
            elif t == "MinMaxSafe":
                steps.append(MinMaxSafe.from_dict(sd))
            elif t == "Log1pSafe":
                steps.append(Log1pSafe.from_dict(sd))
            else:
                # Reject unknown types for safety.
                raise ValueError(f"Unknown transform type: {t}")
        # Return reconstructed pipeline.
        return cls(steps=steps)
