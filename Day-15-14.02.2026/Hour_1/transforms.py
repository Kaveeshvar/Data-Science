# Import annotations so type hints can refer to classes defined later.
from __future__ import annotations

# Import dataclass to reduce boilerplate for small stateful classes.
from dataclasses import dataclass

# Import typing for clear, strict type hints.
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Import numpy for numeric arrays and vectorized math.
import numpy as np


# Define a helper type for inputs that can be converted to numpy arrays.
ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


# Define a strict validation helper to convert input to a 2D float64 numpy array.
def _as_2d_float_array(X: ArrayLike, *, name: str = "X") -> np.ndarray:
    # Convert input into a numpy array of float64 for stable math.
    arr = np.asarray(X, dtype=np.float64)
    # If input is 1D, treat it as a single row (n_samples=1).
    if arr.ndim == 1:
        # Add a leading dimension so it becomes shape (1, n_features).
        arr = arr.reshape(1, -1)
    # If input is not 2D after reshaping, it's invalid for tabular transforms.
    if arr.ndim != 2:
        # Raise a clear error explaining expected shape.
        raise ValueError(f"{name} must be 2D (n_samples, n_features); got ndim={arr.ndim}")
    # If array contains NaN or Inf, reject immediately (security + correctness).
    if not np.isfinite(arr).all():
        # Raise a clear error so callers can handle it cleanly.
        raise ValueError(f"{name} contains NaN or Inf")
    # Return validated 2D float64 array.
    return arr


# Define a base protocol-style interface for transforms (minimal, not abstract class).
class Transformer:
    # Fit learns parameters only from training data.
    def fit(self, X: ArrayLike) -> "Transformer":
        # Raise NotImplementedError for base interface.
        raise NotImplementedError
    # Transform applies stored parameters to any data.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Raise NotImplementedError for base interface.
        raise NotImplementedError
    # Serialize transform parameters to a JSON-safe dictionary.
    def to_dict(self) -> Dict[str, Any]:
        # Raise NotImplementedError for base interface.
        raise NotImplementedError
    # Reconstruct transform from a dictionary produced by to_dict().
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transformer":
        # Raise NotImplementedError for base interface.
        raise NotImplementedError


# A safe StandardScaler that fits mean/std on TRAIN only and reuses everywhere.
@dataclass
class StandardScalerSafe(Transformer):
    # Store mean vector once fit is called.
    mu_: Optional[np.ndarray] = None
    # Store std vector once fit is called.
    sigma_: Optional[np.ndarray] = None
    # Store number of features to enforce consistent transform shapes.
    n_features_: Optional[int] = None
    # Use epsilon to avoid division by tiny std values.
    eps: float = 1e-12

    # Fit mean/std using only the provided data (training split).
    def fit(self, X: ArrayLike) -> "StandardScalerSafe":
        # Validate and convert input to strict 2D float64.
        arr = _as_2d_float_array(X, name="X")
        # Save number of columns to enforce later consistency.
        self.n_features_ = int(arr.shape[1])
        # Compute per-feature mean on TRAIN only.
        self.mu_ = arr.mean(axis=0)
        # Compute per-feature std on TRAIN only (ddof=0 is population std).
        self.sigma_ = arr.std(axis=0)
        # Replace near-zero std with 1.0 so constant columns become all zeros.
        self.sigma_ = np.where(self.sigma_ < self.eps, 1.0, self.sigma_)
        # Return self for chaining.
        return self

    # Transform uses stored mu/sigma (train-fitted) on any input split.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Ensure we have been fit before transforming.
        if self.mu_ is None or self.sigma_ is None or self.n_features_ is None:
            # Raise a clear error for misuse.
            raise RuntimeError("StandardScalerSafe is not fit yet")
        # Validate and convert input to strict 2D float64.
        arr = _as_2d_float_array(X, name="X")
        # Enforce same feature count to prevent silent shape bugs.
        if int(arr.shape[1]) != int(self.n_features_):
            # Raise a clear error describing mismatch.
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_}")
        # Standardize using stored parameters (no re-fitting here).
        out = (arr - self.mu_) / self.sigma_
        # Return transformed array.
        return out

    # Convert internal state into JSON-safe dict.
    def to_dict(self) -> Dict[str, Any]:
        # Ensure transform was fit before serializing.
        if self.mu_ is None or self.sigma_ is None or self.n_features_ is None:
            # Raise a clear error for misuse.
            raise RuntimeError("StandardScalerSafe is not fit yet")
        # Build a JSON-safe dict with lists instead of numpy arrays.
        return {
            # Include a type tag so Pipeline can reload correct class.
            "type": "StandardScalerSafe",
            # Save epsilon used.
            "eps": float(self.eps),
            # Save number of features.
            "n_features": int(self.n_features_),
            # Save mu as list for JSON.
            "mu": self.mu_.tolist(),
            # Save sigma as list for JSON.
            "sigma": self.sigma_.tolist(),
        }

    # Recreate scaler from JSON-safe dict.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardScalerSafe":
        # Validate type tag so we don’t load wrong objects.
        if d.get("type") != "StandardScalerSafe":
            # Raise a clear error for mismatch.
            raise ValueError("Dictionary type mismatch for StandardScalerSafe")
        # Create instance with eps from dict (or default).
        obj = cls(eps=float(d.get("eps", 1e-12)))
        # Load and set expected feature count.
        obj.n_features_ = int(d["n_features"])
        # Load mu into float64 numpy array.
        obj.mu_ = np.asarray(d["mu"], dtype=np.float64)
        # Load sigma into float64 numpy array.
        obj.sigma_ = np.asarray(d["sigma"], dtype=np.float64)
        # Return reconstructed object.
        return obj


# A safe Log1p transform that applies log(1+x) to selected nonnegative columns.
@dataclass
class Log1pSafe(Transformer):
    # Columns to transform; if None, transform all columns.
    columns: Optional[List[int]] = None
    # Store fitted feature count for shape enforcement.
    n_features_: Optional[int] = None

    # Fit checks constraints (like non-negativity on training) and stores shape.
    def fit(self, X: ArrayLike) -> "Log1pSafe":
        # Validate and convert input to strict 2D float64.
        arr = _as_2d_float_array(X, name="X")
        # Store number of features for later shape checks.
        self.n_features_ = int(arr.shape[1])
        # Decide which columns to apply log1p to.
        cols = self._resolve_columns(self.n_features_)
        # Check that selected columns are nonnegative on TRAIN (strict rule).
        if (arr[:, cols] < 0.0).any():
            # Raise error because log1p is undefined for x < -1 and risky for negatives.
            raise ValueError("Log1pSafe requires nonnegative values in selected columns")
        # Return self for chaining.
        return self

    # Transform applies log1p to selected columns using stored configuration.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Ensure we have been fit before transforming.
        if self.n_features_ is None:
            # Raise a clear error for misuse.
            raise RuntimeError("Log1pSafe is not fit yet")
        # Validate and convert input to strict 2D float64.
        arr = _as_2d_float_array(X, name="X")
        # Enforce same feature count to avoid silent bugs.
        if int(arr.shape[1]) != int(self.n_features_):
            # Raise a clear mismatch error.
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_}")
        # Resolve columns based on stored n_features_.
        cols = self._resolve_columns(self.n_features_)
        # Reject negative values at inference too (security + correctness).
        if (arr[:, cols] < 0.0).any():
            # Raise error to avoid producing NaNs.
            raise ValueError("Log1pSafe requires nonnegative values in selected columns")
        # Copy array so we don’t mutate caller input.
        out = arr.copy()
        # Apply log1p to selected columns.
        out[:, cols] = np.log1p(out[:, cols])
        # Return transformed array.
        return out

    # Resolve columns list into a validated numpy index array.
    def _resolve_columns(self, n_features: int) -> np.ndarray:
        # If columns not specified, select all features.
        if self.columns is None:
            # Return all column indices.
            return np.arange(n_features, dtype=int)
        # Convert provided columns into numpy int array.
        cols = np.asarray(self.columns, dtype=int)
        # Ensure columns are within bounds.
        if (cols < 0).any() or (cols >= n_features).any():
            # Raise clear error for invalid indices.
            raise ValueError("Log1pSafe columns out of bounds")
        # Ensure no duplicates for predictable behavior.
        if len(np.unique(cols)) != len(cols):
            # Raise clear error for duplicates.
            raise ValueError("Log1pSafe columns must be unique")
        # Return validated columns.
        return cols

    # Serialize Log1pSafe into JSON-safe dict.
    def to_dict(self) -> Dict[str, Any]:
        # Ensure fit occurred before serializing.
        if self.n_features_ is None:
            # Raise error if not fit.
            raise RuntimeError("Log1pSafe is not fit yet")
        # Return JSON-safe dict.
        return {
            # Type tag for reloading.
            "type": "Log1pSafe",
            # Save columns list (or None).
            "columns": self.columns,
            # Save feature count.
            "n_features": int(self.n_features_),
        }

    # Deserialize Log1pSafe from dict.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Log1pSafe":
        # Validate type tag.
        if d.get("type") != "Log1pSafe":
            # Raise clear mismatch error.
            raise ValueError("Dictionary type mismatch for Log1pSafe")
        # Create instance with columns from dict.
        obj = cls(columns=d.get("columns"))
        # Restore feature count.
        obj.n_features_ = int(d["n_features"])
        # Return object.
        return obj


# A minimal Pipeline that chains multiple transforms (fit/transform/to_dict/from_dict).
@dataclass
class Pipeline(Transformer):
    # Hold a list of transforms in order.
    steps: List[Transformer]

    # Fit each step sequentially on the output of the previous step (TRAIN only).
    def fit(self, X: ArrayLike) -> "Pipeline":
        # Validate and convert input once at pipeline entry.
        arr = _as_2d_float_array(X, name="X")
        # Start with current data as the pipeline input.
        cur = arr
        # Fit each transform in order.
        for step in self.steps:
            # Fit step on current data (TRAIN only).
            step.fit(cur)
            # Transform current data to feed into next step.
            cur = step.transform(cur)
        # Return self for chaining.
        return self

    # Transform applies each step sequentially without re-fitting.
    def transform(self, X: ArrayLike) -> np.ndarray:
        # Validate and convert input once at pipeline entry.
        cur = _as_2d_float_array(X, name="X")
        # Apply each step in order.
        for step in self.steps:
            # Transform current data using stored params.
            cur = step.transform(cur)
        # Return final transformed data.
        return cur

    # Serialize entire pipeline.
    def to_dict(self) -> Dict[str, Any]:
        # Serialize each step to a dict.
        steps_dicts = [step.to_dict() for step in self.steps]
        # Return JSON-safe dict.
        return {
            # Type tag for reloading.
            "type": "Pipeline",
            # Store steps list.
            "steps": steps_dicts,
        }

    # Deserialize pipeline from dict.
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pipeline":
        # Validate pipeline type tag.
        if d.get("type") != "Pipeline":
            # Raise mismatch error.
            raise ValueError("Dictionary type mismatch for Pipeline")
        # Load each step based on its type field.
        steps: List[Transformer] = []
        # Iterate through serialized steps.
        for sd in d["steps"]:
            # Read type tag for each step.
            t = sd.get("type")
            # Dispatch to correct class loader.
            if t == "StandardScalerSafe":
                # Load StandardScalerSafe from dict.
                steps.append(StandardScalerSafe.from_dict(sd))
            elif t == "Log1pSafe":
                # Load Log1pSafe from dict.
                steps.append(Log1pSafe.from_dict(sd))
            else:
                # Reject unknown step types to avoid unsafe loads.
                raise ValueError(f"Unknown transform type: {t}")
        # Return reconstructed pipeline.
        return cls(steps=steps)
