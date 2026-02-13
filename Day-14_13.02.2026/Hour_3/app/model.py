# Import json so we can read model artifacts from disk.
import json
# Import os so we can construct file paths reliably.
import os
# Import typing helpers for better type clarity.
from typing import List, Dict, Any
# Import numpy for vector math and validation.
import numpy as np
# Import our stable sigmoid helper.
from app.utils import stable_sigmoid

# Define a lightweight logistic regression model wrapper.
class LogRegModel:
    # Initialize the wrapper with paths to a model artifact and a threshold policy.
    def __init__(self, model_path: str, threshold_path: str) -> None:
        # Store the model artifact path for later reload/debug.
        self.model_path = model_path
        # Store the threshold policy path for later reload/debug.
        self.threshold_path = threshold_path
        # Load the model artifact from disk immediately.
        self._load_model()
        # Load the threshold policy from disk immediately.
        self._load_threshold()

    # Load the model JSON artifact and validate core structure.
    def _load_model(self) -> None:
        # Open the model JSON file for reading.
        with open(self.model_path, "r", encoding="utf-8") as f:
            # Parse the JSON into a Python dictionary.
            data = json.load(f)

        # Read model_version from the artifact for responses and auditing.
        self.model_version = str(data["model_version"])
        # Read schema_version to enforce client/server compatibility.
        self.schema_version = str(data["schema_version"])
        # Read number of expected features.
        self.n_features = int(data["n_features"])

        # Convert weights into a numpy array of float64 for stable computation.
        self.w = np.array(data["weights"], dtype=np.float64)
        # Convert bias into a float for stable computation.
        self.b = float(data["bias"])
        # Convert mu into a numpy array for standardization.
        self.mu = np.array(data["mu"], dtype=np.float64)
        # Convert sigma into a numpy array for standardization.
        self.sigma = np.array(data["sigma"], dtype=np.float64)

        # Enforce that weights length matches n_features.
        if self.w.shape != (self.n_features,):
            # Raise a clear error if artifact is inconsistent.
            raise ValueError("Model artifact invalid: weights length != n_features")

        # Enforce that mu length matches n_features.
        if self.mu.shape != (self.n_features,):
            # Raise a clear error if artifact is inconsistent.
            raise ValueError("Model artifact invalid: mu length != n_features")

        # Enforce that sigma length matches n_features.
        if self.sigma.shape != (self.n_features,):
            # Raise a clear error if artifact is inconsistent.
            raise ValueError("Model artifact invalid: sigma length != n_features")

        # Replace any zero sigma values with 1 to avoid division by zero.
        self.sigma = np.where(self.sigma == 0.0, 1.0, self.sigma)

    # Load the threshold policy JSON artifact and validate core structure.
    def _load_threshold(self) -> None:
        # Open the threshold policy JSON file for reading.
        with open(self.threshold_path, "r", encoding="utf-8") as f:
            # Parse the JSON into a Python dictionary.
            data = json.load(f)

        # Read policy_version for responses and auditing.
        self.policy_version = str(data["policy_version"])
        # Read threshold value and force float type.
        self.threshold = float(data["threshold"])
        # Read metric_used to document why threshold exists.
        self.metric_used = str(data.get("metric_used", "unknown"))

        # Enforce threshold is in [0, 1] to prevent nonsense configuration.
        if not (0.0 <= self.threshold <= 1.0):
            # Raise a clear error if threshold is invalid.
            raise ValueError("Threshold policy invalid: threshold must be in [0, 1]")

    # Public helper to reload threshold policy at runtime (useful for tests/ops).
    def reload_threshold(self) -> None:
        # Reload threshold policy from disk.
        self._load_threshold()

    # Standardize a raw feature vector using stored mu/sigma.
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        # Return standardized features (x - mu) / sigma.
        return (x - self.mu) / self.sigma

    # Predict probability for a single input vector safely.
    def predict_proba(self, features: List[float]) -> float:
        # Convert input list to numpy array for vectorized operations.
        x = np.array(features, dtype=np.float64)
        # Enforce correct shape (n_features,).
        if x.shape != (self.n_features,):
            # Raise a clear error if feature length mismatches.
            raise ValueError("Feature length mismatch")
        # Standardize input features using model artifact stats.
        xs = self._standardize(x)
        # Compute logit z = x·w + b using float64.
        z = float(xs.dot(self.w) + self.b)
        # Convert logit to probability using stable sigmoid.
        p = stable_sigmoid(z)
        # Clip probability into (0,1) bounds for extra numerical safety.
        p = float(min(max(p, 0.0), 1.0))
        # Return probability.
        return p

    # Predict label using server-side threshold policy.
    def predict_label(self, features: List[float]) -> Dict[str, Any]:
        # Compute probability first using the probability function.
        p = self.predict_proba(features)
        # Apply server-side threshold policy to convert score to label.
        label = int(p >= self.threshold)
        # Return both values so API can respond with score + action.
        return {"label": label, "probability": p}
