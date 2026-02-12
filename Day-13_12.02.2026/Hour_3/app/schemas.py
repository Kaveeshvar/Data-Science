# app/schemas.py  # Pydantic schemas + validation for the inference API.

from typing import List  # Import List type for the feature vector.
import math  # Import math for isfinite checks.
from pydantic import BaseModel, Field, field_validator  # Import Pydantic base types and validators.
from app import config  # Import config to access N_FEATURES and bounds.


class PredictRequest(BaseModel):  # Define the request body schema for /predict.
    schema_version: str = Field(..., description="Client schema version for request contract")  # Required version string.
    features: List[float] = Field(..., description="Raw feature vector")  # Required list of numeric features.

    @field_validator("features")  # Register a validator for the 'features' field.
    @classmethod  # Make it a classmethod per Pydantic v2 conventions.
    def validate_features(cls, v: List[float]) -> List[float]:  # Validator function for features list.
        if v is None:  # If features is missing (should not happen because Field(...) requires it).
            raise ValueError("features is required")  # Raise validation error.

        if len(v) == 0:  # If list is empty, reject it.
            raise ValueError("features must not be empty")  # Raise validation error.

        for i, x in enumerate(v):  # Iterate each feature with its index.
            xf = float(x)  # Cast to float to handle ints cleanly.
            if not math.isfinite(xf):  # Reject NaN or Inf values.
                raise ValueError(f"features[{i}] must be finite (no NaN/Inf)")  # Raise validation error.
            if abs(xf) >= float(config.FEATURE_ABS_MAX):  # Enforce bounded input constraint to block extreme abuse.
                raise ValueError(f"features[{i}] magnitude too large")  # Do not echo the value back (avoid info leaks).

        if config.N_FEATURES is None:  # If startup did not set N_FEATURES, server is not ready.
            raise ValueError("Server not ready: model n_features not loaded yet")  # Raise validation error.

        if len(v) != int(config.N_FEATURES):  # Enforce strict feature length contract.
            raise ValueError(f"Expected {config.N_FEATURES} features, got {len(v)}")  # Raise validation error.

        return v  # Return validated list.


class PredictResponse(BaseModel):  # Define the response schema for /predict.
    prediction: float  # Prediction output.
    model_version: str  # Model version used for prediction.
