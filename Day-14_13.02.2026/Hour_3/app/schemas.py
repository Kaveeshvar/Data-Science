# Import typing so we can define list types cleanly.
from typing import List
# Import BaseModel for request/response validation.
from pydantic import BaseModel, Field, field_validator

# Request schema for predict_proba endpoint.
class PredictProbaRequest(BaseModel):
    # Schema version string used to enforce compatibility.
    schema_version: str = Field(..., description="Client schema version")
    # Feature vector used for inference.
    features: List[float] = Field(..., description="Raw feature vector")

    # Validate that features are finite (no NaN/Inf) at schema level.
    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: List[float]) -> List[float]:
        # Import math for isnan/isinf checks.
        import math
        # Iterate through each feature.
        for x in v:
            # Reject NaN explicitly.
            if math.isnan(x):
                raise ValueError("features contains NaN")
            # Reject Inf explicitly.
            if math.isinf(x):
                raise ValueError("features contains Inf")
        # Return validated list.
        return v

# Response schema for predict_proba endpoint.
class PredictProbaResponse(BaseModel):
    # Probability score for the positive class.
    probability: float
    # Model version for traceability.
    model_version: str

# Request schema for predict endpoint (classification).
class PredictRequest(BaseModel):
    # Schema version string used to enforce compatibility.
    schema_version: str = Field(..., description="Client schema version")
    # Feature vector used for inference.
    features: List[float] = Field(..., description="Raw feature vector")

    # Validate that features are finite (no NaN/Inf) at schema level.
    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: List[float]) -> List[float]:
        # Import math for isnan/isinf checks.
        import math
        # Iterate through each feature.
        for x in v:
            # Reject NaN explicitly.
            if math.isnan(x):
                raise ValueError("features contains NaN")
            # Reject Inf explicitly.
            if math.isinf(x):
                raise ValueError("features contains Inf")
        # Return validated list.
        return v

# Response schema for predict endpoint.
class PredictResponse(BaseModel):
    # Predicted label (0 or 1).
    label: int
    # Probability score used for decisioning.
    probability: float
    # Model version for traceability.
    model_version: str
    # Threshold policy version for traceability.
    threshold_version: str
