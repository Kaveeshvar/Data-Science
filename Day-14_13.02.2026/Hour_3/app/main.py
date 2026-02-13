# Import os for path handling.
import os
# Import FastAPI for building the API.
from fastapi import FastAPI, HTTPException, Request
# Import FastAPI validation error type.
from fastapi.exceptions import RequestValidationError
# Import JSONResponse for custom error output.
from fastapi.responses import JSONResponse
# Import math for NaN/Inf checks in error sanitization.
import math
# Import our schemas for request/response typing.
from app.schemas import PredictProbaRequest, PredictProbaResponse, PredictRequest, PredictResponse
# Import our model wrapper.
from app.model import LogRegModel

# Create the FastAPI application instance.
app = FastAPI(title="Secure Logistic Regression API", version="1.0")

# Build absolute paths to the artifacts folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define path to model artifact JSON.
MODEL_PATH = os.path.join(BASE_DIR, "model_artifacts", "logreg_v1.json")
# Define path to threshold policy JSON.
THRESHOLD_PATH = os.path.join(BASE_DIR, "model_artifacts", "threshold_policy_v1.json")

# Initialize model as None until startup loads it.
model: LogRegModel | None = None

# Startup event to load artifacts once per process.
@app.on_event("startup")
def load_artifacts() -> None:
    # Use global so we can set the module-level model variable.
    global model
    # Instantiate the wrapper with artifact paths.
    model = LogRegModel(model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH)

# Ensure the model is loaded, even if startup was skipped (e.g., tests).
def _get_model() -> LogRegModel:
    global model
    # Lazily load on first use if startup did not run.
    if model is None:
        load_artifacts()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return model

# Shared validation for feature length and schema version.
def _validate_request(schema_version: str, features: list[float]) -> None:
    # Ensure model is available.
    m = _get_model()
    # Enforce schema version match to prevent silent incompatibilities.
    if schema_version != m.schema_version:
        raise HTTPException(status_code=400, detail="schema_version mismatch")
    # Enforce feature length matches model expectation.
    if len(features) != m.n_features:
        raise HTTPException(status_code=400, detail="feature length mismatch")
    # Defense-in-depth: reject absurd magnitudes to reduce overflow/abuse.
    for x in features:
        if abs(x) > 1e6:
            raise HTTPException(status_code=400, detail="feature magnitude too large")

# Sanitize validation errors so NaN/Inf in "input" does not crash JSON rendering.
def _sanitize_non_finite(value):
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_sanitize_non_finite(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    return value

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    sanitized = _sanitize_non_finite(exc.errors())
    return JSONResponse(status_code=422, content={"detail": sanitized})

# Endpoint: probability only.
@app.post("/predict_proba", response_model=PredictProbaResponse)
def predict_proba(req: PredictProbaRequest) -> PredictProbaResponse:
    # Validate schema version and feature shape.
    _validate_request(req.schema_version, req.features)
    # Ensure model is loaded.
    m = _get_model()
    # Compute probability using model wrapper.
    try:
        p = m.predict_proba(req.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Return probability and model version.
    return PredictProbaResponse(probability=p, model_version=m.model_version)

# Endpoint: label + probability using server-side threshold.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Validate schema version and feature shape.
    _validate_request(req.schema_version, req.features)
    # Ensure model is loaded.
    m = _get_model()
    # Compute label and probability using model wrapper.
    try:
        out = m.predict_label(req.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Return full response with traceability metadata.
    return PredictResponse(
        label=int(out["label"]),
        probability=float(out["probability"]),
        model_version=m.model_version,
        threshold_version=m.policy_version,
    )
