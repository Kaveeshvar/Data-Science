# app/main.py  # FastAPI app definition and /predict endpoint.

from fastapi import FastAPI, Request  # Import FastAPI core and Request object.
from fastapi.responses import JSONResponse  # Import JSONResponse for consistent error JSON.
from app.schemas import PredictRequest, PredictResponse  # Import Pydantic request/response schemas.
from app.model import LinearRegressionModel, ModelArtifactError  # Import model wrapper and artifact error type.
from app import config  # Import config values.
import numpy as np  # Import numpy for numeric array handling.


app = FastAPI(  # Create the FastAPI application.
    title="Secure ML API (Minimal)",  # Set title.
    version="0.1.0",  # Set API version (not model version).
    docs_url=None,  # Disable Swagger docs UI to reduce attack surface.
    redoc_url=None,  # Disable ReDoc UI to reduce attack surface.
)  # End app creation.


MODEL = None  # Global model instance holder.


@app.middleware("http")  # Register middleware for all HTTP requests.
async def body_size_limit_middleware(request: Request, call_next):  # Middleware function.
    body = await request.body()  # Read full request body bytes once.
    if len(body) > int(config.MAX_REQUEST_BYTES):  # Enforce body size limit.
        return JSONResponse(  # Return error response.
            status_code=413,  # Payload Too Large.
            content={"error": "Request body too large"},  # Generic message.
        )  # End response.
    request._body = body  # Cache the body to avoid downstream re-read issues.
    response = await call_next(request)  # Continue processing request.
    return response  # Return response.


@app.on_event("startup")  # Register startup event.
def load_model_on_startup():  # Startup function to load model artifact.
    global MODEL  # Use the global MODEL variable.
    MODEL = LinearRegressionModel.load_from_json(config.MODEL_ARTIFACT_PATH)  # Load artifact and validate it.
    config.N_FEATURES = MODEL.n_features  # Publish n_features for schema validators.


@app.exception_handler(ModelArtifactError)  # Handle artifact issues cleanly.
async def artifact_error_handler(request: Request, exc: ModelArtifactError):  # Handler function.
    return JSONResponse(  # Return JSON error.
        status_code=500,  # Treat artifact failure as server-side (misconfiguration).
        content={"error": "Model artifact error"},  # Do not leak internals.
    )  # End response.


@app.exception_handler(ValueError)  # Handle ValueError cleanly (often input issues inside predict).
async def value_error_handler(request: Request, exc: ValueError):  # Handler function.
    return JSONResponse(  # Return JSON error response.
        status_code=400,  # Bad Request.
        content={"error": "Invalid input"},  # Generic message.
    )  # End response.


@app.exception_handler(Exception)  # Catch-all handler for any other unexpected exception.
async def generic_exception_handler(request: Request, exc: Exception):  # Handler function.
    return JSONResponse(  # Return a generic error response.
        status_code=500,  # Internal Server Error.
        content={"error": "Internal server error"},  # Do not leak details.
    )  # End response.


@app.post("/predict", response_model=PredictResponse)  # Define /predict endpoint and its response model.
def predict(req: PredictRequest):  # Endpoint function with validated request schema.
    if req.schema_version != config.EXPECTED_SCHEMA_VERSION:  # Validate schema version.
        return JSONResponse(  # Return error response.
            status_code=400,  # Bad Request.
            content={"error": "Invalid schema_version"},  # Generic message.
        )  # End response.

    if MODEL is None:  # Defensive check for model loaded.
        return JSONResponse(  # Return error response.
            status_code=503,  # Service Unavailable.
            content={"error": "Model not loaded"},  # Generic message.
        )  # End response.

    x = np.asarray(req.features, dtype=float)  # Convert features list into numpy float array.
    y = MODEL.predict(x)  # Run inference (may raise ValueError, which we handle globally).
    return PredictResponse(  # Return typed success response.
        prediction=float(y),  # Ensure JSON float.
        model_version=str(MODEL.model_version),  # Provide model version for traceability.
    )  # End response.
    