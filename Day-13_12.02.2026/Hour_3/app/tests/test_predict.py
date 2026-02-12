# tests/test_predict.py  # Pytest suite for /predict endpoint security and validation behavior.

import pytest  # Pytest framework for tests and fixtures.
from fastapi.testclient import TestClient  # FastAPI test client for HTTP calls.

from app.main import app, load_model_on_startup  # Import app instance and startup loader.
from app import config  # Import config for schema version, bounds, and feature count.


@pytest.fixture(scope="session", autouse=True)  # Run this once automatically for the whole test session.
def _init_model_once():  # Fixture that ensures startup initialization happens.
    load_model_on_startup()  # Load artifact and set config.N_FEATURES so validators work.


@pytest.fixture()  # Provide a fresh client per test.
def client():  # Fixture returning a TestClient with proper startup/shutdown handling.
    with TestClient(app) as c:  # Use context manager so FastAPI startup hooks run consistently.
        yield c  # Provide the client to the test.


def make_valid_payload():  # Helper to build a valid request payload.
    n = int(config.N_FEATURES)  # Read model feature count (must be set by startup).
    return {  # Build a dict payload.
        "schema_version": config.EXPECTED_SCHEMA_VERSION,  # Set correct schema version.
        "features": [0.01] * n,  # Provide n finite, bounded floats.
    }  # End payload.


def test_valid_request_returns_200(client):  # Test that a correct request succeeds.
    payload = make_valid_payload()  # Create valid payload.
    r = client.post("/predict", json=payload)  # Call /predict endpoint.
    assert r.status_code == 200  # Expect HTTP 200 OK.
    data = r.json()  # Parse JSON response.
    assert "prediction" in data  # Ensure prediction key exists.
    assert "model_version" in data  # Ensure model_version key exists.
    assert isinstance(data["prediction"], (int, float))  # Ensure prediction is numeric.
    assert isinstance(data["model_version"], str)  # Ensure model_version is string.


def test_wrong_feature_length_fails_cleanly(client):  # Test that wrong length is rejected.
    payload = make_valid_payload()  # Start from valid payload.
    payload["features"] = payload["features"][:-1]  # Remove one feature to break length contract.
    r = client.post("/predict", json=payload)  # Call endpoint.
    assert r.status_code in (400, 422)  # Expect a clean client error (usually 422 from Pydantic).


def test_wrong_schema_version_fails_cleanly(client):  # Test that wrong schema_version is rejected.
    payload = make_valid_payload()  # Build valid payload.
    payload["schema_version"] = "999.999"  # Set invalid schema version.
    r = client.post("/predict", json=payload)  # Call endpoint.
    assert r.status_code == 400  # Expect 400 Bad Request (handled in endpoint).
    assert "error" in r.json()  # Ensure error field exists.


def test_nan_string_in_features_fails_cleanly(client):  # Test that non-numeric "NaN" is rejected.
    payload = make_valid_payload()  # Build valid payload.
    payload["features"][0] = "NaN"  # JSON cannot carry NaN float, so send a string.
    r = client.post("/predict", json=payload)  # Call endpoint.
    assert r.status_code == 422  # Pydantic should reject invalid type/value.


def test_inf_string_in_features_fails_cleanly(client):  # Test that non-numeric "Infinity" is rejected.
    payload = make_valid_payload()  # Build valid payload.
    payload["features"][0] = "Infinity"  # JSON cannot carry Inf float, so send a string.
    r = client.post("/predict", json=payload)  # Call endpoint.
    assert r.status_code == 422  # Pydantic should reject invalid type/value.


def test_large_magnitude_feature_fails_cleanly(client):  # Test that huge values are rejected.
    payload = make_valid_payload()  # Build valid payload.
    payload["features"][0] = float(config.FEATURE_ABS_MAX)  # Set boundary-violating value (>= abs max).
    r = client.post("/predict", json=payload)  # Call endpoint.
    assert r.status_code == 422  # Expect validation failure at schema layer.


def test_request_body_too_large_fails_cleanly(client):  # Test request size limit returns 413.
    big = "A" * (int(config.MAX_REQUEST_BYTES) + 100)  # Create a body larger than allowed bytes.
    r = client.post(  # Call endpoint with raw oversized body.
        "/predict",  # Endpoint path.
        data=big,  # Raw body (not valid JSON).
        headers={"Content-Type": "application/json"},  # Set JSON content-type to hit middleware path.
    )  # End request.
    assert r.status_code == 413  # Expect Payload Too Large.
    assert "error" in r.json()  # Ensure the response is JSON and has an error field.
