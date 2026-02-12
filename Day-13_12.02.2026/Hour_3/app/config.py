# app/config.py  # This file stores runtime config shared across modules.

from pathlib import Path

EXPECTED_SCHEMA_VERSION = "1.0"  # This is the only request schema_version we accept for now.
APP_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACT_PATH = str(APP_DIR / "model_artifacts" / "linear_regression_v1.json")
MAX_REQUEST_BYTES = 8 * 1024  # Hard limit for request body size (8 KB) to reduce abuse risk.
FEATURE_ABS_MAX = 1e6  # Safety bound (Hour 3 asks this; harmless to set now).
N_FEATURES = None  # Will be populated at startup from the artifact (don’t hardcode).
