import numpy as np
from model import LinearRegressionModel

m = LinearRegressionModel.load_from_json("model_artifacts/linear_regression_v1.json")
print("Artifact SHA256:", m.artifact_sha256)
print("Model version:", m.model_version)

x = np.array([12, 18, 33, 39, 49,12, 18, 33, 39, 49], dtype=float)
print("Prediction:", m.predict(x))
