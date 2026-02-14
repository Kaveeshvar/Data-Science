# Import json to save/load artifacts.
import json

# Import typing for type hints.
from typing import Any, Dict

# Import Pipeline and Transformer types for safe (de)serialization.
from transforms import Pipeline, Transformer


# Save a Transformer (usually a Pipeline) to JSON.
def save_transform(transform: Transformer, path: str) -> None:
    # Convert transform into a JSON-safe dictionary.
    d = transform.to_dict()
    # Open file in write mode with utf-8 encoding.
    with open(path, "w", encoding="utf-8") as f:
        # Dump JSON with indentation for readability.
        json.dump(d, f, indent=2)


# Load a Transformer (Pipeline) from JSON.
def load_transform(path: str) -> Transformer:
    # Open file in read mode.
    with open(path, "r", encoding="utf-8") as f:
        # Parse JSON into a dictionary.
        d = json.load(f)
    # If artifact is a pipeline, reconstruct pipeline.
    if d.get("type") == "Pipeline":
        # Return deserialized pipeline.
        return Pipeline.from_dict(d)
    # Otherwise, reject unknown root types for safety.
    raise ValueError(f"Unknown root transform type: {d.get('type')}")
