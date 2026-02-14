# Import json for artifact save/load.
import json

# Import numpy for array construction.
import numpy as np

# Import transformers.
from transforms import StandardScalerSafe, MinMaxSafe, Pipeline


# 1) Train-only fit: mu/sigma must match train subset, not full data.
def test_train_only_fit_no_leakage():
    # Construct train with zeros only.
    X_train = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    # Construct prod/test with huge values.
    X_prod = np.array([[1000.0], [1000.0]], dtype=np.float64)
    # Fit scaler on train only.
    sc = StandardScalerSafe().fit(X_train)
    # Ensure mean equals train mean (0).
    assert np.isclose(sc.mu_[0], 0.0)
    # Ensure std was stabilized (train std is 0, sigma should be 1.0).
    assert np.isclose(sc.sigma_[0], 1.0)
    # Transform prod should yield huge values (reveals drift, no normalization cheating).
    Z_prod = sc.transform(X_prod)
    # Check it is large.
    assert Z_prod.mean() > 100.0


# 2) NaN/Inf rejection.
def test_nan_inf_rejection():
    # Fit on simple valid data.
    sc = StandardScalerSafe().fit([[1.0, 2.0], [3.0, 4.0]])
    # Create NaN input.
    X_nan = np.array([[np.nan, 2.0]], dtype=np.float64)
    # Expect ValueError.
    try:
        sc.transform(X_nan)
        assert False, "Expected NaN rejection"
    except ValueError:
        assert True
    # Create Inf input.
    X_inf = np.array([[np.inf, 2.0]], dtype=np.float64)
    # Expect ValueError.
    try:
        sc.transform(X_inf)
        assert False, "Expected Inf rejection"
    except ValueError:
        assert True


# 3) Constant column stability.
def test_constant_column_stability():
    # Create data with constant second column.
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float64)
    # Fit scaler.
    sc = StandardScalerSafe().fit(X)
    # Transform.
    Z = sc.transform(X)
    # Constant column should become all zeros.
    assert np.allclose(Z[:, 1], 0.0)


# 4) Unseen out-of-range in MinMax triggers clip counter.
def test_minmax_clip_counter():
    # Train data defines min=0, max=10.
    X_train = np.array([[0.0], [10.0]], dtype=np.float64)
    # Prod data goes out of range.
    X_prod = np.array([[-5.0], [15.0]], dtype=np.float64)
    # Fit MinMax with clipping enabled.
    mm = MinMaxSafe(clip=True).fit(X_train)
    # Transform prod (should clip both entries).
    Z = mm.transform(X_prod)
    # Both values should be within [0, 1].
    assert np.all((Z >= 0.0) & (Z <= 1.0))
    # Clip counter should be 2 (both values clipped).
    assert mm.clip_count_ == 2


# 5) Pipeline determinism: load saved params → identical output.
def test_pipeline_artifact_roundtrip_determinism(tmp_path):
    # Create simple dataset.
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    # Build pipeline with scaler then minmax.
    pipe = Pipeline(steps=[StandardScalerSafe(), MinMaxSafe(clip=True)])
    # Fit pipeline.
    pipe.fit(X)
    # Transform once.
    Z1 = pipe.transform(X)
    # Save artifact to file.
    p = tmp_path / "artifact.json"
    # Write artifact.
    p.write_text(json.dumps(pipe.to_dict(), indent=2), encoding="utf-8")
    # Load artifact.
    d = json.loads(p.read_text(encoding="utf-8"))
    # Reconstruct pipeline.
    pipe2 = Pipeline.from_dict(d)
    # Transform again.
    Z2 = pipe2.transform(X)
    # Must match.
    assert np.allclose(Z1, Z2)
