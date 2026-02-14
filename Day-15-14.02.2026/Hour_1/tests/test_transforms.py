# Import numpy for test arrays.
import numpy as np

# Import transforms to test.
from transforms import StandardScalerSafe, Log1pSafe, Pipeline


# Test: no leakage means mu/sigma must match TRAIN only, not full data.
def test_no_leakage_train_only_params():
    # Create a train set with small values.
    X_train = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    # Create a test set with huge values (different distribution).
    X_test = np.array([[1000.0], [1000.0]], dtype=np.float64)
    # Fit scaler only on train.
    sc = StandardScalerSafe().fit(X_train)
    # Train mean must be 0.
    assert np.isclose(sc.mu_[0], 0.0)
    # If leakage happened, mu would be around 400, not 0.
    assert sc.mu_[0] < 1.0
    # Transform test should produce large values, revealing distribution shift.
    Z_test = sc.transform(X_test)
    # Ensure transformed test is very large (not “normalized away” by leakage).
    assert Z_test.mean() > 100.0


# Test: NaN/Inf rejection.
def test_nan_inf_rejection():
    # Create valid data.
    X = np.array([[1.0, 2.0]], dtype=np.float64)
    # Create scaler.
    sc = StandardScalerSafe()
    # Fit should work.
    sc.fit(X)
    # Create NaN input.
    X_nan = np.array([[np.nan, 2.0]], dtype=np.float64)
    # Transform must reject NaN.
    try:
        sc.transform(X_nan)
        assert False, "Expected NaN rejection"
    except ValueError as e:
        assert "NaN" in str(e) or "Inf" in str(e)
    # Create Inf input.
    X_inf = np.array([[np.inf, 2.0]], dtype=np.float64)
    # Transform must reject Inf.
    try:
        sc.transform(X_inf)
        assert False, "Expected Inf rejection"
    except ValueError as e:
        assert "NaN" in str(e) or "Inf" in str(e)


# Test: constant-column handling (std=0 => sigma becomes 1 => transformed column becomes 0).
def test_constant_column_handling():
    # Create data with constant second column.
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float64)
    # Fit scaler.
    sc = StandardScalerSafe().fit(X)
    # Transform data.
    Z = sc.transform(X)
    # Constant column should become exactly zeros.
    assert np.allclose(Z[:, 1], 0.0)


# Test: determinism (same input gives same output).
def test_determinism_same_input_same_output():
    # Create deterministic data.
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    # Build pipeline with scaler only.
    pipe = Pipeline(steps=[StandardScalerSafe()])
    # Fit pipeline on X.
    pipe.fit(X)
    # Transform twice.
    Z1 = pipe.transform(X)
    # Transform again.
    Z2 = pipe.transform(X)
    # Outputs must be identical.
    assert np.allclose(Z1, Z2)


# Test: Log1pSafe rejects negatives on selected columns.
def test_log1p_rejects_negative():
    # Create data with negative in column 0.
    X = np.array([[-1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    # Create Log1pSafe for column 0.
    t = Log1pSafe(columns=[0])
    # Fit must reject because train contains negative.
    try:
        t.fit(X)
        assert False, "Expected negative rejection"
    except ValueError as e:
        assert "nonnegative" in str(e)


# Test: pipeline works end-to-end with Log1p + scaler.
def test_pipeline_end_to_end():
    # Create data where column 0 is nonnegative.
    X = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]], dtype=np.float64)
    # Build pipeline.
    pipe = Pipeline(steps=[Log1pSafe(columns=[0]), StandardScalerSafe()])
    # Fit pipeline.
    pipe.fit(X)
    # Transform.
    Z = pipe.transform(X)
    # Output shape must match.
    assert Z.shape == X.shape
    # Train mean should be ~0 after full pipeline.
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
