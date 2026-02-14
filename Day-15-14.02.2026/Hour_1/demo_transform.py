# Import numpy for generating demo data.
import numpy as np

# Import our transforms.
from transforms import StandardScalerSafe, Log1pSafe, Pipeline

# Import save/load helpers.
from artifact_io import save_transform, load_transform


# Main demo function.
def main() -> None:
    # Create a deterministic random generator.
    rng = np.random.default_rng(42)
    # Create synthetic dataset: 1000 rows, 4 columns.
    X = rng.normal(size=(1000, 4))
    # Make column 0 strictly nonnegative so Log1pSafe can apply.
    X[:, 0] = np.abs(X[:, 0]) * 10.0
    # Make column 3 a constant column to test constant handling.
    X[:, 3] = 5.0

    # Split indices for train/val/test.
    idx = np.arange(X.shape[0])
    # Shuffle indices deterministically.
    rng.shuffle(idx)
    # Compute split sizes.
    n_train = int(0.7 * len(idx))
    n_val = int(0.15 * len(idx))
    # Slice train indices.
    train_idx = idx[:n_train]
    # Slice val indices.
    val_idx = idx[n_train:n_train + n_val]
    # Slice test indices.
    test_idx = idx[n_train + n_val:]

    # Create splits.
    X_train = X[train_idx]
    # Create val split.
    X_val = X[val_idx]
    # Create test split.
    X_test = X[test_idx]

    # Build pipeline: log1p on column 0, then standardize all columns.
    pipe = Pipeline(
        steps=[
            Log1pSafe(columns=[0]),
            StandardScalerSafe(),
        ]
    )

    # Fit ONLY on train (this is the #1 rule).
    pipe.fit(X_train)

    # Transform each split using train-fitted params.
    Z_train = pipe.transform(X_train)
    # Transform val.
    Z_val = pipe.transform(X_val)
    # Transform test.
    Z_test = pipe.transform(X_test)

    # Compute train mean (should be ~0).
    train_mean = Z_train.mean(axis=0)
    # Compute train std (should be ~1, except constant column becomes 0 after scaling).
    train_std = Z_train.std(axis=0)

    # Print sanity checks.
    print("Train mean (≈0):", np.round(train_mean, 4))
    # Print std checks.
    print("Train std (≈1; constant col may be 0):", np.round(train_std, 4))

    # Save pipeline artifact.
    save_transform(pipe, "transform_artifact.json")
    # Load pipeline artifact back.
    loaded = load_transform("transform_artifact.json")

    # Transform again using loaded pipeline.
    Z_train_2 = loaded.transform(X_train)
    # Check determinism (exact match).
    print("Determinism check:", np.allclose(Z_train, Z_train_2))

    # Print quick validation: val/test mean won't necessarily be 0 (that’s expected).
    print("Val mean (not necessarily 0):", np.round(Z_val.mean(axis=0), 4))
    # Print test mean.
    print("Test mean (not necessarily 0):", np.round(Z_test.mean(axis=0), 4))


# Run the demo.
if __name__ == "__main__":
    # Call main.
    main()
