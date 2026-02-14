# Import json to save artifacts.
import json

# Import numpy for computations.
import numpy as np

# Import sklearn dataset and split helper.
from sklearn.datasets import load_breast_cancer
# Import train/test split.
from sklearn.model_selection import train_test_split

# Import pipeline + transforms.
from transforms import Pipeline, StandardScalerSafe, MinMaxSafe, RobustScalerSafe, Log1pSafe

# Import diagnostics helpers.
from diagnostics import summary_stats, drift_report, psi


# Run demo.
def main() -> None:
    # Load a real dataset.
    data = load_breast_cancer()
    # Get features.
    X = data.data.astype(np.float64)
    # Create a split (train vs "prod") for drift simulation.
    X_train, X_prod = train_test_split(X, test_size=0.3, random_state=42, shuffle=True)

    # Create a fake "prod drift" by shifting a feature a bit (simulates real drift).
    X_prod_drift = X_prod.copy()
    # Shift first feature upward slightly to produce detectable drift.
    X_prod_drift[:, 0] = X_prod_drift[:, 0] + 1.0

    # Build a pipeline (example): robust scaling then minmax with clipping.
    pipe = Pipeline(
        steps=[
            RobustScalerSafe(),
            MinMaxSafe(clip=True),
        ]
    )

    # Fit ONLY on train (the #1 rule).
    pipe.fit(X_train)

    # Transform both sets using train-fitted params.
    Z_train = pipe.transform(X_train)
    Z_prod = pipe.transform(X_prod_drift)

    # Print train sanity checks.
    print("Train transformed mean (not necessarily 0 due to Robust+MinMax):", np.round(Z_train.mean(axis=0)[:5], 4))
    # Print clip counter from MinMaxSafe (step 1 is Robust, step 2 is MinMax).
    mm = pipe.steps[1]
    # Print how many values were clipped due to out-of-range.
    print("MinMax clip_count_:", getattr(mm, "clip_count_", None))

    # Compute stats for train and prod.
    train_stats = summary_stats(Z_train)
    # Compute stats for prod.
    prod_stats = summary_stats(Z_prod)

    # Drift report.
    report = drift_report(train_stats, prod_stats, z_threshold=3.0)
    # Print number of features flagged.
    print("Mean shift flags count:", int(np.sum(np.array(report["mean_shift_flags"]))))

    # PSI per feature.
    psi_vals = psi(Z_train, Z_prod, bins=10)
    # Print a few PSI values.
    print("PSI first 5 features:", [round(v, 4) for v in psi_vals[:5]])

    # Save pipeline artifact.
    artifact = pipe.to_dict()
    # Write JSON artifact to disk.
    with open("transform_artifact.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    # Load pipeline artifact back.
    with open("transform_artifact.json", "r", encoding="utf-8") as f:
        artifact2 = json.load(f)

    # Reconstruct pipeline.
    pipe2 = Pipeline.from_dict(artifact2)

    # Verify determinism of saved/loaded pipeline.
    Z_train2 = pipe2.transform(X_train)
    # Print determinism check.
    print("Artifact roundtrip determinism:", bool(np.allclose(Z_train, Z_train2)))


# Execute demo.
if __name__ == "__main__":
    main()
