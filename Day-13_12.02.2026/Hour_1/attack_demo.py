import numpy as np
from linear_regression_gd import LinearRegressionGD


def make_clean_data(n=200, d=2, noise=0.2, seed=1):
    """
    Generates a clean linear dataset.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = np.array([3.0, -1.0])
    true_b = 2.0
    y = X @ true_w + true_b + noise * rng.normal(size=n)
    return X, y, true_w, true_b


def train_and_report(X, y, tag):
    """
    Trains GD and returns model + metrics.
    """
    model = LinearRegressionGD(
        lr=0.1,
        epochs=5000,
        fit_intercept=True,
        standardize=True,
        reg_lambda=0.0,
        tol=1e-12,
        verbose=False,
        clip_grad=None   # turn on later to see mitigation
    )
    model.fit(X, y)
    y_hat = model.predict(X)
    mse = np.mean((y_hat - y) ** 2)

    print(f"\n=== {tag} ===")
    print("w:", model.w_)
    print("b:", model.b_)
    print("train MSE:", mse)

    return model, mse


def poisoning_outlier_demo():
    """
    Shows how a single extreme point can yank parameters due to squared loss.
    """

    # 1) Clean data
    X_clean, y_clean, true_w, true_b = make_clean_data()
    clean_model, clean_mse = train_and_report(X_clean, y_clean, "CLEAN TRAINING")

    # We'll also create a clean test set to measure generalization impact
    X_test, y_test, _, _ = make_clean_data(n=200, d=2, noise=0.2, seed=999)
    y_test_hat_clean = clean_model.predict(X_test)
    test_mse_clean = np.mean((y_test_hat_clean - y_test) ** 2)

    print("\nClean TEST MSE:", test_mse_clean)

    # 2) Attack A: label outlier (huge y)
    X_poison_y = X_clean.copy()
    y_poison_y = y_clean.copy()

    # Add one poisoned sample
    x_bad = np.array([[10.0, 10.0]])     # leverage-ish input too
    y_bad = np.array([500.0])            # absurd target value

    X_poison_y = np.vstack([X_poison_y, x_bad])
    y_poison_y = np.concatenate([y_poison_y, y_bad])

    poisonY_model, poisonY_mse = train_and_report(X_poison_y, y_poison_y, "POISON: HUGE y OUTLIER")

    # Evaluate on clean test set
    y_test_hat_poisonY = poisonY_model.predict(X_test)
    test_mse_poisonY = np.mean((y_test_hat_poisonY - y_test) ** 2)

    print("\nPOISON-Y TEST MSE:", test_mse_poisonY)

    # Influence report (parameter shift)
    print("\n=== Influence: Clean vs Poison-Y ===")
    print("||Δw||:", np.linalg.norm(poisonY_model.w_ - clean_model.w_))
    print("|Δb|  :", abs(poisonY_model.b_ - clean_model.b_))
    print("Δtest_mse:", test_mse_poisonY - test_mse_clean)

    # 3) Attack B: leverage point (huge X, moderate y)
    X_poison_x = X_clean.copy()
    y_poison_x = y_clean.copy()

    # extreme feature values (high leverage), not even a huge y needed
    x_bad2 = np.array([[50.0, -50.0]])
    # choose y that forces slope distortion
    y_bad2 = np.array([-200.0])

    X_poison_x = np.vstack([X_poison_x, x_bad2])
    y_poison_x = np.concatenate([y_poison_x, y_bad2])

    poisonX_model, poisonX_mse = train_and_report(X_poison_x, y_poison_x, "POISON: HIGH-LEVERAGE X POINT")

    # Evaluate on clean test set
    y_test_hat_poisonX = poisonX_model.predict(X_test)
    test_mse_poisonX = np.mean((y_test_hat_poisonX - y_test) ** 2)

    print("\nPOISON-X TEST MSE:", test_mse_poisonX)

    # Influence report (parameter shift)
    print("\n=== Influence: Clean vs Poison-X ===")
    print("||Δw||:", np.linalg.norm(poisonX_model.w_ - clean_model.w_))
    print("|Δb|  :", abs(poisonX_model.b_ - clean_model.b_))
    print("Δtest_mse:", test_mse_poisonX - test_mse_clean)

    # 4) Optional mitigation knobs (quick preview)
    # Turn on L2 and/or gradient clipping and rerun to see if influence reduces.
    print("\n💡 Mitigation ideas to try next:")
    print("- standardize=True (already on)")
    print("- reg_lambda > 0 (L2 stabilizes weights)")
    print("- clip_grad (stops crazy updates)")
    print("- use robust loss (Huber) (later)")


if __name__ == "__main__":
    poisoning_outlier_demo()
