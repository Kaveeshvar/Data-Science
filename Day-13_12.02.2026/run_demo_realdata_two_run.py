import numpy as np
from sklearn.datasets import load_diabetes

from linear_regression_gd import LinearRegressionGD


def deterministic_split(X, y, n_train=300):
    """
    Deterministic split: first n_train rows are train, the rest are test.
    No randomness. Reproducible every time.
    """
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, y_train, X_test, y_test


def poison_one_point(X_train, y_train):
    """
    Deterministic poisoning:
    - Take the first training sample
    - Make it a high-leverage point by scaling features
    - Give it an extreme label (target)
    - Append it as one additional row
    """
    x0 = X_train[0].copy()
    y0 = float(y_train[0])

    # High leverage: amplify feature magnitude (diabetes features are already standardized-ish)
    x_bad = x0 * 25.0

    # Extreme label: inflate target so squared loss gets dominated
    y_bad = y0 + 500.0

    X_poison = np.vstack([X_train, x_bad.reshape(1, -1)])
    y_poison = np.concatenate([y_train, np.array([y_bad])])
    return X_poison, y_poison


def train_eval(tag, X_train, y_train, X_test, y_test, reg_lambda=0.0):
    """
    Train GD LR and evaluate MSE on both train and test.
    """
    model = LinearRegressionGD(
        lr=0.1,
        epochs=20000,
        fit_intercept=True,
        standardize=True,
        reg_lambda=reg_lambda,
        tol=1e-12,
        verbose=False,
        clip_grad=None,
    )

    model.fit(X_train, y_train)

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    train_mse = float(np.mean((y_hat_train - y_train) ** 2))
    test_mse = float(np.mean((y_hat_test - y_test) ** 2))

    print(f"\n=== {tag} ===")
    print(f"reg_lambda: {reg_lambda}")
    print("w (first 5):", model.w_[:5])
    print("b:", model.b_)
    print("train MSE:", train_mse)
    print("test  MSE:", test_mse)

    return model, train_mse, test_mse


def main():
    # 1) Load real dataset (deterministic)
    data = load_diabetes()
    X = data.data.astype(float)
    y = data.target.astype(float)

    # 2) Deterministic split (no randomness)
    X_train, y_train, X_test, y_test = deterministic_split(X, y, n_train=300)

    # 3) Create poisoned training set (deterministic)
    X_poison, y_poison = poison_one_point(X_train, y_train)

    print("Dataset:", "Diabetes (sklearn built-in)")
    print("Shapes:",
          "X_train", X_train.shape,
          "X_test", X_test.shape,
          "X_poison", X_poison.shape)

    # =========================
    # CONFIG A: Baseline (no L2)
    # =========================
    clean_model_A, clean_train_mse_A, clean_test_mse_A = train_eval(
        tag="RUN 1 (CLEAN) — Baseline",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        reg_lambda=0.0
    )

    poison_model_A, poison_train_mse_A, poison_test_mse_A = train_eval(
        tag="RUN 2 (POISONED) — Baseline",
        X_train=X_poison, y_train=y_poison,
        X_test=X_test, y_test=y_test,
        reg_lambda=0.0
    )

    print("\n--- Baseline impact ---")
    print("Δtest MSE:", poison_test_mse_A - clean_test_mse_A)
    print("||Δw||:", float(np.linalg.norm(poison_model_A.w_ - clean_model_A.w_)))
    print("|Δb|  :", float(abs(poison_model_A.b_ - clean_model_A.b_)))

    # ======================
    # CONFIG B: L2 Defense
    # ======================
    clean_model_B, clean_train_mse_B, clean_test_mse_B = train_eval(
        tag="RUN 1 (CLEAN) — L2 Defense",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        reg_lambda=1.0
    )

    poison_model_B, poison_train_mse_B, poison_test_mse_B = train_eval(
        tag="RUN 2 (POISONED) — L2 Defense",
        X_train=X_poison, y_train=y_poison,
        X_test=X_test, y_test=y_test,
        reg_lambda=1.0
    )

    print("\n--- L2 Defense impact ---")
    print("Δtest MSE:", poison_test_mse_B - clean_test_mse_B)
    print("||Δw||:", float(np.linalg.norm(poison_model_B.w_ - clean_model_B.w_)))
    print("|Δb|  :", float(abs(poison_model_B.b_ - clean_model_B.b_)))

    # Side-by-side summary
    print("\n=== SUMMARY (Test MSE) ===")
    print("Baseline clean :", clean_test_mse_A)
    print("Baseline poison:", poison_test_mse_A)
    print("L2 clean       :", clean_test_mse_B)
    print("L2 poison      :", poison_test_mse_B)


if __name__ == "__main__":
    main()
