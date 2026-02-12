import numpy as np
from linear_regression_gd import LinearRegressionGD


# -----------------------------
# Utility: compute loss directly
# -----------------------------
def compute_loss_half_mse_l2(X, y, w, b, reg_lambda=0.0, fit_intercept=True):
    """
    Computes half-MSE loss + optional L2 regularization on w.

    Loss = 0.5 * mean((y_hat - y)^2) + (lambda/(2n)) * ||w||^2
    """
    n = y.shape[0]

    # prediction
    y_hat = X @ w + (b if fit_intercept else 0.0)

    # residual
    res = y_hat - y

    # half MSE
    loss = 0.5 * np.mean(res ** 2)

    # L2 penalty (w only)
    if reg_lambda > 0.0:
        loss += (reg_lambda / (2.0 * n)) * np.dot(w, w)

    return loss


# -------------------------------------------
# Step E1: Finite-difference gradient checking
# -------------------------------------------
def finite_difference_grad_check():
    """
    Checks analytical gradients against numerical gradients.
    This is the fastest way to catch a wrong gradient implementation.
    """

    np.random.seed(42)

    # tiny dataset so numerical gradients are cheap
    n, d = 6, 3
    X = np.random.randn(n, d)
    true_w = np.array([2.0, -1.0, 0.5])
    true_b = 1.5

    # generate y with small noise
    y = X @ true_w + true_b + 0.01 * np.random.randn(n)

    # set some test parameters (not necessarily optimum)
    w = np.random.randn(d)
    b = 0.3

    reg_lambda = 0.7
    fit_intercept = True

    # ----- Analytical gradients (the ones you derived) -----
    y_hat = X @ w + b
    res = y_hat - y

    # dw = (X^T res)/n + (lambda/n)w
    dw_analytic = (X.T @ res) / n + (reg_lambda / n) * w

    # db = mean(res)
    db_analytic = res.mean()

    # ----- Numerical gradients (finite differences) -----
    eps = 1e-6  # small step for numerical derivative

    # numeric grad for w
    dw_numeric = np.zeros_like(w)
    for j in range(d):
        w_plus = w.copy()
        w_minus = w.copy()

        w_plus[j] += eps
        w_minus[j] -= eps

        # central difference: (f(x+e)-f(x-e)) / (2e)
        loss_plus = compute_loss_half_mse_l2(X, y, w_plus, b, reg_lambda, fit_intercept)
        loss_minus = compute_loss_half_mse_l2(X, y, w_minus, b, reg_lambda, fit_intercept)

        dw_numeric[j] = (loss_plus - loss_minus) / (2.0 * eps)

    # numeric grad for b
    b_plus = b + eps
    b_minus = b - eps

    loss_plus = compute_loss_half_mse_l2(X, y, w, b_plus, reg_lambda, fit_intercept)
    loss_minus = compute_loss_half_mse_l2(X, y, w, b_minus, reg_lambda, fit_intercept)

    db_numeric = (loss_plus - loss_minus) / (2.0 * eps)

    # ----- Compare -----
    # Relative error is better than absolute error (scale-invariant)
    rel_err_w = np.linalg.norm(dw_numeric - dw_analytic) / (np.linalg.norm(dw_numeric) + np.linalg.norm(dw_analytic) + 1e-12)
    rel_err_b = abs(db_numeric - db_analytic) / (abs(db_numeric) + abs(db_analytic) + 1e-12)

    print("\n=== Finite Difference Gradient Check ===")
    print("dw_numeric :", dw_numeric)
    print("dw_analytic:", dw_analytic)
    print("db_numeric :", db_numeric)
    print("db_analytic:", db_analytic)
    print(f"Relative error (w): {rel_err_w:.3e}")
    print(f"Relative error (b): {rel_err_b:.3e}")

    # Rule of thumb: < 1e-5 is usually excellent
    if rel_err_w < 1e-5 and rel_err_b < 1e-5:
        print("✅ Gradient check PASSED (your gradients are correct).")
    else:
        print("❌ Gradient check FAILED (your gradients likely have a bug).")


# ------------------------------------------------
# Step E2: Compare GD vs Normal Equation (pinv)
# ------------------------------------------------
def compare_with_closed_form():
    """
    Trains GD and compares predictions & parameters with closed-form solution.
    Uses pinv to avoid singular matrix issues.
    """

    np.random.seed(0)

    # generate synthetic data
    n, d = 200, 2
    X = np.random.randn(n, d)
    true_w = np.array([4.0, -2.0])
    true_b = 3.0
    y = X @ true_w + true_b + 0.1 * np.random.randn(n)

    # Train GD model
    gd = LinearRegressionGD(
        lr=0.1,
        epochs=5000,
        fit_intercept=True,
        standardize=True,     # helps GD converge faster
        reg_lambda=0.0,
        tol=1e-12,
        verbose=False
    )
    gd.fit(X, y)

    # GD predictions
    y_hat_gd = gd.predict(X)
    mse_gd = np.mean((y_hat_gd - y) ** 2)

    # Closed-form solution (Normal Equation via pinv)
    # IMPORTANT: Use the SAME standardized X if GD used standardization.
    # We want apples-to-apples in parameter space.
    X_std = (X - gd.mu_) / gd.sigma_

    # Add column of ones to solve intercept in closed form
    X_aug = np.c_[X_std, np.ones(n)]

    # theta = pinv(X_aug) @ y
    theta = np.linalg.pinv(X_aug) @ y

    w_cf = theta[:-1]
    b_cf = theta[-1]

    # Closed-form predictions (on standardized X)
    y_hat_cf = X_std @ w_cf + b_cf
    mse_cf = np.mean((y_hat_cf - y) ** 2)

    # Compare
    print("\n=== GD vs Closed-form (pinv) Comparison ===")
    print("GD w:", gd.w_)
    print("GD b:", gd.b_)
    print("CF w:", w_cf)
    print("CF b:", b_cf)
    print(f"MSE (GD): {mse_gd:.6f}")
    print(f"MSE (CF): {mse_cf:.6f}")

    # These should be extremely close (small numeric differences are normal)
    print("||w_gd - w_cf||:", np.linalg.norm(gd.w_ - w_cf))
    print("|b_gd - b_cf|  :", abs(gd.b_ - b_cf))


if __name__ == "__main__":
    finite_difference_grad_check()
    compare_with_closed_form()
