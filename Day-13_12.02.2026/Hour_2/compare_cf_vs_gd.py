import numpy as np  # numerical computing
import matplotlib.pyplot as plt  # plotting


def make_data(n, d, noise_std, seed, condition_mode="well"):
    rng = np.random.default_rng(seed)  # reproducible RNG

    X = rng.normal(0.0, 1.0, size=(n, d))  # base features

    if condition_mode == "ill" and d >= 2:  # near-collinearity
        tiny = 1e-2  # tiny noise to make it "almost" linear combo
        X[:, 1] = X[:, 0] + rng.normal(0.0, tiny, size=n)  # x2 ≈ x1

    true_w = rng.normal(0.0, 1.0, size=d)  # true weights
    true_b = rng.normal(0.0, 1.0)  # true intercept

    y_clean = X @ true_w + true_b  # noiseless labels
    y = y_clean + rng.normal(0.0, noise_std, size=n)  # noisy labels

    mu = X.mean(axis=0)  # feature means
    sigma = X.std(axis=0)  # feature stds

    print("\n=== Data stats ===")  # header
    print(f"mode={condition_mode}, n={n}, d={d}, noise_std={noise_std}")  # config
    print("feature mean (first 5):", np.round(mu[:5], 4))  # mean preview
    print("feature std  (first 5):", np.round(sigma[:5], 4))  # std preview

    corr = np.corrcoef(X, rowvar=False)  # correlation matrix
    snippet = corr[: min(d, 5), : min(d, 5)]  # top-left snippet
    print("corr snippet (top-left):\n", np.round(snippet, 4))  # print snippet

    return X, y, true_w, true_b  # return generated data


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))  # mean squared error


def closed_form_pinv(X, y):
    n = X.shape[0]  # number of rows
    ones = np.ones((n, 1))  # intercept column
    X_aug = np.hstack([ones, X])  # [1, X]

    theta = np.linalg.pinv(X_aug) @ y  # pseudoinverse solution

    b = theta[0]  # intercept
    w = theta[1:]  # weights

    yhat = X_aug @ theta  # predictions

    return w, b, yhat  # return solution


def standardize_fit_transform(X):
    mu = X.mean(axis=0)  # mean per feature
    sigma = X.std(axis=0)  # std per feature
    sigma_safe = np.where(sigma == 0.0, 1.0, sigma)  # avoid division by zero
    Xs = (X - mu) / sigma_safe  # standardized X
    return Xs, mu, sigma_safe  # return X scaled and params


def gd_linear_regression(
    X, y, lr=0.1, epochs=5000, tol=1e-10, standardize=False, log_every=100
):
    n, d = X.shape  # shapes

    if standardize:  # optionally scale
        Xs, mu, sigma = standardize_fit_transform(X)  # standardize
    else:
        Xs = X.copy()  # keep original
        mu = None  # placeholder
        sigma = None  # placeholder

    ones = np.ones((n, 1))  # intercept column
    X_aug = np.hstack([ones, Xs])  # augmented matrix

    theta = np.zeros(d + 1)  # init params (bias + weights)
    loss_hist = []  # store loss
    prev_loss = None  # for early stopping

    for epoch in range(epochs):  # training loop
        yhat = X_aug @ theta  # predictions
        err = yhat - y  # residuals

        loss = float(np.mean(err ** 2))  # MSE
        loss_hist.append(loss)  # store

        grad = (2.0 / n) * (X_aug.T @ err)  # gradient

        theta = theta - lr * grad  # update

        grad_norm = float(np.linalg.norm(grad))  # grad magnitude
        w_norm = float(np.linalg.norm(theta[1:]))  # weight magnitude

        if epoch % log_every == 0:  # periodic logging
            print(
                f"[GD][std={standardize}] epoch={epoch:5d} loss={loss:.6e} "
                f"||grad||={grad_norm:.3e} ||w||={w_norm:.3e}"
            )

        if prev_loss is not None:  # if we have previous loss
            if abs(prev_loss - loss) < tol:  # convergence check
                print(
                    f"[GD][std={standardize}] early stop at epoch={epoch}, "
                    f"delta={abs(prev_loss - loss):.3e}"
                )
                break  # stop training

        prev_loss = loss  # update prev loss

    b_s = theta[0]  # bias in scaled space
    w_s = theta[1:]  # weights in scaled space

    if standardize:  # map back to original space
        w = w_s / sigma  # unscale weights
        b = b_s - np.sum((mu / sigma) * w_s)  # adjust bias
    else:
        w = w_s  # already in original space
        b = b_s  # already in original space

    yhat_final = X @ w + b  # predictions on original X

    return w, b, yhat_final, loss_hist  # return everything


def add_poison_point(X, y, scale=30.0, y_scale=100.0, seed=123):
    rng = np.random.default_rng(seed)  # RNG
    d = X.shape[1]  # feature count

    direction = rng.normal(0.0, 1.0, size=d)  # random direction
    direction = direction / (np.linalg.norm(direction) + 1e-12)  # normalize

    x_poison = scale * direction  # extreme feature vector
    y_poison = y_scale  # extreme label

    X_p = np.vstack([X, x_poison])  # append poison sample
    y_p = np.append(y, y_poison)  # append poison label

    return X_p, y_p, x_poison, y_poison  # return poisoned dataset + point


def run_experiment(n=300, d=5, noise_std=0.5, seed=0, condition_mode="well"):
    X, y, true_w, true_b = make_data(
        n=n, d=d, noise_std=noise_std, seed=seed, condition_mode=condition_mode
    )

    w_cf, b_cf, yhat_cf = closed_form_pinv(X, y)  # closed-form solution
    mse_cf = mse(y, yhat_cf)  # closed-form MSE
    norm_w_cf = float(np.linalg.norm(w_cf))  # closed-form weight norm

    print("\n=== Closed-form (pinv) ===")
    print("mse_cf:", mse_cf)
    print("||w_cf||:", norm_w_cf)

    print("\n=== GD (no scaling) ===")
    w_gd_ns, b_gd_ns, yhat_gd_ns, loss_ns = gd_linear_regression(
        X, y, lr=0.05, epochs=10000, tol=1e-12, standardize=False, log_every=100
    )
    mse_gd_ns = mse(y, yhat_gd_ns)  # GD no-scale MSE

    print("\n=== GD (scaled) ===")
    w_gd_s, b_gd_s, yhat_gd_s, loss_s = gd_linear_regression(
        X, y, lr=0.2, epochs=10000, tol=1e-12, standardize=True, log_every=100
    )
    mse_gd_s = mse(y, yhat_gd_s)  # GD scaled MSE

    print("\n=== Train-set comparison ===")
    print("mse_cf:", mse_cf)
    print("mse_gd_no_scale:", mse_gd_ns)
    print("mse_gd_scaled  :", mse_gd_s)

    w_diff_ns = float(np.linalg.norm(w_cf - w_gd_ns))  # weight diff (no-scale GD)
    w_diff_s = float(np.linalg.norm(w_cf - w_gd_s))  # weight diff (scaled GD)

    y_diff_ns = float(np.linalg.norm(yhat_cf - yhat_gd_ns))  # prediction diff (no-scale GD)
    y_diff_s = float(np.linalg.norm(yhat_cf - yhat_gd_s))  # prediction diff (scaled GD)

    print("||w_cf - w_gd_no_scale||:", w_diff_ns)
    print("||w_cf - w_gd_scaled||  :", w_diff_s)
    print("||yhat_cf - yhat_gd_no_scale||:", y_diff_ns)
    print("||yhat_cf - yhat_gd_scaled||  :", y_diff_s)

    plt.figure()  # new plot
    plt.plot(loss_ns, label="GD (no scaling)")  # plot loss curve no scaling
    plt.plot(loss_s, label="GD (scaled)")  # plot loss curve scaled
    plt.axhline(mse_cf, linestyle="--", label="Closed-form loss")  # baseline line
    plt.xlabel("Epoch")  # x label
    plt.ylabel("MSE loss")  # y label
    plt.title(f"Loss curves | mode={condition_mode}")  # title
    plt.legend()  # legend
    plt.tight_layout()  # layout
    plt.show()  # show plot

    return {
        "X": X,
        "y": y,
        "w_cf": w_cf,
        "b_cf": b_cf,
        "mse_cf": mse_cf,
        "yhat_cf": yhat_cf,
        "w_gd_s": w_gd_s,
        "b_gd_s": b_gd_s,
    }


def run_poison_demo(base_cfg, poison_scale=30.0, poison_y=100.0):
    X = base_cfg["X"]  # base X
    y = base_cfg["y"]  # base y

    X_p, y_p, x_poison, y_poison = add_poison_point(
        X, y, scale=poison_scale, y_scale=poison_y
    )

    print("\n=== Poison point injected ===")
    print("x_poison (first 5):", np.round(x_poison[:5], 4))
    print("y_poison:", y_poison)

    w_cf_p, b_cf_p, _ = closed_form_pinv(X_p, y_p)  # closed-form on poisoned data
    w_gd_p, b_gd_p, _, _ = gd_linear_regression(
        X_p, y_p, lr=0.2, epochs=10000, tol=1e-12, standardize=True, log_every=200
    )

    X_val, y_val, _, _ = make_data(
        n=300, d=X.shape[1], noise_std=0.5, seed=999, condition_mode="well"
    )

    y_val_cf = X_val @ w_cf_p + b_cf_p  # val predictions CF poisoned
    y_val_gd = X_val @ w_gd_p + b_gd_p  # val predictions GD poisoned

    mse_val_cf = mse(y_val, y_val_cf)  # clean val MSE for CF
    mse_val_gd = mse(y_val, y_val_gd)  # clean val MSE for GD

    print("\n=== Clean validation MSE after poisoning ===")
    print("CF poisoned val MSE:", mse_val_cf)
    print("GD poisoned val MSE:", mse_val_gd)

    w_shift_cf = float(np.linalg.norm(base_cfg["w_cf"] - w_cf_p))  # parameter shift CF
    w_shift_gd = float(np.linalg.norm(base_cfg["w_gd_s"] - w_gd_p))  # parameter shift GD

    print("\n=== Parameter shift magnitude ===")
    print("||w_cf_before - w_cf_after||:", w_shift_cf)
    print("||w_gd_before - w_gd_after||:", w_shift_gd)


if __name__ == "__main__":
    cfg_well = run_experiment(n=300, d=5, noise_std=0.5, seed=0, condition_mode="well")
    cfg_ill = run_experiment(n=300, d=5, noise_std=0.5, seed=1, condition_mode="ill")

    run_poison_demo(cfg_well, poison_scale=30.0, poison_y=100.0)
    run_poison_demo(cfg_ill, poison_scale=30.0, poison_y=100.0)
