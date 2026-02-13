# verify_harden_attack_demo.py  # script name comment

import numpy as np  # import NumPy
import matplotlib.pyplot as plt  # plotting
from sklearn.linear_model import LogisticRegression  # sklearn baseline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss  # metrics
from logreg_from_scratch import LogisticRegressionScratch  # your model


def make_two_gaussians(n=2000, seed=0):  # synthetic dataset generator
    rng = np.random.default_rng(seed)  # RNG

    n0 = n // 2  # class 0 count
    n1 = n - n0  # class 1 count

    mean0 = np.array([-2.0, -2.0])  # class 0 mean
    mean1 = np.array([2.0, 2.0])  # class 1 mean
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])  # covariance

    X0 = rng.multivariate_normal(mean0, cov, size=n0)  # class 0 samples
    X1 = rng.multivariate_normal(mean1, cov, size=n1)  # class 1 samples

    X = np.vstack([X0, X1])  # stack features
    y = np.hstack([np.zeros(n0), np.ones(n1)])  # stack labels

    idx = rng.permutation(n)  # shuffle index
    X = X[idx]  # shuffle X
    y = y[idx]  # shuffle y

    return X, y  # return


def train_test_split(X, y, test_size=0.25, seed=0):  # simple split
    rng = np.random.default_rng(seed)  # RNG
    n = X.shape[0]  # samples
    idx = rng.permutation(n)  # permute
    n_test = int(n * test_size)  # test count
    test_idx = idx[:n_test]  # test idx
    train_idx = idx[n_test:]  # train idx
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]  # split


def standardize_fit(X):  # standardize helper for sklearn comparability
    mu = X.mean(axis=0)  # mean
    sigma = X.std(axis=0)  # std
    sigma = np.where(sigma == 0.0, 1.0, sigma)  # avoid divide by zero
    return (X - mu) / sigma, mu, sigma  # standardized + params


def standardize_apply(X, mu, sigma):  # apply standardization
    return (X - mu) / sigma  # return standardized


def reliability_diagram(y_true, p, n_bins=10):  # calibration plot data
    y_true = y_true.astype(np.float64)  # ensure float
    p = np.clip(p, 1e-12, 1 - 1e-12)  # clip to avoid edge weirdness
    bins = np.linspace(0.0, 1.0, n_bins + 1)  # bin edges
    ids = np.digitize(p, bins) - 1  # bin indices

    bin_acc = []  # average true rate per bin
    bin_conf = []  # average predicted prob per bin
    bin_count = []  # counts

    for b in range(n_bins):  # loop bins
        mask = ids == b  # samples in bin
        if np.sum(mask) == 0:  # skip empty bin
            continue  # continue
        bin_acc.append(np.mean(y_true[mask]))  # empirical accuracy in bin
        bin_conf.append(np.mean(p[mask]))  # mean confidence in bin
        bin_count.append(np.sum(mask))  # count

    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count)  # return arrays


def poisoning_attack(X_train, y_train, k=10, scale=25.0, target_label=1, seed=0):  # craft poisoned points
    rng = np.random.default_rng(seed)  # RNG
    n, d = X_train.shape  # shapes

    base = X_train[rng.integers(0, n, size=k)]  # pick k real points as template
    direction = rng.normal(size=(k, d))  # random directions
    direction = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12)  # normalize

    Xp = base + scale * direction  # push them far away (high norm = big gradient impact)
    yp = np.full(k, float(target_label))  # set labels to desired target

    # classic poison: make them contradictory (mislabeled) to create damage
    # if target_label=1, these points try to pull model toward predicting 1 in weird regions

    X_poisoned = np.vstack([X_train, Xp])  # append poison features
    y_poisoned = np.hstack([y_train, yp])  # append poison labels

    return X_poisoned, y_poisoned, Xp, yp  # return all


def evasion_attack_fgsm_like(x, w, b, y_true, eps=0.5):  # simple white-box evasion step
    # For logistic regression, increasing logit z = w·x + b pushes probability toward 1
    # To flip a 0 -> 1, move x in +w direction; to flip 1 -> 0, move x in -w direction

    z = float(x @ w + b)  # compute logit
    p = 1.0 / (1.0 + np.exp(-z))  # sigmoid (safe enough for one scalar)
    direction = w / (np.linalg.norm(w) + 1e-12)  # normalized weight direction

    if int(y_true) == 0:  # if true class is 0
        x_adv = x + eps * direction  # push toward class 1
    else:  # if true class is 1
        x_adv = x - eps * direction  # push toward class 0

    return x_adv, p  # return adversarial sample and original prob


def evaluate(name, y_true, p):  # evaluation helper
    y_hat = (p >= 0.5).astype(np.int64)  # predictions
    acc = float(np.mean(y_hat == y_true))  # accuracy
    auc = float(roc_auc_score(y_true, p))  # AUC
    p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)  # avoid log(0) with sklearn
    ll = float(log_loss(y_true, p_safe))  # log loss
    brier = float(brier_score_loss(y_true, p))  # Brier score (calibration-ish)
    print(f"{name:20s} | acc={acc:.4f} auc={auc:.4f} logloss={ll:.4f} brier={brier:.4f}")  # print
    return acc, auc, ll, brier  # return


def main():  # main runner
    X, y = make_two_gaussians(n=4000, seed=1)  # make dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=2)  # split

    # --- Train your scratch model (clean) ---
    scratch = LogisticRegressionScratch(  # create scratch model
        lr=0.1,  # lr
        epochs=800,  # epochs
        reg_lambda=1.0,  # L2
        fit_intercept=True,  # bias
        standardize=True,  # standardize
        batch_size=128,  # mini-batch
        tol=1e-8,  # stop if tiny improvement
        verbose=False,  # reduce prints in demo
        seed=42,  # seed
        # clip_features=5.0,  # enable if you added patch
        # clip_grad_norm=5.0,  # enable if you added patch
    )

    print("\n[1] Finite-difference gradient check (tiny) 🔎")  # section header
    X_small, y_small = make_two_gaussians(n=12, seed=3)  # tiny data
    scratch.gradient_check(X_small, y_small, eps=1e-5, num_checks=6)  # check

    print("\n[2] Train scratch (clean) 🧱")  # section header
    scratch.fit(X_train, y_train)  # fit

    p_test_scratch = scratch.predict_proba(X_test)  # probs
    evaluate("scratch_clean", y_test, p_test_scratch)  # metrics

    # --- Train sklearn baseline on comparable preprocessing ---
    # We standardize explicitly so sklearn sees the same scale as scratch
    X_train_s, mu, sigma = standardize_fit(X_train)  # standardize train
    X_test_s = standardize_apply(X_test, mu, sigma)  # standardize test

    sk = LogisticRegression(  # sklearn model
        C=1.0,  # inverse of regularization strength (not same as lambda, but ok baseline)
        solver="lbfgs",  # optimizer
        max_iter=2000,  # iterations
    )

    print("\n[3] Train sklearn baseline 🧪")  # section header
    sk.fit(X_train_s, y_train)  # fit sklearn
    p_test_sk = sk.predict_proba(X_test_s)[:, 1]  # sklearn probs
    evaluate("sklearn_clean", y_test, p_test_sk)  # metrics

    # Compare weight direction (cosine similarity)
    w_scratch = scratch.w.copy()  # scratch weights
    w_sk = sk.coef_.reshape(-1).copy()  # sklearn weights
    cos = float((w_scratch @ w_sk) / ((np.linalg.norm(w_scratch) + 1e-12) * (np.linalg.norm(w_sk) + 1e-12)))  # cosine
    print(f"\nWeight direction cosine similarity (scratch vs sklearn): {cos:.4f}")  # print

    # --- Poisoning demo ---
    print("\n[4] Poisoning attack demo (training-time) ☠️")  # section header
    X_pois, y_pois, Xp, yp = poisoning_attack(X_train, y_train, k=25, scale=40.0, target_label=1, seed=7)  # poison

    scratch_pois = LogisticRegressionScratch(  # poisoned model
        lr=0.1,  # lr
        epochs=800,  # epochs
        reg_lambda=0.0,  # turn OFF L2 to show damage clearly
        fit_intercept=True,  # bias
        standardize=True,  # standardize
        batch_size=128,  # mini-batch
        tol=1e-8,  # tol
        verbose=False,  # quiet
        seed=42,  # seed
    )

    scratch_pois.fit(X_pois, y_pois)  # fit on poisoned data
    p_test_pois = scratch_pois.predict_proba(X_test)  # test probs
    evaluate("scratch_poisoned", y_test, p_test_pois)  # metrics

    # Hardened training against poison: L2 + feature clipping + grad clipping (if you enabled patch)
    print("\n[5] Poisoning defenses (L2 + optional clipping) 🛡️")  # section header
    scratch_def = LogisticRegressionScratch(  # defended model
        lr=0.1,  # lr
        epochs=800,  # epochs
        reg_lambda=0.1,  # moderate L2
        fit_intercept=True,  # bias
        standardize=True,  # standardize
        batch_size=128,  # mini-batch
        tol=1e-8,  # tol
        verbose=False,  # quiet
        seed=42,  # seed
        clip_features=8.0,  # clip standardized features
        clip_grad_norm=10.0,  # clip gradient norm
    )

    scratch_def.fit(X_pois, y_pois)  # fit defended
    p_test_def = scratch_def.predict_proba(X_test)  # test probs
    evaluate("scratch_defended", y_test, p_test_def)  # metrics

    # --- Evasion demo ---
    print("\n[6] Evasion attack demo (test-time) 🏃‍♂️💨")  # section header
    rng = np.random.default_rng(0)  # RNG
    idxs = rng.integers(0, X_test.shape[0], size=200)  # pick 200 random test points
    X_sub = X_test[idxs].copy()  # subset X
    y_sub = y_test[idxs].copy()  # subset y

    p_before = scratch.predict_proba(X_sub)  # probs before
    y_before = (p_before >= 0.5).astype(np.int64)  # preds before
    acc_before = float(np.mean(y_before == y_sub))  # acc before
    print(f"Subset accuracy before evasion: {acc_before:.4f}")  # print

    X_adv = []  # collect adversarial samples
    for i in range(X_sub.shape[0]):  # loop points
        x = X_sub[i]  # point
        y0 = y_sub[i]  # label
        x_adv, _ = evasion_attack_fgsm_like(x, scratch.w, scratch.b, y0, eps=0.8)  # adversarial move
        X_adv.append(x_adv)  # store
    X_adv = np.vstack(X_adv)  # stack

    p_after = scratch.predict_proba(X_adv)  # probs after
    y_after = (p_after >= 0.5).astype(np.int64)  # preds after
    acc_after = float(np.mean(y_after == y_sub))  # acc after
    print(f"Subset accuracy after evasion:  {acc_after:.4f}")  # print

    # --- Calibration check (reliability diagram) ---
    print("\n[7] Calibration check (reliability diagram + Brier) 📏")  # section header
    conf, acc, cnt = reliability_diagram(y_test, p_test_scratch, n_bins=10)  # compute bins

    plt.figure()  # new figure
    plt.plot([0, 1], [0, 1])  # perfect calibration line
    plt.scatter(conf, acc, s=np.clip(cnt, 10, 400))  # plot bins sized by count
    plt.xlabel("Mean predicted probability")  # x label
    plt.ylabel("Empirical fraction of positives")  # y label
    plt.title("Reliability diagram (scratch_clean)")  # title
    plt.show()  # show

    # --- Loss curves for visual sanity ---
    losses_clean = [h["loss"] for h in scratch.history_]  # extract clean losses
    losses_pois = [h["loss"] for h in scratch_pois.history_]  # extract poisoned losses
    losses_def = [h["loss"] for h in scratch_def.history_]  # extract defended losses

    plt.figure()  # new figure
    plt.plot(losses_clean, label="clean")  # plot clean
    plt.plot(losses_pois, label="poisoned")  # plot poisoned
    plt.plot(losses_def, label="defended")  # plot defended
    plt.xlabel("epoch")  # x label
    plt.ylabel("loss")  # y label
    plt.title("Training loss curves")  # title
    plt.legend()  # legend
    plt.show()  # show


if __name__ == "__main__":  # entry point
    main()  # run
