# demo_logreg.py  # file name comment

import numpy as np  # import NumPy
import matplotlib.pyplot as plt  # import matplotlib for plotting
from logreg_from_scratch import LogisticRegressionScratch  # import our model


def make_two_gaussians(n=1000, seed=0):  # function to generate simple separable-ish data
    rng = np.random.default_rng(seed)  # RNG for reproducibility

    n0 = n // 2  # number of class 0 samples
    n1 = n - n0  # number of class 1 samples

    mean0 = np.array([-2.0, -2.0])  # mean for class 0
    mean1 = np.array([2.0, 2.0])  # mean for class 1

    cov = np.array([[1.0, 0.3], [0.3, 1.0]])  # shared covariance matrix

    X0 = rng.multivariate_normal(mean0, cov, size=n0)  # sample class 0 points
    X1 = rng.multivariate_normal(mean1, cov, size=n1)  # sample class 1 points

    X = np.vstack([X0, X1])  # stack features
    y = np.hstack([np.zeros(n0), np.ones(n1)])  # stack labels (0 then 1)

    idx = rng.permutation(n)  # shuffle indices
    X = X[idx]  # shuffle X
    y = y[idx]  # shuffle y

    return X, y  # return dataset


def train_test_split(X, y, test_size=0.2, seed=0):  # simple train/test split
    rng = np.random.default_rng(seed)  # RNG
    n = X.shape[0]  # number of samples
    idx = rng.permutation(n)  # permute indices
    n_test = int(n * test_size)  # number of test samples
    test_idx = idx[:n_test]  # test indices
    train_idx = idx[n_test:]  # train indices
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]  # split


def main():  # main function
    X, y = make_two_gaussians(n=2000, seed=1)  # generate dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=2)  # split

    model = LogisticRegressionScratch(  # create model instance
        lr=0.1,  # learning rate
        epochs=1000,  # max epochs
        reg_lambda=1.0,  # L2 strength (helps prevent weight blow-up)
        fit_intercept=True,  # use bias
        standardize=True,  # standardize features
        batch_size=128,  # mini-batch GD
        tol=1e-8,  # early stopping tolerance
        verbose=True,  # print logs
        seed=42,  # reproducibility
    )

    print("\nRunning gradient check on tiny data...")  # message
    X_small, y_small = make_two_gaussians(n=10, seed=3)  # tiny dataset for gradient check
    model.gradient_check(X_small, y_small, eps=1e-5, num_checks=5)  # check gradients

    print("\nTraining model...")  # message
    model.fit(X_train, y_train)  # fit model on training data

    p_test = model.predict_proba(X_test)  # predicted probabilities
    y_pred = model.predict(X_test, threshold=0.5)  # predicted labels

    acc = float(np.mean(y_pred == y_test))  # compute test accuracy
    auc = model._auc_roc(y_test, p_test)  # compute test AUC

    print(f"\nTest accuracy: {acc:.4f}")  # print accuracy
    print(f"Test AUC:      {auc:.4f}")  # print AUC

    losses = [h["loss"] for h in model.history_]  # extract loss history
    plt.figure()  # create new figure
    plt.plot(losses)  # plot loss curve
    plt.xlabel("epoch")  # x label
    plt.ylabel("loss")  # y label
    plt.title("Training loss curve")  # title
    plt.show()  # show plot


if __name__ == "__main__":  # run main only if script executed directly
    main()  # call main
