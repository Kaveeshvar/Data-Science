# logreg_from_scratch.py  # file name comment

import numpy as np  # import NumPy for arrays and math


class LogisticRegressionScratch:  # define our custom logistic regression class
    def __init__(  # constructor: runs when you create the model
        self,  # reference to this object
        lr=0.1,  # learning rate (step size)
        epochs=1000,  # number of passes over the data
        reg_lambda=0.0,  # L2 regularization strength (0 = no regularization)
        fit_intercept=True,  # whether to learn a bias term b
        standardize=True,  # whether to standardize features
        batch_size=None,  # mini-batch size (None = full batch GD)
        tol=1e-6,  # early stopping tolerance on loss improvement
        verbose=True,  # whether to print training logs
        seed=42,  # random seed for reproducibility (shuffling)
        clip_features=None,  # clip X values after standardization (defense)
        clip_grad_norm=None,  # clip gradient norm (defense)  
    ):
        self.lr = float(lr)  # store learning rate
        self.epochs = int(epochs)  # store number of epochs
        self.reg_lambda = float(reg_lambda)  # store L2 strength
        self.fit_intercept = bool(fit_intercept)  # store intercept option
        self.standardize = bool(standardize)  # store standardization option
        self.batch_size = batch_size  # store batch size
        self.tol = float(tol)  # store tolerance
        self.verbose = bool(verbose)  # store verbosity
        self.seed = int(seed)  # store RNG seed

        self.w = None  # weights (will be learned)
        self.b = 0.0  # bias (will be learned if fit_intercept=True)
        self.mu_ = None  # feature means for standardization
        self.sigma_ = None  # feature std devs for standardization

        self.history_ = []  # list of dicts for training logs
        self.clip_features = clip_features  # store feature clipping threshold
        self.clip_grad_norm = clip_grad_norm  # store grad clipping threshold

    def _check_X_y(self, X, y):  # validate and coerce X and y
        X = np.asarray(X, dtype=np.float64)  # convert X to float array
        y = np.asarray(y, dtype=np.float64).reshape(-1)  # convert y to 1D float array

        if X.ndim != 2:  # ensure X is 2D
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")  # error
        if y.ndim != 1:  # ensure y is 1D
            raise ValueError("y must be a 1D array of shape (n_samples,).")  # error
        if X.shape[0] != y.shape[0]:  # ensure same number of rows
            raise ValueError("X and y must have the same number of samples.")  # error

        unique = np.unique(y)  # get unique values in y
        if not np.all(np.isin(unique, [0.0, 1.0])):  # ensure y values are only 0/1
            raise ValueError("y must contain only 0 and 1.")  # error

        if not np.all(np.isfinite(X)):  # reject NaN/Inf in X
            raise ValueError("X contains NaN or Inf.")  # error
        if not np.all(np.isfinite(y)):  # reject NaN/Inf in y
            raise ValueError("y contains NaN or Inf.")  # error

        return X, y  # return validated arrays

    def _standardize_fit(self, X):  # compute standardization stats and apply
        self.mu_ = X.mean(axis=0)  # compute mean per feature
        self.sigma_ = X.std(axis=0)  # compute std per feature
        self.sigma_ = np.where(self.sigma_ == 0.0, 1.0, self.sigma_)  # avoid divide by zero
        Xs = (X - self.mu_) / self.sigma_  # standardize X
        return Xs  # return standardized X

    def _standardize_apply(self, X):  # apply standardization using stored stats
        Xs = (X - self.mu_) / self.sigma_  # standardize using training mean/std
        return Xs  # return standardized X

    def _sigmoid(self, z):  # stable sigmoid function
        z = np.asarray(z, dtype=np.float64)  # ensure float array
        out = np.empty_like(z)  # allocate output array

        pos = z >= 0  # mask for non-negative z (safe region)
        neg = ~pos  # mask for negative z (risk overflow region)

        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))  # normal sigmoid for z>=0
        ez = np.exp(z[neg])  # compute exp(z) for z<0 (safe because z is negative)
        out[neg] = ez / (1.0 + ez)  # rewritten sigmoid for z<0

        return out  # return probabilities

    def _loss_from_logits(self, z, y):  # stable BCE loss directly from logits
        # BCE(y,z) = log(1 + exp(z)) - y*z  (stable with logaddexp)
        per_sample = np.logaddexp(0.0, z) - y * z  # compute per-sample loss safely
        data_loss = per_sample.mean()  # average data loss

        if self.reg_lambda > 0.0:  # if L2 is enabled
            reg_loss = (self.reg_lambda / (2.0 * y.size)) * np.sum(self.w * self.w)  # (λ/2n)||w||^2
        else:  # if L2 is off
            reg_loss = 0.0  # no reg penalty

        return data_loss + reg_loss  # total loss

    def _forward(self, X):  # compute logits and probabilities
        z = X @ self.w  # compute Xw
        if self.fit_intercept:  # if we use bias term
            z = z + self.b  # add bias
        p = self._sigmoid(z)  # convert logits to probabilities
        return z, p  # return both logits and probabilities

    def _gradients(self, X, y, p):  # compute gradients for w and b
        n = y.size  # number of samples
        res = (p - y)  # derivative dL/dz for each sample (key identity)

        dw = (X.T @ res) / n  # gradient of weights from data term
        if self.reg_lambda > 0.0:  # if L2 is enabled
            dw = dw + (self.reg_lambda / n) * self.w  # add L2 gradient λ/n * w

        if self.fit_intercept:  # if bias term is used
            db = res.mean()  # gradient for bias is mean(res)
        else:  # if no bias
            db = 0.0  # no bias gradient

        return dw, db  # return gradients

    def _accuracy(self, y_true, y_pred):  # compute accuracy
        return float(np.mean(y_true == y_pred))  # fraction correct

    def _auc_roc(self, y_true, y_score):  # compute ROC-AUC without sklearn (optional)
        y_true = y_true.astype(np.int64)  # ensure ints
        order = np.argsort(-y_score)  # sort scores descending
        y_true_sorted = y_true[order]  # reorder true labels
        tp = np.cumsum(y_true_sorted == 1)  # cumulative true positives
        fp = np.cumsum(y_true_sorted == 0)  # cumulative false positives
        tp = tp.astype(np.float64)  # float for division
        fp = fp.astype(np.float64)  # float for division

        P = np.sum(y_true == 1)  # number of positives
        N = np.sum(y_true == 0)  # number of negatives
        if P == 0 or N == 0:  # AUC undefined if only one class
            return np.nan  # return NaN to signal undefined

        tpr = tp / P  # true positive rate
        fpr = fp / N  # false positive rate

        # trapezoidal integration over FPR to get AUC
        auc = np.trapezoid(tpr, fpr)  # compute area under curve
        return float(auc)  # return auc

    def fit(self, X, y):  # train model
        X, y = self._check_X_y(X, y)  # validate inputs
        rng = np.random.default_rng(self.seed)  # create RNG for shuffling

        if self.standardize:  # if standardization enabled
            X = self._standardize_fit(X)  # compute stats and transform
        else:  # if no standardization
            self.mu_ = np.zeros(X.shape[1], dtype=np.float64)  # dummy mean
            self.sigma_ = np.ones(X.shape[1], dtype=np.float64)  # dummy std
        if self.clip_features is not None:  # if feature clipping enabled
            X = np.clip(X, -float(self.clip_features), float(self.clip_features))  # clip training features

        n, d = X.shape  # get number of samples and features
        self.w = np.zeros(d, dtype=np.float64)  # initialize weights to zeros
        self.b = 0.0  # initialize bias to zero

        bs = self.batch_size  # local variable for batch size
        if bs is None or bs >= n:  # if full batch requested
            bs = n  # set batch size to full dataset

        prev_loss = np.inf  # track previous loss for early stopping

        for epoch in range(self.epochs):  # loop over epochs
            idx = rng.permutation(n)  # shuffle indices
            Xs = X[idx]  # shuffled X
            ys = y[idx]  # shuffled y

            # mini-batch loop
            for start in range(0, n, bs):  # iterate over batches
                end = start + bs  # end index for batch
                Xb = Xs[start:end]  # batch features
                yb = ys[start:end]  # batch labels

                z, p = self._forward(Xb)  # forward pass (logits + probs)
                dw, db = self._gradients(Xb, yb, p)  # compute gradients
                if self.clip_grad_norm is not None:  # if gradient clipping enabled
                    gnorm = float(np.linalg.norm(dw))  # compute gradient norm
                    if gnorm > float(self.clip_grad_norm):  # if too large
                        scale = float(self.clip_grad_norm) / (gnorm + 1e-12)  # scale factor
                        dw = dw * scale  # scale down dw
                        db = float(db) * scale  # scale db similarly


                self.w = self.w - self.lr * dw  # gradient descent update for weights
                if self.fit_intercept:  # update bias only if used
                    self.b = self.b - self.lr * db  # gradient descent update for bias

            # end-of-epoch evaluation on full training set
            z_all, p_all = self._forward(X)  # forward pass on all data
            loss = self._loss_from_logits(z_all, y)  # compute stable loss
            y_pred = (p_all >= 0.5).astype(np.float64)  # predictions using 0.5 threshold
            acc = self._accuracy(y, y_pred)  # compute accuracy
            grad_norm = float(np.linalg.norm(dw))  # last-batch grad norm (cheap proxy)
            auc = self._auc_roc(y, p_all)  # compute AUC (optional but useful)

            self.history_.append(  # store training logs
                {
                    "epoch": epoch,  # epoch number
                    "loss": float(loss),  # loss value
                    "acc": float(acc),  # accuracy
                    "grad_norm": float(grad_norm),  # gradient norm proxy
                    "auc": float(auc) if np.isfinite(auc) else np.nan,  # AUC or NaN
                }
            )

            if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):  # periodic logging
                print(  # print metrics
                    f"epoch={epoch:4d}  loss={loss:.6f}  acc={acc:.4f}  grad_norm={grad_norm:.4e}  auc={auc:.4f}"
                )

            # early stopping: stop if improvement is tiny
            if abs(prev_loss - loss) < self.tol:  # check if loss improvement is too small
                if self.verbose:  # if logging is on
                    print(f"Early stop at epoch={epoch} (loss improvement < tol)")  # print stop message
                break  # stop training
            prev_loss = loss  # update previous loss

        return self  # return fitted model

    def predict_proba(self, X):  # return probabilities for class 1
        X = np.asarray(X, dtype=np.float64)  # convert input to float array
        if X.ndim != 2:  # ensure 2D input
            raise ValueError("X must be 2D.")  # error
        if self.standardize:  # if standardization used in training
            X = self._standardize_apply(X)  # apply same transform
        if self.clip_features is not None:  # if clipping enabled
            X = np.clip(X, -float(self.clip_features), float(self.clip_features))  # clip inference features

        z, p = self._forward(X)  # compute logits and probs

        return p  # return probabilities

    def predict(self, X, threshold=0.5):  # return class predictions 0/1
        p = self.predict_proba(X)  # compute probabilities
        y_pred = (p >= float(threshold)).astype(np.int64)  # threshold to get 0/1
        return y_pred  # return predictions

    def gradient_check(self, X, y, eps=1e-5, num_checks=10):  # finite-diff gradient check
        X, y = self._check_X_y(X, y)  # validate X and y
        if self.standardize:  # standardize if configured
            X = self._standardize_fit(X)  # fit stats for this check run

        n, d = X.shape  # get shapes
        self.w = np.zeros(d, dtype=np.float64)  # init weights
        self.b = 0.0  # init bias

        z, p = self._forward(X)  # forward pass
        dw, db = self._gradients(X, y, p)  # analytic gradients

        rng = np.random.default_rng(self.seed)  # RNG for picking coordinates
        for _ in range(num_checks):  # do a few random checks
            j = int(rng.integers(0, d))  # choose random weight index

            w_old = self.w[j]  # store old weight value
            self.w[j] = w_old + eps  # w + eps
            z1, _ = self._forward(X)  # forward
            loss1 = self._loss_from_logits(z1, y)  # loss at w+eps

            self.w[j] = w_old - eps  # w - eps
            z2, _ = self._forward(X)  # forward
            loss2 = self._loss_from_logits(z2, y)  # loss at w-eps

            self.w[j] = w_old  # restore original weight

            grad_num = (loss1 - loss2) / (2.0 * eps)  # numerical gradient
            grad_ana = dw[j]  # analytic gradient

            rel_err = abs(grad_num - grad_ana) / (abs(grad_num) + abs(grad_ana) + 1e-12)  # relative error
            print(f"w[{j}]  grad_num={grad_num:.6e}  grad_ana={grad_ana:.6e}  rel_err={rel_err:.6e}")  # print check
