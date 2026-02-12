# Import NumPy (we use it for arrays, matrix multiplication, and math)
import numpy as np


# Define a class (a blueprint) for Linear Regression trained using Gradient Descent
class LinearRegressionGD:
    # __init__ runs when you create the object, e.g. model = LinearRegressionGD(...)
    def __init__(
        self,
        lr=0.01,                 # Learning rate (step size for gradient descent)
        epochs=2000,             # Maximum number of training iterations
        fit_intercept=True,      # If True, learn a bias term b
        standardize=False,       # If True, standardize features (mean 0, std 1)
        reg_lambda=0.0,          # L2 regularization strength (0.0 means no regularization)
        tol=1e-9,                # Early stopping threshold for loss improvement
        verbose=False,           # If True, prints training progress
        clip_grad=None,          # If set (e.g. 1.0), clip gradient to avoid explosions
        random_state=None        # Seed for reproducibility (optional)
    ):
        # Store hyperparameters inside the object (so fit() can use them)
        self.lr = float(lr)                      # Convert to float to avoid type issues
        self.epochs = int(epochs)                # Convert to int because epochs is a count
        self.fit_intercept = bool(fit_intercept) # Ensure boolean
        self.standardize = bool(standardize)     # Ensure boolean
        self.reg_lambda = float(reg_lambda)      # Convert to float
        self.tol = float(tol)                    # Convert to float
        self.verbose = bool(verbose)             # Convert to bool
        self.clip_grad = None if clip_grad is None else float(clip_grad)  # store clipping value
        self.random_state = random_state         # Store seed (can be None)

        # These will be learned during training
        self.w_ = None          # Weight vector (shape: d,)
        self.b_ = None          # Bias scalar (float)

        # These are used only if standardize=True
        self.mu_ = None         # Mean of each feature (shape: d,)
        self.sigma_ = None      # Std of each feature (shape: d,)

        # Store loss values over time for debugging/plotting
        self.loss_history_ = [] # List of loss per epoch

    # Helper: check that X and y have valid shapes
    def _check_shapes(self, X, y):
        # X should be 2D: (n, d)
        if X.ndim != 2:
            # If not 2D, raise an error and stop
            raise ValueError(f"X must be 2D (n,d). Got shape={X.shape}")

        # y should be 1D: (n,)
        if y.ndim != 1:
            # If not 1D, raise error
            raise ValueError(f"y must be 1D (n,). Got shape={y.shape}")

        # Number of rows in X must match number of elements in y
        if X.shape[0] != y.shape[0]:
            # If mismatch, raise error
            raise ValueError(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")

    # Helper: fit standardization parameters (mean/std) on training data, then standardize X
    def _standardize_fit(self, X):
        # Compute mean for each column/feature (shape: d,)
        mu = X.mean(axis=0)

        # Compute standard deviation for each column/feature (shape: d,)
        sigma = X.std(axis=0)

        # Small value to avoid division by zero
        eps = 1e-12

        # If any sigma is too close to 0 (constant feature), replace with 1.0
        sigma_safe = np.where(sigma < eps, 1.0, sigma)

        # Store mean and std so we can standardize future data in predict()
        self.mu_ = mu
        self.sigma_ = sigma_safe

        # Return standardized X: (X - mean) / std
        return (X - mu) / sigma_safe

    # Helper: apply stored standardization parameters to new X
    def _standardize_apply(self, X):
        # If mean/std are missing, it means fit() wasn’t called
        if self.mu_ is None or self.sigma_ is None:
            # Stop and tell the user
            raise RuntimeError("Standardization params not fitted. Call fit() first.")

        # Standardize using stored training mean/std
        return (X - self.mu_) / self.sigma_

    # Helper: compute loss (half MSE + optional L2 regularization)
    def _loss(self, y_hat, y, w):
        # n is number of samples
        n = y.shape[0]

        # residuals = prediction - truth
        res = y_hat - y

        # half-MSE = (1/2) * mean(res^2)
        mse_half = 0.5 * np.mean(res ** 2)

        # If regularization is enabled (lambda > 0)
        if self.reg_lambda > 0.0:
            # L2 penalty = (lambda/(2n)) * ||w||^2
            reg = (self.reg_lambda / (2.0 * n)) * np.dot(w, w)
            # Total loss = data loss + regularization loss
            return mse_half + reg

        # If no regularization, just return half-MSE
        return mse_half

    # Helper: compute gradients dw and db
    def _gradients(self, X, y, y_hat, w):
        # n is number of samples
        n = y.shape[0]

        # residuals = prediction - truth
        res = y_hat - y

        # Gradient wrt w: (X^T @ res) / n
        dw = (X.T @ res) / n

        # Gradient wrt b: mean(res)
        db = res.mean()

        # Add regularization gradient if enabled (do NOT regularize bias)
        if self.reg_lambda > 0.0:
            # dw += (lambda/n) * w
            dw = dw + (self.reg_lambda / n) * w

        # Return both gradients
        return dw, db

    # Optional helper: clip gradients to prevent exploding updates
    def _clip(self, dw, db):
        # If clip_grad is None, do nothing
        if self.clip_grad is None:
            return dw, db

        # Compute L2 norm of dw
        dw_norm = np.linalg.norm(dw)

        # If norm exceeds threshold, scale it down
        if dw_norm > self.clip_grad:
            # Scale factor = clip / norm
            dw = dw * (self.clip_grad / (dw_norm + 1e-12))

        # Clip db by value (simple safe guard)
        db = float(np.clip(db, -self.clip_grad, self.clip_grad))

        # Return clipped gradients
        return dw, db

    # Fit the model parameters w and b on training data
    def fit(self, X, y):
        # Convert X to a float NumPy array
        X = np.asarray(X, dtype=float)

        # Convert y to float NumPy array and force shape (n,)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Validate shapes
        self._check_shapes(X, y)

        # If standardize=True, fit standardization and transform X
        if self.standardize:
            X_train = self._standardize_fit(X)
        else:
            # Otherwise, use X as-is
            X_train = X

        # Get n (samples) and d (features)
        n, d = X_train.shape

        # Create RNG (random generator) for reproducibility if needed
        rng = np.random.default_rng(self.random_state)

        # Initialize weights (zeros is fine for linear regression)
        self.w_ = np.zeros(d, dtype=float)

        # Initialize bias
        self.b_ = 0.0

        # Reset loss history
        self.loss_history_ = []

        # Set previous loss to infinity so early stopping works
        prev_loss = np.inf

        # Training loop
        for epoch in range(self.epochs):
            # Compute predictions: y_hat = Xw + b (if intercept enabled)
            y_hat = X_train @ self.w_ + (self.b_ if self.fit_intercept else 0.0)

            # Compute loss value
            loss = self._loss(y_hat, y, self.w_)

            # If loss is NaN or Inf, stop (divergence)
            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"Loss became {loss}. Likely divergence: lower lr, standardize=True, or enable clip_grad."
                )

            # Compute gradients
            dw, db = self._gradients(X_train, y, y_hat, self.w_)

            # If not fitting intercept, force db=0 (so b doesn't change)
            if not self.fit_intercept:
                db = 0.0

            # Clip gradients if enabled
            dw, db = self._clip(dw, db)

            # Update weights using gradient descent
            self.w_ -= self.lr * dw

            # Update bias using gradient descent
            self.b_ -= self.lr * db

            # Save loss for analysis
            self.loss_history_.append(loss)

            # Print progress sometimes if verbose=True
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(
                    f"epoch={epoch:5d} loss={loss:.6e} ||dw||={np.linalg.norm(dw):.3e} db={db:.3e}"
                )

            # Early stopping: if loss improvement is tiny, stop training
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Early stop at epoch={epoch} (Δloss<{self.tol})")
                break

            # Update prev_loss for next iteration
            prev_loss = loss

        # Return the model object (common sklearn-style)
        return self

    # Predict outputs for new data X
    def predict(self, X):
        # Ensure the model was trained
        if self.w_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        # Convert input X to float array
        X = np.asarray(X, dtype=float)

        # Validate that X is 2D
        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape={X.shape}")

        # Apply standardization if it was used in training
        if self.standardize:
            X = self._standardize_apply(X)

        # Return predictions: Xw + b (if intercept enabled)
        return X @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
