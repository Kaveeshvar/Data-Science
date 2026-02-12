import numpy as np
from linear_regression_gd import LinearRegressionGD

# Create fake data: y = 3x + 5 + noise
np.random.seed(0)
X = np.random.randn(100, 1)
y = 3 * X[:, 0] + 5 + 0.1 * np.random.randn(100)

model = LinearRegressionGD(lr=0.1, epochs=2000, fit_intercept=True, standardize=True, verbose=True)
model.fit(X, y)

print("w:", model.w_)
print("b:", model.b_)

pred = model.predict(X[:5])
print("first 5 preds:", pred)
