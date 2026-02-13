# Import math so we can use exp safely and predictably.
import math

# Define a numerically-stable sigmoid function.
def stable_sigmoid(z: float) -> float:
    # If z is non-negative, we compute sigmoid in the standard stable way.
    if z >= 0.0:
        # Compute exp(-z) which is safe when z is large positive.
        ez = math.exp(-z)
        # Return 1 / (1 + exp(-z)) which is stable in this branch.
        return 1.0 / (1.0 + ez)
    # If z is negative, we use an alternate stable form to avoid overflow.
    else:
        # Compute exp(z) which is safe when z is large negative.
        ez = math.exp(z)
        # Return exp(z) / (1 + exp(z)) which is stable in this branch.
        return ez / (1.0 + ez)
