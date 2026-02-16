# Term Meanings

## What is gradient in linear regression
Gradient is the **slope number** 📈

It tells how much **Y changes** when **X changes a little bit**.

Big gradient = steep line.  
Small gradient = flat line.

---

## Convexity
Convexity means the curve is shaped like a **bowl 🥣**

If you stand anywhere, you will always roll down to the **lowest point**.

---

## Rank Deficiency
Rank deficiency means some columns are **copying each other 🤝**

So they don’t give new information.

It’s like asking the same question twice.

---

## GD
GD (Gradient Descent) is a way to **find the lowest point ⛰️➡️⬇️**

You take small steps downhill  
until you reach the **bottom (best answer)**.

---

## Why does gradient descent converge slowly or diverge when features aren’t scaled? Explain geometrically
When features aren’t scaled, the loss surface becomes a **long skinny oval 🥚** instead of a nice round bowl 🥣

So gradient descent **zig-zags** across the valley instead of going straight down.

If one direction is very steep and the other is flat:

- Small step → very slow 😴  
- Big step → jumps across and may explode 💥

Scaling makes the bowl round → smooth, fast convergence.

---

## local minima
Local minima is a **small valley ⛰️**

You reach the lowest point nearby…

But there might be a **deeper valley somewhere else**.

---

## Closed-form (Normal Equation / Pseudoinverse)
Closed-form means you get the answer **in one big formula 🧮**

No small steps. No iteration.

For linear regression:  
[
w = (X^\top X)^{-1} X^\top y
]

It directly gives the **best weights**.

---

## Closed-form (Normal Equation / Pseudoinverse) vs Gradient Descent (GD)
**Closed-form** 🧮  
One big formula → answer in one shot.  
Fast for small data. Slow or impossible for huge data.

**Gradient Descent** ⛰️⬇️  
Takes small steps to reach answer.  
Works well for big data. Needs many steps.

---

## convergence
Convergence means **getting closer and closer to the final answer 🎯**

Steps become tiny…  
Changes become very small…  
Then you basically stop moving.

---

## XTX
(X^\top X) means **multiply X’s transpose with X 🔄✖️**

It measures how features relate to each other.

In linear regression, it helps compute the **best weights**.

---

## Explain why GD converges slowly when XtX is ill-conditioned. What does feature scaling do geometrically?
When (X^\top X) is ill-conditioned, the bowl is **very stretched 🥚**

One side is steep, one side is flat.  
So GD **zig-zags slowly** across the valley instead of going straight down.

Feature scaling makes the bowl **rounder 🥣**

Now all directions are similar → GD goes smoothly and faster.

---

## When would you prefer pseudoinverse/normal equation vs GD/SGD in production? Consider compute, memory, and numerical stability.
**Use Normal Equation / Pseudoinverse 🧮**

- Small dataset  
- Few features  
- Enough memory  
- Want exact solution in one shot

**Use GD / SGD ⛰️⬇️**

- Huge dataset  
- Many features  
- Memory is limited  
- Need scalable & stable training

Big data → GD.  
Small clean data → Closed-form.

---

## Why is MSE regression vulnerable to outliers/poisoning, and what defenses would you implement in a real pipeline?
**Why vulnerable?** ⚠️  
MSE **squares the error**.  
Big error → huge square → model bends too much toward outliers.

So one bad point can pull the line strongly.

**Defenses in real pipeline 🛡️**

- Remove / clip extreme values  
- Use robust loss (MAE, Huber)  
- Scale features  
- Add regularization  
- Monitor data for anomalies

Don’t let one crazy point control everything.

---

## Weights (w)
Weights are **importance numbers 🎛️**

They tell how much each input affects the output.

Big weight = strong effect.  
Small weight = weak effect.

---

## bias
Bias is the **starting value 🎯**

It shifts the line up or down,  
even when inputs are zero.

---

## mu
μ (mu) means **mean (average) 📊**

Add all values,  
divide by how many there are.

---

## sigma
σ (sigma) means **standard deviation 📏**

It tells how much values **spread out** from the average.

Small σ = close together.  
Big σ = far apart.

---

## artifact
Artifact is an **extra unwanted thing 🧩**

It appears because of the process,  
not because it’s truly part of the data.

---

## Pydantic
Pydantic checks and cleans your data 🧼

It makes sure the input is the **right type and format** before using it.

---

## Gradient
Gradient is the **direction to move 📍**

It tells which way goes up fastest —  
so we move the opposite way to go down.

---

## Log-loss
Log-loss measures **how wrong your probability is 📉**

If you’re very confident and wrong → big punishment.  
If you’re confident and correct → small loss.

---

## Bernoulli likelihood
Bernoulli likelihood is the **chance of seeing 0 or 1 🎯**

If outcome = 1 → probability = p  
If outcome = 0 → probability = 1 − p

It measures how likely your prediction matches a yes/no result.

---

## grad_norm, auc
**grad_norm** 📏  
Size of the gradient.  
Big = big step.  
Small = almost done.

**AUC** 🎯  
How well model separates 0 and 1.  
1 = perfect.  
0.5 = random guessing.

---

## Why can logistic regression weights blow up on linearly separable data, and how does L2 fix it?
If data is perfectly separable 🎯

Logistic regression keeps increasing weights  
to make probabilities closer to 0 and 1.

Bigger weights → smaller loss → so it keeps growing  
and can go to infinity 🚀

**L2 regularization** adds a penalty for big weights 🧱

So large weights become costly →  
model stops growing → stays stable.

---

## Roc, AUC
**ROC** 📈  
A curve showing how well the model separates 0 and 1  
by plotting True Positive Rate vs False Positive Rate.

**AUC** 🎯  
Area under that curve.  
1 = perfect.  
0.5 = random guessing.

---

## Threshold
Threshold is the **cut-off line ✂️**

If probability ≥ threshold → predict 1  
If probability < threshold → predict 0

It decides when to say “yes” or “no.”

---

## f1
F1 is a **balance score ⚖️**

It combines precision and recall.

High F1 = model is good at  
finding positives **and** not making many mistakes.

---

## Fit and transform
**Fit** 🧠  
Learn from the data (like finding mean or rules).

**Transform** 🔄  
Use what was learned to change the data.

Fit = learn.  
Transform = apply.

---

## Standardization (Z-score)
Standardization (Z-score) 📏

Subtract the mean (μ)  
Divide by standard deviation (σ)

So data has:  
Mean = 0  
Std = 1

It makes everything on the same scale.

---

## Min-Max scaling
Min-Max scaling 📊

Take smallest value → make it 0  
Take biggest value → make it 1

All other values are squeezed between 0 and 1.

---

## Robust scaling (median/IQR)
Robust scaling 📦

Subtract the **median**  
Divide by **IQR** (spread of middle 50%)

It ignores extreme outliers,  
so big weird values don’t affect it much.

---

## clipping
Clipping ✂️

If a value is too big → cut it to a max limit.  
If too small → raise it to a min limit.

It stops extreme values from going wild.

---

## Log transform / power transforms
**Log transform / power transform** 🔄

They shrink big values more than small ones.

Helps make data less skewed  
and more nicely shaped.

---

## variance_inflation_factor
Variance Inflation Factor (VIF) 📏

It tells how much a feature is **copied by other features**.

Big VIF = lots of overlap 😬  
Small VIF = mostly independent 👍

---

## Residuals
Residuals 📉

The **difference between real value and predicted value**.

Residual = Actual − Predicted.