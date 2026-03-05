# Decision Tree

Simplest definition
A Decision Tree is a flowchart-like model that makes predictions by asking a sequence of if–then questions about the input features.

Classification example (spam detection):

“Does the email contain the word ‘free’?”

If yes: “Does it contain many links?”

If yes → predict spam; else → maybe not spam.

Regression example (house price):

“Is the area > 1200 sq ft?”

If yes: “Is the neighborhood school rating > 8?”

Follow the path to a leaf that outputs a numeric price estimate.

## How It Works Fundamentally

A decision tree is a **piecewise-constant** function over feature space built by recursive partitioning using greedy search.

### Predictions at Leaf Nodes

- **Classification:** Returns class distribution or majority class
- **Regression:** Returns mean, median, or fitted simple model

### Key Insight

A single decision tree can learn nonlinear patterns without explicit feature engineering by automatically discovering feature interactions through splits.

## Why Use Decision Trees?

- **Nonlinear by default:** Learns complex patterns without manual feature engineering
- **Mixed data types:** Handles numeric and categorical features naturally
- **Interpretable:** Produces human-readable decision rules
- **Fast inference:** Only $O(\log n)$ depth comparisons needed per prediction
- **Robust:** Thresholds adapt to monotonic transformations of features
- **No scaling required:** Tree splits are scale-invariant

## Impurity Measures

### Gini Impurity

$$\text{Gini} = 1 - \sum_{i=1}^{c} p_i^2$$

- Probability of misclassification if you randomly sample a label from the node's distribution
- Range: 0 (pure) to 0.5 (maximum impurity for binary classification)

### Entropy (Information Entropy)

$$\text{Entropy} = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

- Measures theoretical uncertainty in the node
- Range: 0 (pure) to $\log_2(c)$ (maximum uncertainty)

### Why Gini/Entropy Over Misclassification Error?

- More sensitive to class distribution changes
- Better at selecting splits that separate classes
- Lead to more balanced, better-generalizing trees

## Finding the Best Split

### Algorithm for Numeric Features

For a numeric feature $x_j$:

1. **Sort** samples by $x_j$
2. **Generate candidate thresholds** at midpoints between consecutive unique values
3. **Compute information gain** $\Delta I$ for each split efficiently using prefix counts
4. **Select** the split with maximum $\Delta I$

### Information Gain

$$\Delta I = I(\text{parent}) - \frac{n_L}{n} I(\text{left}) - \frac{n_R}{n} I(\text{right})$$

Where $I$ is impurity (Gini or Entropy), $n_L$, $n_R$ are left/right child sizes.

## Implementation Steps (CART-Style)

### 1. Define Impurity Function

- **Classification:** Gini or Entropy using class counts
- **Regression:** Variance using sample sums and sum-of-squares

### 2. Node Building (Recursive)

**Input:** Dataset indices at current node

1. Compute node impurity
2. Check stopping conditions (return if met)
3. For each feature:
   - Enumerate candidate splits
   - Compute information gain
4. Choose best split (highest gain)
5. Partition data and recurse on children

### 3. Stopping Rules (Must Implement)

- `max_depth` reached
- Node impurity is zero (pure node)
- Number of samples $|D| < $ `min_samples_split`
- Child(ren) would violate `min_samples_leaf`
- Information gain $\Delta I < $ `min_impurity_decrease`

### 4. Leaf Prediction

- **Classification:** Store class counts and probability vector
- **Regression:** Store mean, standard deviation, and quantiles (for intervals)

### 5. Pruning (Optional but Recommended for Production)

1. Build full tree subject to basic constraints
2. Apply **cost-complexity pruning** via validation or cross-validation
3. Select $\alpha$ (complexity parameter) that minimizes validation error
4. Reduces overfitting and improves generalization

## Common Hyperparameters

| Parameter               | Effect                      | Default          |
| ----------------------- | --------------------------- | ---------------- |
| `max_depth`             | Limits tree depth           | None (unlimited) |
| `min_samples_split`     | Min samples to split a node | 2                |
| `min_samples_leaf`      | Min samples in leaf         | 1                |
| `min_impurity_decrease` | Min gain to split           | 0                |
| `criterion`             | Impurity measure            | 'gini'           |
| `splitter`              | 'best' or 'random'          | 'best'           |

## Tips for Production

- Always use `max_depth` to prevent overfitting
- Set `min_samples_leaf ≥ 10` for stable predictions on new data
- Use pruning via cross-validation for better generalization
- Shallow trees (depth 3-5) often outperform deep trees in practice
- Ensemble methods (Random Forest, Gradient Boosting) fix single tree weaknesses

# Youtube :

## Tree Structure Components

- **Root Node:** The very top of the tree where the first decision is made
- **Internal Nodes/Branches:** Nodes with arrows pointing both toward and away from them
- **Leaf Nodes:** Terminal nodes where final predictions are made

## Building a Classification Tree

### Process Overview

To build a tree, determine which feature best splits the data using **Gini Impurity**.

### Steps

1. **Calculate Impurity for Each Feature**
   - Create a candidate tree for each feature
   - Gini Impurity Formula:
     $$\text{Gini} = 1 - (p_{\text{yes}})^2 - (p_{\text{no}})^2$$
   - Total Gini Impurity: Weighted average of leaf impurities

2. **Handle Numeric Data**
   - Sort data by the numeric value
   - Calculate averages of adjacent values to create candidate thresholds
   - Calculate Gini Impurity for each threshold
   - Pick the threshold with lowest impurity

3. **Select Root**
   - Feature with lowest total Gini Impurity becomes the root

4. **Repeat Recursively**
   - Continue splitting until leaves are pure or stopping criteria are met

## Preventing Overfitting

### Problem

Trees that fit training data perfectly often perform poorly on new data (high variance).

### Solutions

- **Pruning:** Remove tree parts that provide little classification power
- **Growth Limits:** Set minimum samples required in leaf nodes (commonly 10-20)
- **Cross-Validation:** Test different limits to find optimal parameters
- **Min Impurity Decrease:** Require significant impurity reduction to justify splits

## Feature Selection

Feature selection happens **automatically** based on impurity reduction:

- Features that don't reduce impurity are not included in the tree
- Set thresholds for minimum impurity decrease to prevent overfitting
- Results in simpler, more robust trees

## Handling Missing Data

### Categorical Data

- **Most Common:** Assign the most frequent category value
- **Correlation-Based:** Use correlated features to impute values
  - Example: Use "chest pain" to fill missing "blocked arteries" data

### Numerical Data

- **Mean/Median:** Replace with average or middle value of the column
- **Linear Regression:** Use correlated features to predict missing values
  - Example: Use height to predict missing weight values

---

## Regression Trees

### Key Differences from Classification

- **Purpose:** Predict continuous numeric values instead of categories
- **Leaf Prediction:** Returns the **mean** of observations in that region
- **Advantage:** Handles non-linear relationships without feature engineering

### Building a Regression Tree

Use **Sum of Squared Residuals (SSR)** instead of Gini Impurity:

$$\text{SSR} = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

### Algorithm

1. **For each candidate threshold:**
   - Calculate average value for observations on each side
   - Compute residual (observed - predicted) for each point
   - Square residuals and sum them

2. **Select Best Split:**
   - Test all possible thresholds (midpoints of adjacent values)
   - Choose the split with **lowest SSR**

3. **Multiple Predictors:**
   - Find best threshold for each variable separately
   - Compare SSR values across all variables
   - Pick overall winner to split the data

### Preventing Overfitting in Regression Trees

- **Minimum Observations:** Only split if node contains sufficient observations (typically ≥20)
- **Leaf Creation:** Nodes below minimum become leaves automatically
- **Prediction:** Leaf predicts the mean of its observations
