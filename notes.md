# Notes

Chloé-Agathe Azencott, Centre for Computational Biology

## 1. 23 Mars 2020

### 1.1 Introduction

- risk minimization
    - ingredients: data, hypothesis class (shape of the decision function `f`), loss function
    - recipe: find among all functions of the hypothesis class, one that minimizes the loss on the training data
- (un)supervised learning
    - data matrix
    - supervision
        - binary classification
        - multi-class classification
        - regression
- large scale
    - does not fit in RAM
    - data streams
    - considerations:
        - performance increases with nb of samples
        - likely to overfit
- ML problems
    - unsupervised learning (new way to see data)
        - dimensionality reduction (find a lower-dimensional representation)
            - (PCA)[https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad]
        - clustering
            - K-means
        - density estimation
        - feature learning
    - supervised learning (make predictions)
        - build a predictor `f(x)~y`
        - classification (discrete predictions)
            - logistic regression
            - SVM
        - regression (continuous predictions)
            - linear regression: [ordinary least squares](https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/)
            - [ridge regression](https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db)
            - [bias and variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
    - semi-supervised learning
    - reinforcement learning

### 1.2. Dimensionality reduction: PCA

- Principal Compenents Analysis
- The k-th principal component:
    – is orthogonal to all previous components;
    – captures the largest amount of variance;
    – solution: w is the k-th eigenvector of X^TX

### 1.3. Clustering: K-means

- find a cluster assignement that minimizes the intra-cluster variance (sum on each centroid of the distances of each point to its centroid)
    - [Voronoi tesselation](https://fr.wikipedia.org/wiki/Diagramme_de_Vorono%C3%AF)
    - NP-hard, iterative (fix centroids, update assignments, update centroids)

### 1.4. Ridge regression

- Least Squares (OLS) fit: only exists when X^TX invertible
- ridge regularization, solution unique and always exists
    - parameter lambda tells bias and variance
        - `lambda ~ 0`: OLS, low bias, high variance
        - `lambda ~ +inf`: `weights ~ 0` high bias, low variance

![Under-over-fitting](./pics/under-over-fitting.png)

- hyperparameter setting: `lambda`
    - cross-validation, choose `lambda` with best cross-validation score
- l2-regularization learning
    - generalization of ridge regression to any loss

### 1.5. Gradient descent

- if the loss is convex, then the problem is **strictly convex** and has a **unique global solution** which can be found numerically

### 1.6. Classification (to do)

### 1.7. Kernel methods

- non-linear mapping to a feature space
- efficient to compute

![Complexity](./pics/complexity.png)
