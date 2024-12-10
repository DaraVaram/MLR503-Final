## Lesson 11 - FEATURE SELECTION AND DIMENSIONALITY REDUCTION

### Feature Selection

There are a few ways that we can do feature selection (that we have discussed earlier): 
- Pearson's correlation coefficient (linear relationships)
- Spearman's rank correlation coefficient (linear or non-linear, but has to be monotonic)

These both allow us to see the magnitude and direction of correlation **between numerical features** between $-1$ and $1$. 

For categorical vs. categorical:

- $\chi^2$ tet for dependence or association between features (no correlation)
- To quantify the relationship itself, we use Cramer's V coefficient:

$$\text{Cramer's V: } \sqrt{\frac{\chi^2}{n(k-1)}}$$

This shows only the magnitude of the association, between 0 and 1.

For numerical vs. categorical: 

We take the correlation ratio ($\eta$)

$$\eta^2 = \frac{\text{SS}_{\text{between}}}{\text{SS} _{\text{total}}}$$

We take the square root such that $\eta = \sqrt{\eta^2}$. This gives us the magnitude of the association between 0 and 1.

#### High-dimensional data:

We often deal with data that is not very easy to visualize. This is data that has many features (high values of $n$). Things like Principle Component Analysis (PCA) become very useful here because it allows us to see what the data looks like. 

### PCA

Finds new axes (also called principle components) along which the variance of the data is maximized. These axes are orthogonal to each other and area ranked by the amount of variance they explain. The whole idea is the amount of explained variance. They are derived from the **eigenvectors** of the covariance matrix of the data. 

Here is a step-by-step approach on how to do PCA: 

1. Center the data around the origin (basically just subtract the mean from all the features), assuming that the features are scaled the same way (may need to normalize otherwise). If they are not on the same scale, subtract the mean and divide by the standard deviation (so that we can get $\sim \mathcal{N}(\mu = 1, \sigma^2 = 0)$
2. The covariance matrix is given as:

$$\Sigma = \frac{1}{m} \sum_{i = 1}^{m} (x^{(i)})(x^{(i)})^{\top}$$

This is just a matrix. 

3. Find the eigenvalues and the eigenvectors of the covariance matrix $\Sigma$
4. Sort the eigenvectors by the corresponding eigenvalues in descending order. The eigenvectors with the largest eigenvalues corresponds to the first principle component, and so on. These are defined as the direction of the new axes, and explain how much variance is explained along each principle component.
5. Transform the data into the new principle component space by projecting the original data into the eigenvectors.

That's all there is to know about PCA, really. 

### $t$-SNE

The main idea of $t$-SNE is to (kind of like PCA) map high-dimensional points to a lower dimension whilst preserving local relationships between the points. We measure the similiarity between the data points in the high-dimensional space and represent this similarity as probabilities. Then, it constructs a similiar **probability distribution** in the lower-dimensional space and minimizes the difference between the two distributions using gradient descent. As a result, the local structure of the data is captured. Mathematically: 

Given a dataset $X = \{x^{(1)}, x^{(2)} x^{(3)}, ... , x^{(m)} \}$ with pairwise similarities $p_{j | i}$, where $p_{j | i}$ represents the conditional probability of picking $x^{(j)}$ as a neighbor of $x^{(i)}$ under a Gaussian distribution centered around $x^{(i)}$, defined as: 

$$p_{j | i} = \frac{\exp{- \frac{||x^{(i)} - x^{(j)}||^2}{2\sigma_i^2}}}{\sum_{k \neq i} \exp{ -\frac{||x^{(i)} - x^{(j)}||^2}{2\sigma_i^2}}}$$

Note that $p_{i j} = \frac{p_{i | j} + p_{j | i}}{2m}$. We use something called _kernel width_ to adaptively choose the desired perplexity, which, by default, is 30. 

$$\mathcal{P} = 2^{\mathcal{H}}, \mathcal{H} = \sum_{i \neq j} p_{j | i} \log_2 (p_{j | i})$$

For the low-dimensional space, $Y = \{y^{(1)}, y^{(2)} y^{(3)}, ... , y^{(m)} \}$ with pairwise similarities $q_{j | i}$, where $q_{j | i}$ represents the conditional probability of picking $y^{(j)}$ as a neighbor of $y^{(i)}$ under a student $t$-distribution. This is defined as: 

$$q_{i j} = \frac{w_{i j}}{Z}$$

$$w_{i j} = k(||y_i - y_j||)$$

$$Z = \sum_{k \neq l} w_{k l}$$

The similarity kernel in $t$-SNE is given by: 

$$k(d) = \frac{1}{(1 + d^2)}$$

As a result of all the above, $q_{i j}$ can be given as: 

$$q_{i j} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_i - y_j||^2)^{-1}}$$

The objective function $J$ (the loss function or cost) is defined as teh **Kullback-Leibler Divergence**, or KL-divergence. This is given as: 

$$J = \sum_i \text{KL} (p_{ij} || q_{ij}) = \sum_{i \neq j} p_{ij} \log{\frac{p_{ij}}{q_{ij}}} = \sum_{i \neq j} p_{ij} \log{p_{ij}} - \sum_{i \neq j} p_{ij} \log{q_{ij}}$$

$$=  \sum_{i \neq j} p_{ij} \log{p_{ij}} -  \sum_{i \neq j} p_{ij} \log{w_{ij}} +  \sum_{i \neq j} w_{ij}$$

Performing gradient descent gives us: 

$$\frac{\partial J}{\partial y_i} = 4 \sum_{j \neq i} (p_{ij} - q_{ij})(y_i - y_j)$$

#### Intuition behind $t$-SNE:

The main idea is all about similarity. If we know how data is distributed in the higher-dimensional space, we are able to effectively understand how it would be distributed if we were to enter a lower-dimensional space. Something along those lines. 
