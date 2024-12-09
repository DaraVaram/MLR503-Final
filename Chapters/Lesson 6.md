## Lesson 6 - Regularization

### Overfitting

The main idea here is that we want to prevent overfitting. What is overfitting? If we have too many features, the hypothesis that we are learning may fit the training data "too well," to a point where we have some $J(w) \approx 0$. When this happens, we will not be able to generalize to new unseen (testing) data well, effectively "overfitting" the model to training data. 

Overfitting is synonymous with having high variance. 

On the other hand, underfitting (high bias) is when we are not actually learning a whole lot from the training data. As a result, we generalize to the point where we don't actually pick out any of the important features that would contribute to the prediction.

How would be address the issue of overfitting? There's a couple of things that we can do: 
- Reduce the number of features: We can select a few of the features that we want (based on, maybe, some sort of feature importance metric), and remove the rest (could be redundant or irrelevant features)
- Regularization: We keep all the features, but reduce the magnitude of the parameters $w_j$. This works well when we have a lot of features that all contribute to the target variable (basically in the cases where we don't have irrelevant features).

### Ridge / Linear regression w/ Regularization

When we want to regularize, we introduce another parameter in the cost function $J(w)$. This is what the formula would look like:

$$J(w) = \frac{1}{2m} \[\sum_{i = 1}^{m} (h_w(x_i) - y_i)^2 + \lambda \sum_{j=1}^n w_j^2\]$$

Where our $\lambda$ is the regularization parameter. By adjusting it, we can decide how much we want to minimize the parameters $w_j$. In this case, gradient descent and the update rule would be as follows: 

$$w_0 =   w_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_w(x_i) - y_i))$$

$$w_1 = w_1(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i = 1}^m (h_w(x^{(i)}) - y^{(i)}) x_j$$

Note that $j = 1, 2, 3, ..., n$ where n is the dimension of the input variable, $x$. 

We can actually use the closed-form normal equation in this case. The formula for the new $w$ would be as follows: 

```math
  \begin{align}
    w & =(X^{\top}X + \lambda & \begin{bmatrix}
           0 & 0 \\ 0 & I \\
         \end{bmatrix} 
  \end{align})^{-1}X^{\top}y
```

This is the whole idea behind **Ridge** regression. It builds in the regularization term, and _shrinks_ the coefficients to small, **non-zero** values.  On the other hand, we have LASSO regression. 


### LASSO Regression (Least Absolute Shrinkage and Selection Operator)

The difference between Ridge and LASSO is the way the regularization term is treated in the cost / loss function. Here is the equation: 

$$J(w) = \frac{1}{2m} \[\sum_{i = 1}^{m} (h_w(x_i) - y_i)^2 + \lambda \sum_{j=1}^n |w_j|\]$$

In this case, the partial derivative w.r.t. $w_j$ is given by the following: 

$$\frac{\partial}{\partial w_j} J(w) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x^{(i)}) - y^{(i)}) + \frac{\lambda}{2m} \text{sign}(w_j)$$

Therefore, when we want to get the actual parameters $w_j$, we would be dealing with something of the following sort: 

$$w_0 := w_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_w(x_i) - y_i))$$

$$w_1 = w_1(1 - \alpha \frac{\lambda}{2m})\text{sign}(w_j) - \alpha \frac{1}{m} \sum_{i = 1}^m (h_w(x^{(i)}) - y^{(i)}) x_j$$

They're very similar (Ridge and LASSO). But they are not the same. For $w_0$, they are the exact same. 

**Important Notes about LASSO:** In gradient descent for LASSO, the update rule for each coefficient includes a term that is proportional to the sign of the coefficient. In other words: 
- If we have a positive coefficient s.t. $w_j > 0$: The reg. term _substracts_ a constanst value to pull it towards 0.
- If we have a negative coefficient s.t. $w_j < 0$: The reg. term _adds_ a constant term to push it towards 0.
- The way it works is that we completely eliminate the parameters $w_j$. We make them get as close to 0 as possible until they actually become 0. That's how this works. This is different to Ridge, which just pulls them close to non-zero values.
- This is good for feature selection as well as regression.
- Since we have this sign business going on, there is no analytical (normal equation) way to do LASSO regression.

### Logistic Regression - Regularization

We basically have the same Ridge and LASSO reg. going on here, except now they're called L1 (LASSO) and L2 (Ridge). A good way to remember this is based on the power of the regularization term. Ridge is squared, so L2, and so on. 

#### L2 Regularization: 

$$J(w) = \[ \frac{1}{m} \sum_{}^{} y^{(i)}\log(h_w(x^{(i)})) + (1 - y^{(i)})\log(1 - h_w(x^{(i)}))          \] + \frac{\lambda}{2m} \sum_{j = 1}^{n} w_j^2$$

The partial derivative w.r.t. $w_j$ would thus be: 

$$\frac{\partial}{\partial w_j} J(w) = \frac{1}{m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)}))x^{(i)}_j + \frac{\lambda}{m} w_j$$

Thus: 

$$w_0 =   w_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_w(x_i) - y_i))$$

$$w_1 = w_1(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i = 1}^m (h_w(x^{(i)}) - y^{(i)}) x_j$$

#### L1 Regularization: 

$$J(w) = -\frac{1}{m} \sum_{i=1}^{m} y_i \log(h_w(x_i)) + (1-y_i)\log(1-h_w(x_i)) + \frac{\lambda}{2m} \sum_{j = 1}^{n} |w_j|$$

Thus the derivative w.r.t. $w_j$ is: 

$$\frac{\partial}{\partial w_j} J(w) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x^{(i)}) - y^{(i)}) + \frac{\lambda}{2m} \text{sign}(w_j)$$

And finally, the parameters themselves are updated through: 

$$w_0 := w_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_w(x_i) - y_i))$$

$$w_1 = w_1(1 - \alpha \frac{\lambda}{2m})\text{sign}(w_j) - \alpha \frac{1}{m} \sum_{i = 1}^m (h_w(x^{(i)}) - y^{(i)}) x_j$$

Let's make our lives easier and put all of this into a table. ChatGPT cannot render this for some stupid reason. We'll revisit it later.
