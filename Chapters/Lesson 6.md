## Lesson 6 - Regularization

### Overfitting

The main idea here is that we want to prevent overfitting. What is overfitting? If we have too many features, the hypothesis that we are learning may fit the training data "too well," to a point where we have some $J(w) \approx 0$. When this happens, we will not be able to generalize to new unseen (testing) data well, effectively "overfitting" the model to training data. 

Overfitting is synonymous with having high variance. 

On the other hand, underfitting (high bias) is when we are not actually learning a whole lot from the training data. As a result, we generalize to the point where we don't actually pick out any of the important features that would contribute to the prediction.

How would be address the issue of overfitting? There's a couple of things that we can do: 
- Reduce the number of features: We can select a few of the features that we want (based on, maybe, some sort of feature importance metric), and remove the rest (could be redundant or irrelevant features)
- Regularization: We keep all the features, but reduce the magnitude of the parameters $w_j$. This works well when we have a lot of features that all contribute to the target variable (basically in the cases where we don't have irrelevant features).

### Regularization

When we want to regularize, we introduce another parameter in the cost function $J(w)$. This is what the formula would look like:

$$J(w) = \frac{1}{2m} \[\sum_{i = 1}^{m} (h_w(x_i) - y_i)^2 + \lambda \sum_{j=1}^n w_j^2\]$$

Where our $\lambda$ is the regularization parameter. By adjusting it, we can decide how much we want to minimize the parameters $w_j$. In this case, gradient descent and the update rule would be as follows: 

$$w_0 =   w_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_w(x_i) - y_i))$$

$$w_1 = w_1(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i = 1}^m (h_w(x^{(i)}) - y^{(i)}) x_j$$

We can actually use the closed-form normal equation in this case. The formula for the new $w$ would be as follows: 

$$w = (X^{\top}X + \lambda \begin{pmatrix} 0  & 0 \\ 0  & I  \end{pmatrix}) X^{\top} y$$

```math
  \begin{align}
    w & =(X^{\top}X + \lambda & \begin{bmatrix}
           0 & 0 \\ 0 & I \\
         \end{bmatrix} 
  \end{align})^{-1}X^{\top}y
```

This is the whole idea behind **Ridge** regression. It builds in the regularization term, and _shrinks_ the coefficients to small, **non-zero** values. 
