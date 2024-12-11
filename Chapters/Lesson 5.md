## Lesson 5 - LOGISTIC REGRESSION

When do we use logistic regression? It's similar in terms of how it works to linear regression. However, we use this for classification problems. Pretty self-explanatory. 

We could use some sort of thresholding, where we say that if $h_w(x)$ is greater than an amount, it assigns the target variable $y = 1$, and if it's less than that amount, we assign $y = 0$. We want to have some $h_w(x)$ such that $0 \leq h_w(x) \leq 1$. This would allow us to essentially look at it in terms of a probability. We can use the sigmoid function here: 

$$h_w(x) = g(w^{\top}x), \text{ where } g(z) = \frac{1}{1 + e^{-z}}$$

$$h_w(x) = \frac{1}{1 + e^{-w^{\top}x}}$$

Using words, this will give us the probability of $y = 1$ given $x$, using the parameters $w$. The decision boundary is basically the line that is drawn that separates the classes. For more complex data, we may not be able to "separate" the data with a line. We would need to use some sort of higher order parameterization. 

### Logistic regression cost / loss function

Recall the linear regression cost function: $J(w) = \frac{1}{2m} \sum_{i = 1}^{m} (h_w(x_i) - y_i)^2$. Therefore, $\text{cost}(h_w(x), y) = \frac{1}{2} (h_w(x) - y)^2$. We know that $h_w(x) = \frac{1}{1 + e^{-w^{\top}x}}$. Note that this makes the cost function **convex**, meaning that there is only one global minimum as opposed to multiple local minima. 

- Linear regression: Non-convex
- Logistic regression: Convex

To put everything in one equation: 

$$J(w) = -\frac{1}{m} \sum_{i=1}^{m} y_i \log(h_w(x_i)) + (1-y_i)\log(1-h_w(x_i))$$

We want to minimize over $w$ again. To do that, we will need to get the partial derivatives w.r.t. $w$ first, which is a whole process because of $h_w(x) = \frac{1}{1 + e^{-w^{\top}x}}$. To summarize it below: 

$$\frac{d g(z)}{dz} = -(1 + e^{-z})^{-2}(e^{-z}) (-1) = \frac{e^{-z}}{(1+e^{-z})^2}$$

By re-arranging, we have that $e^{-z} = \frac{1-g(z)}{g(z)}$. Therefore:

$$\frac{d g(z)}{dz} = g(z)(1-g(z))$$

As a result: 

$$\frac{\partial h_w(x)}{\partial w_j} = h_w(x) (1- h_w(x)) x_j$$

We keep going (some other stuff happens along the way), but we eventually end up with: 

$$w_j := w_j - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_w(x^{(i)}) - y^{(i)})x_j^{(i)}$$

This is the update rule for logistic regression, given the use of the new cost function. The algorithm is literally identical to the linear regression cost function. 

### Extending beyond binary classification

We can use a one-vs-all approach. This is self-explanatory. We pick the highest probability out of all the runs. 

### Evaluation metrics to determine how good the model is

This is where we need to know about the confusion matrix: 

|              | Predicted Negative | Predicted Positive |
| ------------ | ------------------ | ------------------ |
| **Actual Negative** | TN | FP |
| **Actual Positive** | FN | TP |

- Accuracy: $\frac{\text{TN} + \text{TP}} {\text{All values}}$
- Precision: $\frac{\text{TP}}{TP + FP}$. Of all instances predicted positive, how many were actually positive?
- Recall: $\frac{\text{TP}}{TP + FN}$. Of all actual positive instances, how many were correctly predicted as positive?
- F1-Score: $2 \frac{\text{Precision Recall}}{\text{Precision} + \text{Recall}}$

### Interpretation of Coefficients
Logistic regression predicts the **log-odds** of the target variable $y$ being 1. We will use an example here: 

$$w^{\top}x = -1 + 0.5x_1 - 0.8x_2$$

When both $x_1$ and $x_2$ are 0, then the **log-odds** of $y = 1$ is -1, $\longrightarrow$ the odds are $e^{-1} = 0.37$ to 1. Therefore: 

$$P(y = 1) = \frac{0.37}{1 + 0.37} = 0.27$$

- For a one-unit increase in the value of $x_1$, the **log-odds** of $y = 1$ increase by 0.5. In other words, the odds of $y = 1$ increase by $e^{0.5} = 1.65$ which is equivalent to around 65%.
- For a one-unit increase in the value of $x_2$, the **log-odds** of $y = 1$ decrease by 0.8. In other words, the odds of $y = 1$ decrease by $e^{-0.8} = 0.45$, or around a 55% decrease.

That's it. That's the entire idea of logistic regression. At least in theory.
