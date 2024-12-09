## Lesson 4 - SUPERVISED LEARNING – LINEAR REGRESSION

We don't need to go into details about what is regression or supervised learning. This is self-explanatory. However, here is some notation that will be useful: 
- $m$: number of training examples
- $x$: input variables / features / attributes
- $n$: dimension of an input variable $x$.
- $y$: output variable (target)
- Each pair of $(x^{(i)}, y^{(i)})$ is the $i^{\text{th}}$ training example

The way supervised learning works is we pass in some $x$ and based on some hypothesis $h$, we can an estimated output $y$. The role of $h$ as a function is to map the input variables $x$ to the output variable $y$.

In the case of univariate linear regression, our hypothesis $h$ would look something like: 

$$h_w (x) = w_0 + w_1 x$$

Where $w_0$ and $w_1$ are the parameters we can change. The idea here is to get the best pair of $w$s to estimate $y$ with the least amount of error. Speaking of error, let's looking at the cost function we will be using in this case, which is just the squared error function: 

$$J(w_0, w_1) = \frac{1}{2m} \sum_{i = 1}^{m} (h_w(x_i) - y_i)^2$$

Our goal is to minimize the loss / cost function. Mathematically: 

$$\min_{w_0, w_1} J(w_0, w_1) = \min_{w_0, w_1} \frac{1}{2m} \sum_{i = 1}^{m} (h_w(x_i) - y_i)^2$$

Pick the best pair of $w_0$ and $w_1$ such that the loss is at a minimum. We start with an initial pair, and then keep changing it with the hopes of arriving at a minimum. This is where we get into the business of partial derivatives and whatnot, or in other words, gradient descent: 

$$w_j := w_j - \alpha \frac{\partial}{\partial w_j} J(w_0, w_1)$$

The term $\frac{\partial}{\partial w_j} J(w_0, w_1)$ can be expanded as follows. We know that $J(w_0, w_1) = \frac{1}{2m} \sum_{i = 1}^{m} (h_w(x_i) - y_i)^2$. Then, the partial derivative with respect to $w_0$ can be given by: 

$$\frac{\partial}{\partial w_0} J(w_0, w_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x_i) - y_i)$$

$$\frac{\partial}{\partial w_1} J(w_0, w_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x_i) - y_i)(x_i)$$

So when we get into gradient descent, these are the terms that we will be using in order to update the values of $w_0$ and $w_1$. 

### What if we extend this case to more variables $n \geq 1$?

Let us further cement the notation: 
- x^{(i)}: the $i^{text{th}} training example's input features
- x^{(i)}_j: value of the feature $j$ in the $i^{text{th}} training example

Previously, we would say that we have some $h_w (x) = w_0 + w_1 x$, for one input feature. Now, we can extend this to the following: 

$$h_w (x) = w_0 + w_1 x + w_2 x_2 + ... + w_n x_n$$

The same cost function applies: $J(w_0, w_1, ..., w_n) = \frac{1}{2m} \sum_{i = 1}^{m} (h_w(x_i) - y_i)^2$, but now obviously we have more $w$s. For gradient descent, we are once again dealing with the case of $\frac{\partial}{\partial w_j} J(w_0, w_1, ..., w_n)$. Now, we will have the following: 

$$\frac{\partial}{\partial w_0} J(w_0, w_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x^{(i)}) - y_i)$$

$$\frac{\partial}{\partial w_1} J(w_0, w_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x^{(i)}) - y_i)(x^{(i)}_1)$$

$$\frac{\partial}{\partial w_1} J(w_0, w_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x^{(i)}) - y_i)(x^{(i)}_2)$$

$$\vdots$$

$$\frac{\partial}{\partial w_1} J(w_0, w_1) = \frac{1}{m} \sum_{i = 1}^{m} (h_w(x^{(i)}) - y_i)(x^{(i)}_n)$$

It is typically a good idea to transform the data to be within a specified range. We don't have to have ranges that are too big. Recall the gradient descent update rule: $w_j := w_j - \alpha \frac{\partial}{\partial w_j} J(w_0, w_1)$. How do we choose the $\alpha$ such that we get to the minimum? 
- We want the cost function $J$ to be decreasing at each iteration of training. If we see that $J$ is actually increasing, that may be a sign that $\alpha$ is too large.
- It could also be the case that $\alpha$ is too small, meaning that across the training iterations, it would be too difficult to reach a minimum.
- Basically, just find a sweet-spot that is not too little or not too large.

### Vector Representation
To make things look nicer and all, we can represent things as vectors. 

```math
  \begin{align}
    w & = \begin{bmatrix}
           w_{1} \\
           w_{2} \\
           \vdots \\
           w_{n}
         \end{bmatrix} \in \mathbb{R}^{n+1}
  \end{align}
```

```math
  \begin{align}
    x & = \begin{bmatrix}
           x_0 \\
           x_{1} \\
           x_{2} \\
           \vdots \\
           x_{n}
         \end{bmatrix} \in \mathbb{R}^{n+1}
  \end{align}, x_0 = 1
```

Using this notation, we can re-write $h_w (x) = w_0 + w_1 x + w_2 x_2 + ... + w_n x_n$ as $h_w (x) = w^{\top} x$. We can then also re-write the cost function: 

$$J(w) = \frac{1}{2m} \sum_{i = 1}^{m} (w^{\top}x_i - y_i)^2$$
