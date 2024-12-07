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
