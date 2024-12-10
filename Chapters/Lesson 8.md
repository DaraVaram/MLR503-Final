## Lesson 8 - SUPPORT VECTOR MACHINES (SVMs)

The first part of this is just a review of how vectors work. No real theory here. Recall the following: 
- You scale a vector with a scalar. Sign indicates direction, the value itself determines the strength of the scaling
- The magnitude of a vector is given by $\sqrt{v_1^2 + v_2^2}$, denoted $||V||$
- You can add vectors and that means you put the end of one to the tip of the other. Self-explanatory. The resulting vector starts at origin and ends where the tip of the second is. 
- Subtracting is the opposite. Connect both tips to each other with a new vector. That new vector is $a - b$.
- Dot product: $a \cdot b$ is the product of their magnitudes and the cosine of the angle between them. Again, self-explanatory. $a \dot b = ||a||||b|| \cos(\theta_{ab})$
- Follow up: If two vectors are orthogonal (i.e. the angle between them is 90 degrees), then the dot product is 0 (because $\cos(\frac{\pi}{2}) = 0$

### How do we link all this back together to the idea of making decision boundaries for classification?

Let's take the following example: 

![SVM example](https://github.com/DaraVaram/MLR503-Final/blob/main/Figures/SVMs.PNG)

Here, we can see a line such that $x_1 + x_2 = 3$. The two vectors that are going from the origin are $u = \[2, 1\]$ and $v = \[1, 2\]$. Our hypothesis is given by the equation: 

$$h_w(x) = g(w_0 + w_1 x_1 + w_2 x_2) = g(-3 + x_1 + x_2)$$

We will have some vector $w'$ such that it is orthogonal to the decision boundary itself. Observe the following: 

- If $u = \[2, 1\]$ and $v = \[1 , 2\]$, then $u - v = \[1 , -1\]$
- $w' (u - v) = w'^{\top}(u - v) = 0$
- The distance is calculated as $w^{\top}x$ where $x$ is the coordinates $x_1, x_2$ at that point where we are trying to calculate the distance. The size of the vector $w'$ is important here.

$$\text{Distance} = \frac{w^{\top}x}{||w||}$$

Why are we doing all of this?
- **Logistic Regression:** Emphasis on probabilities and maximizing the likelihood of correct classifications. The cost function is tied to how much "confidence" we place in the prediction
- **SVM:** Margins and distances _from_ the decision boundary. The focus is not on probabilities, but ensuring that the points are correctly classified with a sufficient margin.

The decision boundary for a hyperplane is defined as a weight vector $w$ and a bias $b$ such that $h_w(x) = \text{sign}(w^{\top}x + b)$. Points on either side of the boundary are classified as $y = \pm 1$, depending on whether it's above or below. For the time being, let us ignore the idea of having a bias $b$ present.

Let's quickly review what logistic regression was: 

$$h_w(x) = \frac{1}{1 + e^{-w^{\top}x}}$$

If we want to predict $y = 1$, then we want $e^{-w^{\top}x} >> 0$. Otherwise, if we want $e^{-w^{\top}x} << 0$

The SVM cost function (loss function) is given by: 

$$C \sum_{i = 1}^{m} \max{(0, 1 - y^{(i)}(w^{\top}x^{(i)}))} + \frac{1}{2}\sum_{j = 1}^{n} w^2_j$$

Where $C = \frac{1}{\lambda}$. This is the hinge loss function. Where: 
- $w^{\top}x^{(i)} \geq 1$ if $y^{(i)} = 1$
- $w^{\top}x^{(i)} \leq -1$ if $y^{(i)} = -1$

The reason why we use the hinge loss function is as follows: 
- It is repsonsible for ensuring that data points are classified correctly and lie outside or **on** the margin.
- For a point $x^{(i)}$ with the label $y^{(i)}$, the classifier predicts a value $w^{\top}x^{(i)}$ such that $y^{(i)}w^{\top}x^{(i)} \geq 1$
- If a point is classified correctly and has a large enough margin (it lies beyond the decision boundary by at least one unit), then the hinge loss would be 0.
- If a point is within the margin or misclassified, the hinge loss **increases**, penalizing predictions that don't satisfy the margin.

The actual optimization itself boils down to the following: 

$$\min_w C \sum_{i = 1}^m \max{(0, 1 - y^{(i)}(w^{\top}x^{(i)}))} + \frac{1}{2}\sum_{j = 1}^{n} w^2_j$$

### Hard vs. soft margin: 

- **Hard Margin:** Assumes that the data is perfectly separable. It enforces stricter constraints, and the optimization problem is simplified to:

$$\min_w \frac{1}{2} ||w||^2 \text{ s.t. } w^{\top}x^{(i)} \geq 1 \text{ if } y^{(i)} = 1, w^{\top}x^{(i)} \leq 1 \text{ if } y^{(i)} = -1 $$

- **Soft Margin:** Allows for some misclassification or margin violations, controlled by the parameter $C$. This is for the case of noisy data or data that is overlapping.

In general, the hypothesis would give us one of the following: 

```math
h_w(x) =
    \begin{cases}
      1, & \text{if}\ w^{\top}x^{(i)} \geq 0 \\
      -1, & \text{if}\  w^{\top}x^{(i)} < 0
    \end{cases}
```

## Non-linear Decision Boundaries: 

What if we're daeling with a case where the data is non-linearly separable? Then we run into some problems. Namely the fact that we need to use kernels now. SVMs can use kernels to project data into a higher-dimensional space where a linear boundary can be found. Mathematically: 

$$\phi(x) : \mathbb{R}^{r_1} \longrightarrow \mathbb{R}^{r_2}$$

With some $r_2 > r_1$. That is the motivation here. Noww let's say we have the following: 

$$w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2 + ... \geq 0$$

Then:

```math
h_w(x) =
    \begin{cases}
      1, & \text{if}\ w_0 + w_1 x_1 + w_2 x_2 + ... \geq 0 \\
      -1, & \text{otherwise}\ 
    \end{cases}
```

Given our $x$, we compute the new features $f$ based on proximities to landmarks $l^{(1)}, l^{(2)}, ...$. There's a lot of ways to compute this similarity business. 

$$f_i = \text{similarity}(x, l^{(i)}) = \exp{(- \frac{||x- l^{(i)}||^2}{2\sigma^2})}$$

This is an example of a Gaussian kernel, which is actually seen for $t$-SNE later. We know that if $x$ is close to the "landmark," i.e. $x \approx l^{(1)}$, then the measure $f_1$ would be close to 1. However, if $x$ is far from $l^{(1)}$, then it would be near-zero. 

#### How do we get the landmarks?

We are given a dataset $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ... , (x^{(m)}, y^{(m)})$. We initially choose our landmarks as: 

$$l^{(1)} = x^{(1)}, l^{(2)} = x^{(2)}, ... , l^{(m)} = x^{(m)}$$

This ensures that each training example is evaluated relative to every other example. The resulting kernel matrix $K$ will have dimensions $m \times m$, where $m$ is the number of training examples. This is good because we will have a rich representation of the dataset, and it guarantees that the kernel matrix is well-defined whilst capturing the relationship between **all** the points. However, it is computationally expensive.

To further formulate SVM with kernels, recall the optimization function that we were using earlier: 

$$\min_w C \sum_{i = 1}^m \max{(0, 1 - y^{(i)}(w^{\top}x^{(i)}))} + \frac{1}{2}\sum_{j = 1}^{n} w^2_j$$

For SVMs with kernels, this would become: 

$$\min_w C \sum_{i = 1}^m \max{(0, 1 - y^{(i)}(w^{\top}f^{(i)}))} + \frac{1}{2}\sum_{j = 1}^{m} w^2_j$$

Such that: 

```math
    \begin{cases}
      w^{\top}f^{(i)} \geq 1 & \text{if}\ y^{(i)} = 1\\w^{\top}f^{(i)} \leq 1 & \text{if}\ y^{(i)} = -1
    \end{cases}
```

#### Parameters used for SVMs: 

Recall that $C = \frac{1}{\lambda}. Then: 

- For large $C$ (or in other words, small $\lambda$), we would have lower bias but higher variance
- For small $C$ (or large $\lambda$, we would have low variance but higher bias.
- In terms of $\sigma^2$, if $\sigma^2$ is large, then the features $f_i$ vary more smoothly.
- if  $\sigma^2$ is small, features $f_i$ vary less smoothly.

Note that for multi-class classification, we use the same one-vs-all approach we had for logistic regression. We get some $w^1, w^2, ..., w^k$ and pick the class $i$ with the largest $(w^{(i)})^{\top}x$.

### When do we use SVM and when do we use logistic regression?
- If $n$ is large (relative to $m$): Use logistic regression or SVM with linear kernel
- If $n$ is small and $m$ is intermediate, use SVM with Gaussian kernel
- If $n$ is small and $m$ is large, create or add more features, then use logistic regression nor SVM with linear kernel.

