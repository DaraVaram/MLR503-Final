## Lesson 9 - ANOMALY DETECTION

This is relatively straight-forward from the get-go. You have a collection of points. You get two new points. One of them falls within the same region as the others, the other falls outside of that area. What is an anomaly and what isn't? Exactly. Done. 

We have a dataset $\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$ containing $m$ training examples. We want to see if a model $p(x)$ is able to detect anomalies. This is usually given as a threshold: 

$$ p(x_{\text{test}}) \geq \epsilon \longrightarrow \text{ OK}$$
$$ p(x_{\text{test}}) < \epsilon \longrightarrow \text{ Flag anomaly}$$

There's a bunch of examples of applications for anomaly detection. It's kind of useless to go through it here. 

### Gaussian distribution

$$x \sim \mathcal{N}(\mu, \sigma^2)$$

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi} \sigma} \exp{(- \frac{(x - \mu)^2}{2 \sigma^2})}$$

Obviously changing the mean will make it centered at somewhere else. Changing $\sigma$ can effect the spread of the data itself. Let's say we take the same training set as before : $\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$. Each of the features of $x$, denoted as $x_1, x_2, ... , x_n$ are distributed according to a certain probability distribution, say $x_1 \sim \mathcal{N}(\mu_1, \sigma^2_1), x_2 \sim \mathcal{N}(\mu_2, \sigma^2_2), ..., x_n \sim \mathcal{N}(\mu_n, \sigma^2_n)$. Then we can take $p$ as: 

$$p(x) = \prod_{j = 1}^{n} p(x_j; \mu_j, \sigma_j^2)$$

Let's put this into an algorithm: 
1. Choose the features $x_i$ that may be indicative of anomalous examples
2. Fit the parameters $\mu_1, \sigma_1^2, \mu_2, \sigma_2^2, ..., \mu_n, \sigma_n^2$ such that:
   $$\mu_j = \frac{1}{m} \sum_{i = 1}^{m} x_j^{(i)}, \sigma^2_j = \frac{1}{m} \sum_{i = 1}^{m} (x^{(i)}_j - \mu_j)^2$$ 
3. Then, given a new example $x$, compute $p(x)$:

$$p(x) = \prod_{j = 1}^{n} p(x_j; \mu_j, \sigma_j^2) = \prod_{j = 1}^{n} \frac{1}{\sqrt{2 \pi} \sigma_j} \exp{(- \frac{(x - \mu_j)^2}{2 \sigma_j^2})}$$

### Multivariate Gaussian distribution

In this case, we fit the model $p(x)$ by setting: 

$$ \mu = \frac{1}{m} \sum_{i =1}^{m} x^{(i)}$$
$$ \Sigma = \frac{1}{m}  \sum_{i = 1}^{m} (x^{(i)} - \mu)(x^{(i)} - \mu)^{\top}$$

Then, given a new example $x$, we would compute: 

$$\frac{1}{2\pi^{(n/2)} \Sigma^{1/2}} \exp{(- \frac{1}{2} (x - \mu)^{\top} \Sigma^{-1} (x - \mu))}$$

What is the difference between multi-variate Gaussian and the standard Gaussian?

#### Standard: 
- We basically are manually creating features to try and see if there are any unusualy combinations
- Computationall cheaper
- Works well even if $m$ is small (training size is small)

#### Multivariate: 
- Automatically captures the correlations between features (no need to manually create features)
- Computationally more expensive
- Need to have $m > n$ or else $\Sigma$ is non-invertible. In other words, the number of training examples has to exceed the number of features.

