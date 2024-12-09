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
- $w' (u - v) = w'^{\top}(u - v) = 0
- The distance is calculated as $w^{\top}x$ where $x$ is the coordinates $x_1, x_2$ at that point where we are trying to calculate the distance. The size of the vector $w'$ is important here.

$$\text{Distance} = \frac{w^{\top}x}{||w||}$$

Why are we doing all of this?
- **Logistic Regression:** Emphasis on probabilities and maximizing the likelihood of correct classifications. The cost function is tied to how much "confidence" we place in the prediction
- **SVM:** Margins and distances _from_ the decision boundary. The focus is not on probabilities, but ensuring that the points are correctly classified with a sufficient margin.

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
- For a point $x^{(i)}$ with the label $y^{(i)}$, the classifier predicts a value $w^{\top}x^{(i)}$ such that $y^{(i)}w^{\top}x^{(i)} /geq 1$
- If a point is classified correctly and has a large enough margin (it lies beyond the decision boundary by at least one unit), then the hinge loss would be 0.
- If a point is within the margin or misclassified, the hinge loss **increases**, penalizing predictions that don't satisfy the margin.

Some other stuff happens here. We need to revisit this for theory, because it's definitely coming in the final.
