## Lesson 7 - DECISION TREES AND ENSEMBLES. 

This is a very visual chapter. The only things included here will be the notes on the theory itself, which hopefully shouldn't take up too much space. 

### Decisions in a Decision Tree (DT)
- The decision we want to make regarding what node(s) to split at is based on the purity. We either want to maximize purity, or minimize impurity.
- When do we stop splitting? This is based on a few things. We either stop when a node 100% belongs to one class, exceeds a certain depth, or the number of examples / purity is below a certain threshold.

When do we use and not use DTs?
- It works well on tabular or structured data. Not for images and stuff like that. That's not structured.
- It's very fast.
- Bigger DTs are harder to interpret, whilst smaller ones generally make more sense (human interpretability)

### How do DTs work?
We need to revisit this based on my understanding of how decision trees work. 

$$H(p_1) = \sum_{i = 1}^{n} p_i \log_2(p_i)$$

Since we are dealing with a binary case, then: 

$$H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$

Where the $p$ values are just the fraction of examples that belong to one class. 

Then we do something related to the information gain, which is just: 

$$ H(p_1^{\text{root}}) - (w^{\text{left}} H(p_1^{\text{left}}) + w^{\text{right}} H(p_1^{\text{right}})   $$

The $w$ here is the total weight, I think? Like basically how many there are in total on that side (left or right). This is so much nicer and easier on paper. We will get to that eventually. Here is a step-by-step guide on how to do this stuff anyway: 

1. Start with all examples at the root note
2. Calculate information gain for all possible features, and pick the one with the highest information gain
3. Split data based on selected feature, and create left and rigth branches
4. Keep repeating this process of splitting until stopping criteria (mentioned earlier) is met.

### What if we are dealing with continuous variables?

I think this will come on the final because it didn't in the midterm. Just seems likely. Here, instead of taking $(w^{\text{left}} H(p_1^{\text{left}}) + w^{\text{right}} H(p_1^{\text{right}})$, we take the variance instead. Therefore, it would be something like $(w^{\text{left}} \text{var(left examples)} + w^{\text{right}} \text{var(right examples)})$

### Some observations about DTs:
- They are highly sensitive to small channges in the data. This is not ideal.
- Single decision trees are not very robust (hard to generalize)
- As a result, if you train multiple decision trees, you're more likely to get better predictions. This is where we get to the idea of ensemble trees. This basically just means take a bunch of decision trees (with slight modifications in the training examples, whether that be taking different subsets, etc...) and combine the output predictions with each other.

--> Sampling with replacement: When you sample something, you replace it and put it back in the space you're sampling from. 
--> Sampling without replacement: Do not put it back in the sampling space  

#### Bagging Decision Trees: 

For some $b = 1, ..., B$, we use sampling **with replacement** to create a new training set of size $m$. We train subsequent decision trees on the new dataset(s) which are actually subsets of the original dataset. 

The typical choice of $B$ here is between 64 and 128. The way the predictions work is based on a voting thing. The more trees vote for a certain prediction, the more likely it is to be correct. It is important to note that after a certain number of trees, we get diminishing returns. The splits are commmonly the same. 

#### Random Forest: 

At each node, when choosing a feature for the split, if $n$ features are available, we pick a random subset $k$ from the $n$ available features and train the tree only on those $k$ subsets. Typically, if we have a large value of $n$ where $n$ is the number of features, $k = \sqrt{n}$. 


#### Boosted Decision Trees: 

For some $b = 1, ..., B$, we create subsets of training size $m$ from the original data (sampling **with replacement**), but then instead of mkaing the chance of examples being chosen the same ($\frac{1}{m}$), we make it so that it's more likely that the examples that are misclassified are picked (from previously trained trees). 

##### eXtreme Gradient Boosting (XGB): 

Subsequent trees are trained to predict the residuals from the previous trees. 
