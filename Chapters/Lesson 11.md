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
