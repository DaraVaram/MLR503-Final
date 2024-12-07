## Lesson 3 - DATA, CORRELATIONS, PREPROCESSING, AND PREPARATION
The overarching idea: _If you feed garbage data into a model, you will get garbage outputs. This is the motivation for data pre-processing_

**Attribute:** Property or characteristic of an object. This is also known as a _feature_, _input variable_, etc... --> Attributes have **values**, which can be numerical or symbols. We will get into this later.


**Object:** Collection of attributes describe an object in the dataset. This is also known as a _data point_.

![Attribute Types](https://github.com/DaraVaram/MLR503-Final/blob/main/Figures/Data%20Attributes.png)

No need to go into too much detail regarding what is nominal or ordinal, etc... because they are relatively self-explanatory.

**Note:** Numerical data can be either interval or ratio. 
- Interval: Differences between two values is meaningful. The difference in temperature between 20° and 30° is the same as the difference in temperature between 40° and 50°. However, there is **no true zero value**. A temperature of 0° does not mean that temperature does not exist. You also cannot say that 20° is twice as hot as 10° (no meaningful ratios).
- Differences between two values is meaningful and there is a **true zero value**. The main difference here is the fact that ratios are meaningful. Someone that weighs 40kg is twice as heavy as someone that weighs 20kg.

### How do we clean data?
In datasets, we can have things like: 
- Duplicate data: The same data can be reflected across different dimensions of the dataset
- Missing data: No data is avaiable for a specific variable (or data point in general)
- Outliers: Significant difference between the data observed and the "usual" values for a given attribute
- Wrong data: Straight up wrong.

#### How do we deal with outliers?
The first thing we have to do is identify them. We can use visualizations for this if it's possible (based on dimensionality). 

Let's say we want to use a boxplot. Then, we can calculate the IQR and get a min and max value (min: Q1 - 1.5 IQR and max: Q3 + 1.5 IQR). Whatever falls outside of that can be an outlier.

This is the case for numerical attributes. If we are dealing with categorical attributes, the logical thing to do is to look at the frequency and identify items that are less frequent. Those can be the outliers depending on the context. 

**Reminder:**
- Mean: Average of the values
- Median: The "middle" value when organized
- Mode: The most frequent (repeated) value
- Right-skewed data: The concentration of the data is on the **left**. Therefore, the mean > median. In other words, the average value's frequency is higher than the middle value's frequency.
- Left-skewed data: The concentration of the data is on the **right**. Therefore, median > mean. The average value's frequency is lower than the middle value's frequency.
- Normal distribution: Mean = median = mode.


#### Imputing Data
For numerical attributes: It depends on whether it is discrete or continuous. This is relatively self-explanatory. If we have a continuous variable, then we can impute data based on the mean or the median. If we have discrete values, then wwe can use the mode instead of the mean. 

For categorical attributes: If it is nominal, then either take the mode, or assign a new category called "Missing" or "Unkown." If it is ordinal (there is a hierarchy), then either use the median or the mode. In any case, since they are not numerical, you cannot take the mean. 

There is a few things that you need to keep in mind when deciding to impute data. The important things are the fact that it really depends on how the data is distributed. If the data is not normal (skewed), then it wouldn't make sense to impute data with the mean, because that would result in large spikes in the frequency. On the other hand, bias also has to be taken into account.

### Correlations
I am pretty sure he mentioned that for the midterm, this wouldn't be included. However, we will cover it anyway just in case. This depends on what we are trying to test against each other. 

- **Numerical vs. numerical**

**Pearson Correlationn Coefficient:** This measures the _linear_ relationship between two continuous variables. We use this method when both variables are normally distributed. Otherwise, we shouldn't use this.

$$\rho (X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\frac{1}{N} (x_i - \bar{x})(y_i - \bar{y})}{\sigma_X \sigma_Y}, -1 < \rho < 1$$

**Spearman's Rank Correlation:** Measures the _strength_ and _direction_ of the monotonic relationship between two continuous or ordinal variables. This is used when the data is non-linear, or if it is not normally distributed. 

$$\rho = 1 - \frac{6 \sum_{i} d_i^2}{n(n^2 - 1)}, -1 < \rho < 1$$

Note that $d$ here represents the differences in ranks for $X$ and $Y$. 

- **Numerical vs. categorical**

**Point-Biserial Correlation:** Special case of Pearson's correlation wwhen one variable is continuous and the other is binary (categorical).

$$r_{pb} = \frac{M_1 - M_0}{\sigma} \sqrt{p_1 p_2}$$

Where $M$ is the mean of the continuous variable for each of the two groups (binary), $\sigma$ is the standard deviation for the entire continuous data, and $p$ and $q$ are the proportions of the binary variable. I'm not sure what proportion refers to here. It could mean literally anything. 

**ANOVA:** Tests the difference in means across multiple categories. If we have a categorical attribute with more than two levels (beyond binary), then we use ANOVA. 

$$F = \frac{\text{variance between groups}}{\text{variance within groups}}$$

We compute the $F$-statistic using the variances of the difference groups. Then, we compare the $F$-statistic to a critical value from the $F$-distribution table to determine its significance. 

- **Categorical vs. categorical**

**Chi-squared Test of Independence:** Tests whether two categorical attributes are independent. 

$$\chi^2 = \sum \frac{(O-E)^2}{E}$$

We compute $\chi^2$ by summing the squared differences between the observed and expected frequencies, divided by the expected frequencies. We compare $\chi^2$ to the $\chi^2$ distribution to table to determine its significance.

### Transforming Categorical features to numerical: 

If we are dealing with ordinal data, then we can use label-encoding: 

| Category | Encoding |
| ---- | ---- |
| First class | 0 |
| Second class | 1 |
| Third class | 2 |



If we are dealing with nominal data (where the order does not matter), then we can use one-hot encoding: 


| Category | $x_1$ | $x_2$ | $x_3$ |
| -------- | ----- | ----- | ----- |
| Man      |   1   |   0   |   0   |
| Woman    |   0   |   1   |   0   |
| Child    |   0   |   0   |   1   |



This is all relatively self-explanatory. I'm just including it here so that we have a reference.

### Transforming numerical data: 
- One popular thing to do is transform data based on a min-max scaler, that takes it and encodes it between $[0, 1]$

$$X' = \frac{X - \min{X}}{\max{X} - \min{X}}$$

- Another type of transformation you can do is make it have a mean of 0 and a standard deviation of 1.

$$Z = \frac{X - \mu}{\sigma}$$
