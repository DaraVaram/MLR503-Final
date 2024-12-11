### **Time-Series Data Science Challenge**

#### **Scenario**
You are working as a data scientist for an energy company. The company collects hourly data on electricity consumption (in kilowatt-hours) from residential customers across multiple cities. They want you to analyze and model this data to address several business-critical questions. Your task involves predicting future electricity consumption and explaining the results to both technical and non-technical audiences.

---

#### **Dataset**
The dataset consists of the following features:
1. **Timestamp**: The timestamp for the reading (YYYY-MM-DD HH:MM format).
2. **City_ID**: Identifier for the city.
3. **Temperature**: Average hourly temperature (in °C).
4. **Humidity**: Average hourly humidity (%).
5. **Electricity_Consumption**: Hourly electricity consumption (kWh).

The dataset is provided in CSV format and covers data for five cities over three years.

---

#### **Tasks**
### **Part 1: Data Preprocessing**
1. Handle missing values in the dataset.
2. Check for outliers and remove or transform them if necessary.
3. Normalize or scale the features for better model performance.
4. Split the dataset into training (first two years) and testing (last year) sets based on the timestamp.

---

### **Part 2: Exploratory Data Analysis (EDA)**
1. Visualize the time-series data to identify trends and seasonality.
2. Perform feature correlation analysis to understand relationships between features.
3. Group the data by city and analyze the differences in electricity consumption patterns.

---

### **Part 3: Unsupervised Learning**
1. Use Principal Component Analysis (PCA) to reduce dimensionality and identify key patterns in the data.
2. Apply t-SNE to visualize clusters of similar consumption patterns across cities.

---

### **Part 4: Modeling**
#### **Regression Models**
1. Predict future electricity consumption using the following models:
   - Linear Regression (univariate and multivariate)
   - Decision Trees
   - Random Forests
2. Compare the performance of these models using metrics like Mean Absolute Error (MAE) and R-squared.

#### **Classification Models**
1. Classify consumption levels into three categories (Low, Medium, High) using:
   - Logistic Regression
   - Support Vector Machines (SVM)
2. Compare the performance using F1-score, precision, and recall.

---

### **Part 5: Explainable AI**
1. Use SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-Agnostic Explanations) to explain the predictions of the best-performing model.
2. Visualize feature importance for both regression and classification models.

---

### **Part 6: Presentation**
#### **To Technical Audiences**
- Provide a detailed report on your methodology, model performance, and the rationale behind feature engineering and selection.

#### **To Non-Technical Audiences**
- Summarize your findings, explaining trends, consumption patterns, and actionable insights for decision-makers.
- Use simple visuals to show how your model predicts consumption and classify usage levels.
