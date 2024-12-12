import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import zscore

# Load Dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Basic Data Summary
def data_summary(df):
    print("Shape of the dataset:", df.shape)
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())

# Handling Missing Values
def handle_missing_values(df, strategy="mean", columns=None):
    imputer = SimpleImputer(strategy=strategy)
    if columns:
        df[columns] = imputer.fit_transform(df[columns])
    else:
        df[:] = imputer.fit_transform(df)
    return df

# Encoding Categorical Variables
def encode_categorical(df, columns, encoding_type="label"):
    if encoding_type == "label":
        encoder = LabelEncoder()
        for col in columns:
            df[col] = encoder.fit_transform(df[col])
    elif encoding_type == "onehot":
        df = pd.get_dummies(df, columns=columns)
    return df

# Rare Category Handling
def handle_rare_categories(df, columns, threshold=0.01):
    for col in columns:
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        df[col] = df[col].replace(rare_categories, 'Rare')
    return df

# Splitting Dataset
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Feature Scaling
def scale_features(X_train, X_test, scaling_type="standard"):
    if scaling_type == "standard":
        scaler = StandardScaler()
    elif scaling_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling_type. Choose 'standard' or 'minmax'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Outlier Detection and Removal
def remove_outliers(df, columns):
    for col in columns:
        z_scores = zscore(df[col])
        df = df[(np.abs(z_scores) < 3)]
    return df

# Feature Engineering
def feature_engineering(df, interaction_terms):
    for term in interaction_terms:
        col1, col2 = term
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return df

# Automated Preprocessing Pipeline
def preprocessing_pipeline(numeric_features, categorical_features, \
                          numeric_strategy="mean", scaling_type="standard"):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler() if scaling_type == "standard" else MinMaxScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

# Cross-Validation Scores
def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores

# Data Visualization
def visualize_data(df):
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Pairplot
    if df.select_dtypes(include=[np.number]).shape[1] > 1:  # Check if numeric columns exist
        sns.pairplot(df)
        plt.title("Pairplot of Numeric Features")
        plt.show()

    # Countplot for Categorical Features
    for col in df.select_dtypes(include=["object", "category"]).columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        plt.show()

    # Distribution Plots
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    # Boxplots
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

    # Scatter Plots for Numeric Features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df, x=col1, y=col2)
                plt.title(f"Scatter Plot between {col1} and {col2}")
                plt.show()

    # Target vs. Categorical Variables
    if 'target' in df.columns:
        for col in df.select_dtypes(include=["object", "category"]).columns:
            plt.figure(figsize=(8, 4))
            sns.barplot(x=col, y='target', data=df, estimator=np.mean, ci=None)
            plt.title(f"Target vs {col}")
            plt.xticks(rotation=45)
            plt.show()

# Time Series Analysis
def visualize_time_series(df, date_column, target_column):
    if date_column in df.columns and target_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.sort_values(by=date_column, inplace=True)
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_column], df[target_column])
        plt.title(f"Time Series Analysis of {target_column}")
        plt.xlabel("Date")
        plt.ylabel(target_column)
        plt.grid(True)
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Load your dataset
    df = load_data("your_dataset.csv")

    # Summarize the data
    data_summary(df)

    # Handle rare categories
    df = handle_rare_categories(df, columns=['categorical_col'], threshold=0.01)

    # Remove outliers
    df = remove_outliers(df, columns=['num_col1', 'num_col2'])

    # Feature engineering
    df = feature_engineering(df, interaction_terms=[('num_col1', 'num_col2')])

    # Visualize the data
    visualize_data(df)

    # Visualize time series
    visualize_time_series(df, date_column='date', target_column='target')

    # Handle missing values
    df = handle_missing_values(df, strategy="mean", columns=['col1', 'col2'])

    # Encode categorical variables
    df = encode_categorical(df, columns=['categorical_col'], encoding_type="onehot")

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column='target')

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, scaling_type="standard")

    # Build preprocessing pipeline
    preprocessor = preprocessing_pipeline(numeric_features=['num_col1', 'num_col2'], 
                                           categorical_features=['cat_col1', 'cat_col2'])

    # Example pipeline usage with a model
    from sklearn.ensemble import RandomForestClassifier
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    model.fit(X_train, y_train)
    evaluate_model(model, X_train, y_train)
