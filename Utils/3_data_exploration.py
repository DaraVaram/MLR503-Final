import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from scipy.stats import pearsonr, spearmanr, chi2_contingency, pointbiserialr
from sklearn.impute import SimpleImputer

##### Data exploration #####

def print_missing(df):
    '''Prints a DataFrame with frequency and percentage of missing values for each feature'''
    pd.set_option('display.max_colwidth', None)
    pd.DataFrame({'Frequency': df.isnull().sum(), 'Percentage': df.isnull().sum() / len(df) * 100})

def print_unique(df, cols):
    '''Prints unique entries per feature, best for categorical features'''
    for col in cols:
        print(f'{col:15}: {df[col].unique()}')

def drop_columns(df, cols):
    '''Expected a list cols = [col1, col2, ...]'''
    return df.drop(cols, axis=1)

def concat_df(dfs):
    '''Expected a list dfs = [df1, df2, ...]'''
    return pd.concat(dfs, axis=1)

##### Visualizing attributes #####

def plot_numerical_attribute(df, col):
    '''Plots histogram of one numerical attrbute'''
    print(f"Median of {col}: {df[col].median()}")
    print(f"Mode of {col}: {df[col].mode()[0]}")
    print(f"Max of {col}: {df[col].max()}")
    print(f"Min of {col}: {df[col].min()}")
    print(f"Range of {col}: {df[col].max() - df[col].min()}\n")

    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.show()

def plot_categorical_attribute(df, col):
    '''Plots bar chart of one categorical attribute'''
    print(df[col].value_counts())

    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Frequency of {col}")
    plt.show()

##### Imputing missing values #####

def impute(df, col, strat):
    '''strat must be in ['median', 'mean', 'mode']'''
    imputer = SimpleImputer(strategy='most_frequent' if strat == 'mode' else strat)
    return imputer.fit_transform(df[[col]])

##### Calculate correlations #####

def print_numerical_corr(df, attribute_A, attribute_B, test):
    '''Prints numerical correlation test results'''
    if test == 'pearson':
        corr, p_value = pearsonr(df[attribute_A], df[attribute_B])
    elif test == 'spearman':
        corr, p_value = spearmanr(df[attribute_A], df[attribute_B])
    elif test == 'point biserial':
        corr, p_value = pointbiserialr(df[attribute_A], df[attribute_B])

    print(f"{test} correlation between {attribute_A} and {attribute_B}: {corr}, p-value: {p_value}")

def print_categorical_corr(df, attribute_A, attribute_B):
    '''Prints categorical correlation test results'''
    contingency_table = pd.crosstab(df[attribute_A], df[attribute_B])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-squared Test Statistic: {chi2}")
    print(f"p-value: {p_value}")
    print(f"Degrees of Freedom: {dof}")

##### Handle skewness #####

def apply_log(df, col):
    ''' Apply log(1 + x) transformation to handle skewness in the data'''
    return np.log1p(df[col])

##### Scale features #####

def standard_scaler(X_train, X_test):
    '''Assuming X_train and X_test is a DataFrame'''
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Optional
    return scaler, X_train_scaled, X_test_scaled

def min_max_scaler(X_train, X_test):
    '''Assuming X_train and X_test is a DataFrame'''
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Optional
    return scaler, X_train_scaled, X_test_scaled

def one_hot_encoder(df, cols):
    '''One-hot encoding nominal categorical variables'''
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cols = encoder.fit_transform(df[cols])
    return encoder, pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(cols))

def ordinal_encoder(df, cols):
    '''Ordinal encoding ordinal categorical variables'''
    encoder = OrdinalEncoder()
    encoded_cols = encoder.fit_transform(df[cols])
    return encoder, pd.DataFrame(encoded_cols, columns=cols)
