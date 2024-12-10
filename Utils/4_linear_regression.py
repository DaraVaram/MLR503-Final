import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

##### Visuals #####

def plot_scatter(x, y, x_label, y_label):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_scatter_line(X, y, X_test, y_pred, x_label, y_label):
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X_test, y_pred, color='red', label='Regression line') # Another plt.plot could be added to compare to fits in one graph
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    ax = plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

##### Ordinary least squares #####

def fit_ols(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

##### SGDRegressor #####

def fit_sgd(X_train, y_train):
    model = SGDRegressor(max_iter=10000, tol=1e-3, eta0=0.01, random_state=42)
    model.fit(X_train, y_train)
    return model

##### General functions for regression models #####

def split(X, y, split = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    return X_train, X_test, y_train, y_test

def print_hyp(model):
    w_0 = model.intercept_  
    w_1 = model.coef_  
    print(f"Intercept (w_0): {w_0}")
    print(f"Coefficient (w_1): {w_1}")
    print(f"h_w(x) = {w_0} + {w_1} * x")

def adjust_standard_coef(model, scaler):
    '''
    - Adjust coefficients based on scaled data for an interpretabel hypothesis
    - Assumes StandardScaler was used
    '''
    w_0 = model.intercept_  
    w_1 = model.coef_  
    
    X_mean = scaler.mean_ 
    X_std = scaler.scale_ 

    w_1_adjusted = w_1 / X_std
    w_0_adjusted = w_0 - (w_1 * X_mean / X_std)

    print(f"Intercept (w_0): {w_0_adjusted}")
    print(f"Coefficient (w_1): {w_1_adjusted}")
    print(f"h_w(x) = {w_0_adjusted} + {w_1_adjusted} * x")

def adjust_min_max_coef(model, scaler):
    '''
    - Adjust coefficients based on scaled data for an interpretabel hypothesis
    - Assumes MinMaxScaler was used
    '''
    w_0 = model.intercept_  
    w_1 = model.coef_  
    
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    data_range = data_max - data_min

    w_0_adjusted = w_0 - (w_1 * data_min / data_range)
    w_1_adjusted = w_1 / data_range

    print(f"Intercept (w_0): {w_0_adjusted}")
    print(f"Coefficient (w_1): {w_1_adjusted}")
    print(f"h_w(x) = {w_0_adjusted} + {w_1_adjusted} * x")

def get_regression_metrics(y, y_pred):
    '''For some reason this never works but whatever heres the code anyway'''
    print(f"RMSE = {root_mean_squared_error(y, y_pred)}")
    print(f"R^2 = {r2_score(y, y_pred)}")
