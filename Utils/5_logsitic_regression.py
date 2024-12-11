import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

##### Logisitic regression
def fit_logisitc(X_train, y_train):
    model = LogisticRegression(penalty=None, max_iter=10000)
    model.fit(X_train, y_train)
    return model

def print_hyp(model):
    '''Applicable to any model with n coefficients'''
    w = [model.intercept_[0]]
    coeff_str = ''
    for w_i in model.coef_[0]:
        w.append(w_i)
        coeff_str += f' + {w_i} * x'

    print(f"Intercept: {w[0]}")
    print(f"Coefficients: {w_i}")
    print(f"h_w(x) = {w[0]}{coeff_str}")

def get_probs(model, X):
    '''Returns model probabilities for each class'''
    probabilities = model.predict_proba(X)
    prob_class_0 = probabilities[:, 0]
    prob_class_1 = probabilities[:, 1]
    return prob_class_0, prob_class_1

def get_log_loss(model, X, y):
    '''Returns log_loss of training or testing data'''
    return log_loss(y, model.predict_proba(X))

##### Visuals #####
def plot_scatter_classes(X, y, title):
    '''Plots scatter plot of data for n classes, assuming 2 features for X'''
    plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y, cmap='rainbow', edgecolor='k')
    plt.title(title)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.show()

def plot_decision_boundary(model, X, y):
    '''Plots decision boundary of logistic regression, assuming two features in X'''
    feature_1, feature_2 = X.columns[0], X.columns[1]
    x_min, x_max = X[feature_1].min() - 0.1, X[feature_1].max() + 0.1
    y_min, y_max = X[feature_2].min() - 0.1, X[feature_2].max() + 0.1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=[feature_1, feature_2])
    
    Z = model.predict(grid_df)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
    plt.contour(xx, yy, Z, colors='k', linewidths=2)
    plt.scatter(X[feature_1], X[feature_2], c=y, s=20, edgecolor='k', cmap='rainbow')
    plt.title("Decision Boundary")
    plt.show()

##### Engineering polynomial features #####
def plot_decision_boundary_poly(model, X, y, features_generator):
    '''Plot decision boundary for a logisitc regression model with polynomial features'''
    feature_1, feature_2 = X.columns[0], X.columns[1]

    x_min, x_max = X[feature_1].min() - 0.1, X[feature_1].max() + 0.1
    y_min, y_max = X[feature_2].min() - 0.1, X[feature_2].max() + 0.1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=[feature_1, feature_2])
    
    grid_poly = features_generator.transform(grid_df)
    
    Z = model.predict(grid_poly)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
    plt.contour(xx, yy, Z, colors='k', linewidths=2)
    plt.scatter(X[feature_1], X[feature_2], c=y, s=20, edgecolor='k', cmap='rainbow')
    plt.title("Decision Boundary with Polynomial Features")
    plt.show()

def fit_plot_poly_logistic(X_train, X_test, y_train, y_test, degree):
    '''Fit and plot a logisitc regression model with polynomial features'''
    features_generator = PolynomialFeatures(degree)
    X_poly_train = features_generator.fit_transform(X_train)
    X_poly_test = features_generator.transform(X_test)

    model = LogisticRegression(penalty=None, max_iter=10000)
    model.fit(X_poly_train, y_train)

    plot_decision_boundary_poly(model, X_train, y_train, features_generator)
    plot_decision_boundary_poly(model, X_test, y_test, features_generator)
    
    train_loss = log_loss(y_train, model.predict_proba(X_poly_train))
    test_loss = log_loss(y_test, model.predict_proba(X_poly_test))
    
    return train_loss, test_loss

def plot_compare_degrees(degrees, train_losses, test_losses):
    '''Compares losses of logistic regression models with varying degrees, given that each parameter is a list'''
    plt.plot(degrees, train_losses, label="Training Loss", marker='o')
    plt.plot(degrees, test_losses, label="Test Loss", marker='o')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Log Loss")
    plt.title("Training vs Test Loss by Polynomial Degree")
    plt.legend()
    plt.show()