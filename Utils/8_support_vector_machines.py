import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC

##### Support Vector Machines #####

def fit_svm(X_train, y_train, C=1e10, kernel='linear'):
    '''
    Higher C value -> Hard SVM (C = 1e10)
    Lower C value -> Soft SVM (C = 1)
    kernel in ['linear', 'rbf']
    if RBF and sigma specified, then gamma=1 / (2 * (sigma**2)) as SVC parameter
    '''
    model = SVC(C=C, kernel=kernel)
    model.fit(X_train, y_train)
    return model

##### Visuals #####

def plot_decision_boundary_linear_svm(model, X, y):
    '''Plots decision boundary of SVM model, assuming two features in X, assuming LINEAR kernel '''
    
    feature_1, feature_2 = X.columns[0], X.columns[1]
    feature_1_min, feature_1_max = X[feature_1].min() - 0.1, X[feature_1].max() + 0.1

    coef = model.coef_
    intercept = model.intercept_   

    x_values = np.linspace(feature_1_min, feature_1_max, 100)
    decision_boundary = -(coef[0][0] * x_values + intercept) / coef[0][1]

    margin_distance = 1 / np.sqrt((coef[0][0] ** 2) + (coef[0][1] ** 2))
    upper_margin, lower_margin = decision_boundary + margin_distance, decision_boundary - margin_distance

    support_vectors = model.support_vectors_
    # To get number of support vectors: model.support_vectors_.shape[0]

    plt.scatter(X[feature_1], X[feature_2], c=y, s=20, edgecolor='k', cmap='rainbow') # Data points
    plt.plot(x_values, decision_boundary, color='black', linestyle='--', label='Decision Boundary') # Decision boundary
    plt.plot(x_values, upper_margin, color='grey', linestyle=':', label='Upper SVM Margin') # Upper margin
    plt.plot(x_values, lower_margin, color='grey', linestyle=':', label='Lower SVM Margin') # Lower margin
    plt.scatter(support_vectors[:,0], support_vectors[:,1], s=200, marker='*', facecolors='orange', label='Support vectors') # Support vectors

    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title('SVM Decision Boundary')

    plt.legend()
    plt.show()

def plot_decision_boundary_rbf_svm(model, X, y):
    '''Plots decision boundary of SVM model, assuming two features in X, assuming RBF kernel '''

    feature_1, feature_2 = X.columns[0], X.columns[1]
    x_min, x_max = X[feature_1].min() - 0.1, X[feature_1].max() + 0.1
    y_min, y_max = X[feature_2].min() - 0.1, X[feature_2].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    grid = np.c_[xx.ravel(), yy.ravel()]
    decision_boundary = model.decision_function(grid).reshape(xx.shape)

    plt.scatter(X[feature_1], X[feature_2], c=y, s=20, edgecolor='k', cmap='rainbow', label='Data')
    plt.contour(xx, yy, decision_boundary, levels=[0], colors='black', linewidths=1.5, linestyles='-')
    plt.contour(xx, yy, decision_boundary, levels=[-1, 1], colors='black', linewidths=1, linestyles='--')
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title("Decision Boundary with RBF Kernel SVM ")
    plt.legend()
    plt.show()