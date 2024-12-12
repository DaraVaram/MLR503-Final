import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

##### Decision Trees #####
DECISION_TREES = {
    'classification': {
        'dt': DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42),
        'random forest': RandomForestClassifier(random_state=42),
        'xgboost': XGBClassifier(eval_metric='logloss', random_state=42)
    },
    'regression': {
        'dt': DecisionTreeRegressor(max_depth=3, random_state=42),
        'random forest': RandomForestRegressor(random_state=42),
        'xgboost': XGBRegressor(random_state=42)
    }
}

def fit_dt(X_train, y_train, task = 'classification', type='dt'):
    '''type must be in ['dt', 'random forest']'''
    model = DECISION_TREES[task][type]
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(model, X_train, y_train ,param_grid):
    '''
    Example of param_grid:
    param_grid = {
        'n_estimators': [51, 101, 201],
        'max_depth': [3, 5, 10, None]
    }
    '''
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f'Best Parameters are: {grid_search.best_params_}')

    best_model = grid_search.best_estimator_
    return best_model

##### Visuals #####

def plot_dt(model, df, class_names):
    '''Visualizes a decision tree. class_names = [class1, class2, ...]'''
    plt.figure(figsize=(15, 8))
    plot_tree(model, rounded=True, precision=2, filled=True, feature_names=df.columns, class_names=class_names)
    plt.show()