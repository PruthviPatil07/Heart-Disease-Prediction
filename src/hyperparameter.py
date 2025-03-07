from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


# Define hyperparameter grids
param_grids = {
    "DecisionTreeClassifier(random_state=42)": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "RandomForestClassifier(random_state=42)": {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    "AdaBoostClassifier()": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1, 10]
    }
}

import joblib
model = joblib.load(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\models\model')

param_grids = param_grids[str(model)]

import pandas as pd

X_train = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\x_train.csv')
Y_train = pd.read_csv(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\data\preprocess\y_train.csv')

# Perform Grid Search
grid_search = GridSearchCV(model, param_grids, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)  # Assuming you have X_train and y_train
best_model = grid_search.best_estimator_
print(f"Best parameters for: {grid_search.best_params_}")

import joblib
import os
joblib.dump(best_model,r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\models\best_model')