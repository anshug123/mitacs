import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import sys

# Get the torsion number from the command line arguments
torsion_number = int(sys.argv[1])

# Load the training and testing data from CSV files
x_train = pd.read_csv("x_train_pca.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("x_test_pca.csv")
y_test = pd.read_csv("y_test.csv")

# Convert data to numpy arrays
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

# Define the parameter grid for the GridSearchCV
param = {
    'max_depth': [1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50], # Possible values for the maximum depth of the tree
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], # Possible criteria for the tree
    'splitter': ['best', 'random'], # Possible strategies for choosing the split at each node
    'max_features': ['auto', 'sqrt', 'log2'] # Possible features to consider for the best split
}

# Initialize the DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# Initialize GridSearchCV with the regressor, parameter grid, and scoring method
grid = GridSearchCV(regressor, param_grid=param, cv=5, scoring='neg_mean_squared_error')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Fit the model using GridSearchCV to find the best parameters
grid.fit(x_train, y_train[:, torsion_number])

# Get the best parameters from the grid search
best_params = grid.best_params_

# Print the best parameters to a file
with open('best_params.txt', 'w') as f:
    f.write(f"Best parameters found: {best_params}")
