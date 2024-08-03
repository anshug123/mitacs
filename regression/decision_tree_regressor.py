import pandas as pd
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

x_train = pd.read_csv("x_train_pca.csv")
y_train=pd.read_csv("y_train.csv")
x_test=pd.read_csv("x_test_pca.csv")
y_test = pd.read_csv("y_test.csv")

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

# Initialize lists to store the models and their predictions
models = []
y_train_preds = []
y_test_preds = []

# Train a Decision Tree Regressor for each target in y
for i in range(y_train.shape[1]):
    model = DecisionTreeRegressor(random_state=42)
    # model = DecisionTreeRegressor()

    model.fit(x_train, y_train[:, i])
    models.append(model)

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    y_train_preds.append(y_train_pred)
    y_test_preds.append(y_test_pred)

    # Evaluate the model
    train_mse = mean_squared_error(y_train[:, i], y_train_pred)
    test_mse = mean_squared_error(y_test[:, i], y_test_pred)

    print(f"Target {i} - Train MSE: {train_mse:.9f}, Test MSE: {test_mse:.9f}")
