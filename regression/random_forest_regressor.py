import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

x_train = pd.read_csv("x_train_pca.csv")
y_train=pd.read_csv("y_train.csv")
x_test=pd.read_csv("x_test_pca.csv")
y_test = pd.read_csv("y_test.csv")

def evaluate_model(true,predicted):
  mae=mean_absolute_error(true,predicted)
  mse=mean_squared_error(true,predicted)
  r2_square=r2_score(true,predicted)
  rmse=np.sqrt(mse)
  return mae,mse,rmse,r2_square

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

models = []
y_train_preds = []
y_test_preds = []

# Train a Decision Tree Regressor for each target in y
for i in range(y_train.shape[1]):
    model = RandomForestRegressor(random_state=42)
    # model = DecisionTreeRegressor()

    model.fit(x_train, y_train[:, i])
    models.append(model)

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    y_train_preds.append(y_train_pred)
    y_test_preds.append(y_test_pred)

    # Evaluate the model
    train_mae,train_mse,train_rmse,train_r2_square = evaluate_model(y_train[:, i], y_train_pred)
    test_mae,test_mse,test_rmse,test_r2_square = evaluate_model(y_test[:, i], y_test_pred)

    # Print the results
    # print
    print(f"Target {i} - Train MAE: {train_mae:.9f}, Test MAE: {test_mae:.9f}")
    print(f"Target {i} - Train MSE: {train_mse:.9f}, Test MSE: {test_mse:.9f}")
    print(f"Target {i} - Train RMSE: {train_rmse:.9f}, Test RMSE: {test_rmse:.9f}")
    print(f"Target {i} - Train R2 Square: {train_r2_square:.9f}, Test R2 Square: {test_r2_square:.9f}")

    print('------------------------------------------------------------------------------------------------------------------------')
    # print(f"Target {i} - Train MSE: {train_mse:.9f}, Test MSE: {test_mse:.9f}")