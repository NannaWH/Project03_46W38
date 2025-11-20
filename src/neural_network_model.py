################################################################################## Creating a Machine Learning Forecasting Model using Neural Networks ### 
################################################################################

#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

#Import modules 
from data_cleaning import load_data, data_split

## Importing and splitting the data 
if __name__ == "__main__": 
    data = load_data('Location4.csv')
    data['Power_l1'] = data["Power"].shift(1) #add a lagged power variable
    data = data.dropna() #drop n/a
    X_train, y_train, X_test, y_test = data_split(data) #split data

#We drop the time column (Timestamp)
X_train = X_train.drop(['Time','date', 'time', 'year', 'day', 'relativehumidity_2m','windgusts_10m','winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)
X_test = X_test.drop(['Time','date', 'time', 'year', 'day', 'relativehumidity_2m', 'windgusts_10m', 'winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)

# We make a Neural Network Prediction Model
model = MLPRegressor(hidden_layer_sizes=(5, 5),
                   activation='relu',
                   solver='adam',
                   max_iter=100,
                   random_state=12)

# We make a prediction of y
model.fit(X_train, y_train)

# We predict y and tranform it fron log value to the normalized power output
y_pred = model.predict(X_test)
y_test = y_test.values

## We calculate the error measures to test the prediction model
MAE_NN_model = mean_absolute_error(y_pred, y_test)
MSE_NN_model = mean_squared_error(y_pred, y_test)
RMSE_NN_model = root_mean_squared_error(y_pred, y_test)
R2_NN_model = r2_score(y_pred, y_test)

print(MAE_NN_model, MSE_NN_model, RMSE_NN_model, R2_NN_model)


plt.figure()
plt.scatter(y_pred, y_test)
plt.xlabel("Power Predicted")
plt.ylabel("Power Actual")
plt.title("Scatter Plot: Power predicted vs actual")
plt.show()


fig, ax = plt.subplots()
fig.suptitle("Predictions vs. Actual Power")
ax.plot(y_test, color = 'black', linestyle = '-', label="Actual Power")
ax.plot(y_pred, color = 'orange', linestyle = ':', label="Predicted Power")
plt.show()

y_test_subset = y_test[2000:4000]
y_pred_subset = y_pred[2000:4000]
fig, ax = plt.subplots()
fig.suptitle("Predictions vs. Actual Power")
ax.plot(y_test_subset, color = 'black', linestyle = '-', label="Actual Power")
ax.plot(y_pred_subset, color = 'orange', linestyle = ':', label="Predicted Power")
plt.show()