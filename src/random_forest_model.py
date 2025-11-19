################################################################################## Creating a Machine Learning Forecasting Model using Random Forest Model ### 
################################################################################

#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

#Import modules 
from data_cleaning import load_data, data_split

## Importing and splitting the data 
if __name__ == "__main__": 
    data = load_data('Location4.csv')
    # A lagged power variable
    data['Power_l1'] = data["Power"].shift(1)
    X_train, y_train, X_test, y_test = data_split(data)

#We drop the time column (Timestamp)
X_train = X_train.drop(['Time','date', 'time', 'year', 'day', 'relativehumidity_2m','windgusts_10m','winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)
X_test  = X_test.drop(['Time','date', 'time', 'year', 'day','relativehumidity_2m', 'windgusts_10m', 'winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)

# We make a Random Forest Prediction Model
model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=12)
model.fit(X_train,y_train)

# We make a prediction of y
y_pred = model.predict(X_test)

## We calculate the error measures to test the prediction model
MAE_RFR_model = mean_absolute_error(y_pred, y_test)
MSE_RFR_model = mean_squared_error(y_pred, y_test)
RMSE_RFR_model = root_mean_squared_error(y_pred, y_test)
R2_RFR_model = r2_score(y_pred, y_test)

print(MAE_RFR_model, MSE_RFR_model, RMSE_RFR_model, R2_RFR_model)

plt.figure()
plt.scatter(y_pred, y_test)
plt.xlabel("Power Predicted")
plt.ylabel("Power Actual")
plt.title(f"Scatter Plot: Power predicted vs actual")
plt.show()

"""
##We scale the data to minimize the impact of big outliers
# Create scaler
scaler = StandardScaler()

# Fit on data and transform
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""