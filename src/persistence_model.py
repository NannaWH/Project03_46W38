################################################################################## Creating a simple persistence forecasting model ## 
################################################################################

#Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

#Import modules 
from data_cleaning import load_data, data_split

## Importing data 
if __name__ == "__main__": 
    data = load_data('Location4.csv')
    X_train, y_train, X_test, y_test = data_split(data)

# Persistence prediction
prev = y_train.iloc[-1]          # last value from training set
y_pred = y_test.shift(1)         # lag 1
y_pred.iloc[0] = prev            # first prediction

## We calculate the Mean Absoluate Error to test the prediction model
MAE_pers_model = mean_absolute_error(y_test, y_pred)
MSE_pers_model = mean_squared_error(y_test, y_pred)
RMSE_pers_model = root_mean_squared_error(y_test, y_pred)
R2_pers_model = r2_score(y_pred, y_test)

print(MAE_pers_model, MSE_pers_model, RMSE_pers_model, R2_pers_model)


plt.figure()
plt.scatter(y_pred, y_test)
plt.xlabel("Power Predicted")
plt.ylabel("Power Actual")
plt.title(f"Scatter Plot: Power predicted vs actual")
plt.show()


fig, ax = plt.subplots()
fig.suptitle("Predictions vs. Actual Power")
ax.plot(y_test, color = 'black', linestyle = '-', label="Actual Power")
ax.plot(y_pred, color = 'orange', linestyle = ':', label="Predicted Power")
plt.savefig("outputs/persistence_model/power_forecast_full.png") 


# We make a subset of the predicted and actual power data
y_test_subset = y_test[2000:4000]
y_pred_subset = y_pred[2000:4000]
fig, ax = plt.subplots()
fig.suptitle("Predictions vs. Actual Power")
ax.plot(y_test_subset, color = 'black', linestyle = '-', label="Actual Power")
ax.plot(y_pred_subset, color = 'orange', linestyle = ':', label="Predicted Power")
plt.savefig("outputs/persistence_model/power_forecast_subset.png")