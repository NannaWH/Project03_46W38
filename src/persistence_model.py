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


### OBS FIGURE OUT HOW TO MAKE THIS GRAPH
"""
## We make a graph to vizualize the difference between the predicted and actual power output

# Ensure 'Time' is a datetime type
y_test['Time'] = pd.to_datetime(y_test['Time'])

# Filter data for the specified time range
mask = (y_test['Time'] >= pd.to_datetime("2021-12-29 00:00")) & (y_test['Time'] <= pd.to_datetime("2021-12-31 00:00"))
y_test = y_test.loc[mask]

# Create the plot
fig, ax1 = plt.subplots()

# Plot normalized power
line1, = ax1.plot(y_test['Time'], y_test['Power'], 
                color='black', linestyle='-', label='Actual Power Output and Lagged (AR1) Power Output', linewidth=0.8)
line2, = ax1.plot(y_test['Time'], y_test['Power_lag1'], 
                color='green', linestyle='--', label='Actual Power Output and Lagged (AR1) Power Output', linewidth=0.8)

# Labels and title
ax1.set_xlabel('Time', fontsize=8)
ax1.set_ylabel('Power Output (Normalized 0-1)', fontsize=8)
plt.title("Actual and Predicted Power Output using Persistence Forecasting", fontsize=11)

# We set the size of the tick labelse
ax1.tick_params(axis='both', which='major', labelsize=7)

# Save figure
plt.savefig("outputs/persistence_model/power_forecast.png", bbox_inches='tight', dpi=300)
"""