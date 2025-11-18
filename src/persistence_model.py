################################################################################## Creating a simple persistence forecasting model ## 
################################################################################

## Splitting data into Traning (80%) and Evaluation (20%) Data
import pandas as pd
from sklearn.model_selection import train_test_split
from data_vizualization import load_data
import matplotlib.pyplot as plt

## Importing data 
if __name__ == "__main__": 
    data = load_data('Location2.csv')

# Ensure your DataFrame is sorted by time
data = data.sort_index()

# Define the split point (80% for training, 20% for testing)
split_point = int(len(data) * 0.8)

train_data = data.iloc[:split_point]
test_data = data.iloc[split_point:]

# Separate features (X) and target (y) for both sets
X_train = train_data.drop('Power', axis=1)
y_train = train_data[['Time', 'Power']]

X_test = test_data.drop('Power', axis=1)
y_test = test_data[['Time', 'Power']]

## Creating a persistence forecasting (y_t = y_t-1)
y_test['Power_lag1'] = y_test['Power'].shift(1)

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