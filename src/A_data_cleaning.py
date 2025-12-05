######################################################################
## Cleaning Data and Splitting it into Traning and Evaluation data ##
######################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

# We create a function to import meteorological and wind power data


def load_and_clean_data(filename: str):
    """Load meterological and wind data.

    Args:
        filename: the name of the file with meterological and wind data (Location1.csv, Location2.csv, Location3.csv, Location4.csv).

    Returns:
        data: cleaned raw data to be used in the prediction models.
    """

    # We load the data
    data = pd.read_csv(f'inputs/{filename}', sep=',')

    # We change the wind direction from degrees to radians
    data['wdir_radians_10m'] = np.deg2rad(data['winddirection_10m'])
    data['wdir_radians_100m'] = np.deg2rad(data['winddirection_100m'])

    # We change the wind direction into cos and sin (make it into a circle)
    data['sin_wdir_10m'] = np.sin(data['wdir_radians_10m'])
    data['cos_wdir_10m'] = np.cos(data['wdir_radians_10m'])
    data['sin_wdir_100m'] = np.sin(data['wdir_radians_100m'])
    data['cos_wdir_100m'] = np.cos(data['wdir_radians_100m'])

    # We extract month and hour of day variables from the Time Variable
    data[['date', 'time']] = data['Time'].str.split(" ", expand=True)
    data['year'] = data['date'].str.split("-").str[0].astype("int64")
    data['month'] = data['date'].str.split("-").str[1].astype("int64")
    data['day'] = data['date'].str.split("-").str[2].astype("int64")
    data['hour'] = data['time'].str.split(":").str[0].astype("int64")

    # We convert the hour variable into a "circle" (a full circle of 360 degrees = 2 * pi radians )
    data['sin_hour'] = np.sin(2 * np.pi * (data['hour']/24))
    data['cos_hour'] = np.cos(2 * np.pi * (data['hour']/24))

    # We ensure 'Time' is a datetime type
    data['Time'] = pd.to_datetime(data['Time'])

    # We ensure the DataFrame is sorted by time
    data = data.sort_values("Time")

    # We lag power output variables 1–6
    for i in range(1, 7):
        data[f'Power_l{i}'] = data['Power'].shift(i)

    # We lag windspeed variable 1-2
    for i in range(1, 3):
        data[f'windspeed_100m_l{i}'] = data['windspeed_100m'].shift(i)

    # We add variables to reflect changes in power in the previous hours
    data["power_change_l1"] = data['Power'] - data['Power_l1']
    data["power_change_l2"] = data['Power_l1'] - data['Power_l2']
    data["power_change_momentum"] = data["power_change_l1"] - data["power_change_l2"]

    # We add variables to reflect changes in windspeeds in the previous hours
    data["windspeed_change_l1"] = data['Power'] - data['windspeed_100m_l1']
    data["windspeed_change_l2"] = data['windspeed_100m_l1'] - data['windspeed_100m_l2']
    data["windspeed_change_momentum"] = data["windspeed_change_l1"] - data["windspeed_change_l2"]

    # We add variables for the standard deviations in power and wind speed
    data["std_power_l0-l2"] = data[['Power', 'Power_l1', 'Power_l2']].std(axis=1)
    data["std_windspeed_l0-l2"] = data[['windspeed_100m', 'windspeed_100m_l1', 'windspeed_100m_l2']].std(axis=1)
    
    # We drop n/a values
    data = data.dropna()

    # We save data in a CSV file
    data.to_csv('inputs/cleaned_data.csv', index=False)

    print("Data is loaded and cleaned")

    return data


def data_split(data, splittype, prediction_horizon):
    """Splitting data into Traning (80%) and Evaluation (20%) Data

    Args:
        data: weather and power output data
        splittype: sequential or timeseries split
        prediction_horizon: how many years in the future should be predicted
        e.g., 1 year or 6 years.

    Returns:
        X_train: training dataset (the data the machine learning models aretested on) including the explanatory variables i.e., lagged power output and wind and weather data. 
        y_train: training y-variable (power outcome in the future).
        X_test: testing dataset (the data the model should predict power output based on) including the explanatory variables i.e., lagged power output and wind and weather data. 
        y_test: testing y-variable (power outcome in the future).
    """

    # We ensure the DataFrame is sorted by time
    data = data.sort_values("Time")

    # We define the power target (forecasting 1, 2, 3, 4, 5 or 6 hours ahead?)
    data["Power_target"] = data["Power"].shift(-prediction_horizon)

    # We rop n/a values
    data = data.dropna()

    # We drop the power value if prediction_horizon is 0 (zero)
    if prediction_horizon == 0:
        data = data.drop('Power', axis=1)

    # We split data and ensure no shuffle (time series data)
    if splittype == "Sequential":
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['Power_target'], axis='columns'), data.Power_target, test_size=0.2, shuffle=False)

    elif splittype == "Timeseries":
        X = data.drop(['Power_target'], axis=1)
        y = data['Power_target']

        tscv = TimeSeriesSplit(n_splits=2, test_size=int(len(X) * 0.2), gap=2)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("Data is split")

    return X_train, y_train, X_test, y_test
