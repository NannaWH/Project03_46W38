################################################################################## Cleaning Data and Splitting it into Traning and Evaluation data #############
################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

## We create a function to import meteorological and wind power data
def load_data(filename:str): 
    """Load meterological and wind data.
    
    Args:
        filename: the name of the file with meterological and wind data (Location1.csv, Location2.csv, Location3.csv, Location4.csv)

    Returns:
        data: cleaned raw data to be used in the prediction models-
    
    """

    data = pd.read_csv(f'inputs/{filename}', sep=',')

    #We change the wind direction from degrees to radians
    data['wdir_radians_10m'] = np.deg2rad(data['winddirection_10m'])
    data['wdir_radians_100m'] = np.deg2rad(data['winddirection_100m'])

    #Change the wind direction into cos and sin 
    data['sin_wdir_10m'] = np.sin(data['wdir_radians_10m'])
    data['cos_wdir_10m'] = np.cos(data['wdir_radians_10m'])
    data['sin_wdir_100m'] = np.sin(data['wdir_radians_100m'])
    data['cos_wdir_100m'] = np.cos(data['wdir_radians_100m'])

    # We extract from month and hour of day variables from the Time Variables
    data[['date', 'time']] = data['Time'].str.split(" ", expand=True)
    data['year'] = data['date'].str.split("-").str[0].astype("int64")
    data['month'] = data['date'].str.split("-").str[1].astype("int64")
    data['day'] = data['date'].str.split("-").str[2].astype("int64")
    data['hour'] = data['time'].str.split(":").str[0].astype("int64")

    # Ensure 'Time' is a datetime type
    data['Time'] = pd.to_datetime(data['Time'])

    return data


def data_split(data, splittype, prediction_horizon):
    """Splitting data into Traning (80%) and Evaluation (20%) Data
    
    data = weather and power output data
    splittype = sequential or timeseries split
    prediction_horizon = how many years in the future should be predicted e.g., 1 year or 6 years
    """
    
    # Ensure your DataFrame is sorted by time
    data = data.sort_values("Time") #data.sort_index()

    # Define the power target (forecasting 1, 2, 3, 4, 5 or 6 hours ahead?)
    data["Power_target"] = data["Power"].shift(-prediction_horizon)

    data['Power_l1'] = data["Power"].shift(1) #add a lagged power variable 
    #data['Power_l2'] = data["Power"].shift(2) #add a lagged power variable 
    #data['Power_l3'] = data["Power"].shift(3) #add a lagged power variable 
    #data['Power_l4'] = data["Power"].shift(4) #add a lagged power variable 
    #data['Power_l5'] = data["Power"].shift(5) #add a lagged power variable 

    data = data.dropna() #drop n/a values

    #Drop the power value
    if prediction_horizon == 0:
        data = data.drop('Power', axis=1) 

    #Splitting data and ensuring no shuffle as we are working with time series data
    if splittype == "Sequential":
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['Power_target'], axis = 'columns'), data.Power_target, test_size=0.2, shuffle=False)

    elif splittype == "Timeseries":
        X = data.drop(['Power_target'], axis=1)
        y = data['Power_target']

        tscv = TimeSeriesSplit(n_splits=2, test_size=int(len(X) * 0.2), gap=2)
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
    return X_train, y_train, X_test, y_test 
