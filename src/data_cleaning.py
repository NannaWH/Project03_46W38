################################################################################## Cleaning Data and Splitting it into Traning and Evaluation data ## 
################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

## We create a function to import meteorological and wind power data
def load_data(filename:str): 
    """Load meterological and wind data."""
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


def data_split(data):
    """Splitting data into Traning (80%) and Evaluation (20%) Data"""
    # Ensure your DataFrame is sorted by time
    data = data.sort_index()

    #Splitting data and ensuring no shuffle as we are working with time series data
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Power'], axis = 'columns'), data.Power, test_size=0.2, shuffle=False)

    return X_train, y_train, X_test, y_test 

"""
def data_split(data):
    ""Splitting data into Traning (80%) and Evaluation (20%) Data""
    # Ensure your DataFrame is sorted by time
    data = data.sort_index()

    # Define the split point (80% for training, 20% for testing)
    split_point = int(len(data) * 0.8)

    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]

    # Separate features (X) and target (y) for both sets
    X_train = train_data.drop('Power', axis=1)
    y_train = train_data[['Time', 'Power']].copy()

    X_test = test_data.drop('Power', axis=1)
    y_test = test_data[['Time', 'Power']].copy()

    return X_train, y_train, X_test, y_test 
"""