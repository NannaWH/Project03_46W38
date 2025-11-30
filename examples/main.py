################################################################################## Running Machine Learning Prediction Model ## 
################################################################################

import numpy as np
import pandas as pd
import sys
import os


### OBS to do: Put a lot of lagged power input in the data cleaning model instead of in here.

# project_root to call function from another folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.A_data_cleaning import load_data, data_split
from src.B_data_vizualization import timeseries_plot, scatter_plots
from src.C_persistence_model import persistence_model
from src.D_random_forest_model import random_forest_model_1hour
from src.E_neural_network_model import neural_network_model_1hour
from src.F_error_calcs import error_measures
from src.G_results_vizualization import scatter_actualvspred, plot_actualvspred

if __name__ == "__main__": 
    ### Data Loading - We load the data
    data = load_data('Location4.csv')

    ### We plot the raw data and save the plots in the output folder
    timeseries_plot(data, start_time="2017-08-01 00:00", end_time="2017-09-01 00:00")
    scatter_plots(data)

    ### We make an empty file (xlsx) for error outputs
    empty_df = pd.DataFrame()
    empty_df.to_excel("outputs/error_output.xlsx", index=False)

    ### Forecast: 1 hour ahead 
    ### Assuming external weather forecast is provided and is 100% correct 

    ### We split data into traning and test data
    X_train, y_train, X_test, y_test = data_split(data = data, splittype = "Sequential", timehorizon = 0) 


    ### Persistence Model

    # We use the persistence model to predict power output
    y_pred = persistence_model(y_train=y_train, y_test=y_test)

    # We calculate errors 
    error_measures(y_pred, y_test, model_name = "persistence")

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name = "persistence")
    plot_actualvspred(y_pred, y_test, model_name = "persistence", subset_start=2000, subset_end=4000)


    ### Random Forest Model

    # We use the random forest model to predict power output
    y_pred = random_forest_model_1hour(X_train=X_train, y_train=y_train, X_test=X_test)

    # We calculate errors 
    error_measures(y_pred, y_test, model_name = "random_forest")

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name = "random_forest")
    plot_actualvspred(y_pred, y_test, model_name = "random_forest", subset_start=2000, subset_end=4000)

    
    ### Neural Network Model

    # We use the random forest model to predict power output
    y_pred = neural_network_model_1hour(X_train=X_train, y_train=y_train, X_test=X_test, model_regressor= "Keras")

    # We calculate errors 
    error_measures(y_pred, y_test, model_name = "neural_network")

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name = "neural_network")
    plot_actualvspred(y_pred, y_test, model_name = "neural_network", subset_start=2000, subset_end=4000)

    ### Forecast: 1 hour ahead 
    ### Assuming no weather forecast is provided - forecast is based on current and past weather and power output

    ### We split data into traning and test data
    X_train, y_train, X_test, y_test = data_split(data = data, splittype = "Sequential", timehorizon = 1) 

    ### Random Forest Model

    # We use the random forest model to predict power output
    y_pred = random_forest_model_1hour(X_train=X_train, y_train=y_train, X_test=X_test)

    # We calculate errors 
    error_measures(y_pred, y_test, model_name = "random_forest")

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name = "random_forest")
    plot_actualvspred(y_pred, y_test, model_name = "random_forest", subset_start=2000, subset_end=4000)

    
    ### Neural Network Model

    # We use the random forest model to predict power output
    y_pred = neural_network_model_1hour(X_train=X_train, y_train=y_train, X_test=X_test, model_regressor= "Keras")

    # We calculate errors 
    error_measures(y_pred, y_test, model_name = "neural_network")

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name = "neural_network")
    plot_actualvspred(y_pred, y_test, model_name = "neural_network", subset_start=2000, subset_end=4000)








