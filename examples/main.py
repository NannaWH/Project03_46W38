################################################################################## Running Machine Learning Prediction Model ################################## 
################################################################################

# We import relevant packages
import pandas as pd
import sys
import os

# project_root to call function from another folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# We import the relevant modules that are used in the model
from src.A_data_cleaning import load_and_clean_data, data_split
from src.B_data_vizualization import timeseries_plot, scatter_plots, wind_rose_plot
from src.C_persistence_model import persistence_model
from src.D_random_forest_model import random_forest_model_1hour,  random_forest_model_6hour
from src.E_neural_network_model import neural_network_model_1hour, neural_network_model_6hour
from src.F_error_calcs import error_measures
from src.G_results_vizualization import scatter_actualvspred, plot_actualvspred

if __name__ == "__main__": 
    
    # Data Loading - We load the data that should be used by our machine learning models
    data = load_and_clean_data('Location4.csv')

    # We plot the raw data and save the plots in the output folder to understand the trends and relationships in the data
    timeseries_plot(data, start_time="2017-08-01 00:00", end_time="2017-09-01 00:00")
    scatter_plots(data)
    wind_rose_plot(data)

    # We make an empty file (xlsx) for error outputs 
    empty_df = pd.DataFrame()
    empty_df.to_excel("outputs/error_output.xlsx", index=False)

    ### Forecast: 1 hour ahead 
    # Assuming no weather forecast is provided - forecast is based on current and past weather data and power output

    print("Forecasting 1 hour ahead")

    # We split data into traning and test data
    X_train, y_train, X_test, y_test = data_split(data=data, splittype="Timeseries", prediction_horizon=1) 

    ### Persistence Model
    # We use the persistence model to predict power output
    y_pred = persistence_model(y_train=y_train, y_test=y_test, lag=1)

    # We calculate errors 
    error_measures(y_pred, y_test, model_name="persistence", prediction_horizon=1)

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name="persistence", prediction_horizon=1)
    plot_actualvspred(y_pred, y_test, model_name="persistence", prediction_horizon=1, subset_start=2000, subset_end=4000)
    plot_actualvspred(y_pred, y_test, model_name="persistence", prediction_horizon=1, subset_start=500, subset_end=700)
    
    ### Random Forest Model

    # We use the random forest model to predict power output
    y_pred = random_forest_model_1hour(X_train=X_train, y_train=y_train, X_test=X_test)

    # We calculate errors 
    error_measures(y_pred, y_test, model_name="random_forest", prediction_horizon=1)

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name="random_forest", prediction_horizon=1)
    plot_actualvspred(y_pred, y_test, model_name="random_forest", prediction_horizon=1, subset_start=2000, subset_end=4000)
    plot_actualvspred(y_pred, y_test, model_name="random_forest", prediction_horizon=1, subset_start=500, subset_end=700)
    
    ### Neural Network Model

    # We use the neural network class to predict power output
    NNM = neural_network_model_1hour(X_train=X_train, y_train=y_train, X_test=X_test, model_regressor="Keras")

    y_pred = NNM.predict_ypred(X_train=X_train, y_train=y_train, X_test=X_test, model_regressor="Keras")

    # We calculate errors 
    error_measures(y_pred, y_test, model_name="neural_network", prediction_horizon=1)

    # We Visualise the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name="neural_network", prediction_horizon=1)
    plot_actualvspred(y_pred, y_test, model_name="neural_network", prediction_horizon=1, subset_start=2000, subset_end=4000)
    plot_actualvspred(y_pred, y_test, model_name="neural_network", prediction_horizon=1, subset_start=500, subset_end=700)
    
    ### Forecast: 6 hours ahead 
    # Assuming no weather forecast is provided - forecast is based on current and past weather data and power output

    print("Forecasting 6 hours ahead")

    # We split data into traning and test data
    X_train, y_train, X_test, y_test = data_split(data=data, splittype="Timeseries", prediction_horizon=6) 

    ### Persistence Model

    # We use the persistence model to predict power output
    y_pred = persistence_model(y_train=y_train, y_test=y_test, lag=6)

    # We calculate errors 
    error_measures(y_pred, y_test, model_name="persistence", prediction_horizon = 6)

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name="persistence", prediction_horizon=6)
    plot_actualvspred(y_pred, y_test, model_name="persistence", prediction_horizon=6, subset_start=2000, subset_end=4000)
    plot_actualvspred(y_pred, y_test, model_name="persistence", prediction_horizon=6, subset_start=500, subset_end=700)

    ### Random Forest Model

    # We use the random forest model to predict power output
    y_pred = random_forest_model_6hour(X_train=X_train, y_train=y_train, X_test=X_test)

    # We calculate errors 
    error_measures(y_pred, y_test, model_name="random_forest", prediction_horizon=6)

    # We Visualise the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name="random_forest", prediction_horizon=6)
    plot_actualvspred(y_pred, y_test, model_name="random_forest", prediction_horizon=6, subset_start=2000, subset_end=4000)
    plot_actualvspred(y_pred, y_test, model_name="random_forest", prediction_horizon=6, subset_start=500, subset_end=700)
   
    ### Neural Network Model

    # We use the neural network class to predict power output
    NNM = neural_network_model_6hour(X_train=X_train, y_train=y_train, X_test=X_test, model_regressor="Keras")

    y_pred = NNM.predict_ypred(X_train=X_train, y_train=y_train, X_test=X_test, model_regressor="Keras")

    # We calculate errors 
    error_measures(y_pred, y_test, model_name="neural_network", prediction_horizon=6)

    # We Vizualize the predicted and actual power output
    scatter_actualvspred(y_pred, y_test, model_name="neural_network", prediction_horizon=6)
    plot_actualvspred(y_pred, y_test, model_name="neural_network", prediction_horizon=6, subset_start=2000, subset_end=4000)
    plot_actualvspred(y_pred, y_test, model_name="neural_network", prediction_horizon=6, subset_start=500, subset_end=700)
