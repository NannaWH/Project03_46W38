################################################################################## Creating functions to run predictions and error measures ################### 
################################################################################

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# We define a function that calculates errors to test the performance of the prediction model


def error_measures(y_pred, y_test, model_name, prediction_horizon):
    """Calculating errors 
    
    Args:
        y_pred: predicted power output
        y_test: actual power output from test data
        model_name: the prediction model name (persistence, random_forest, neural_network)

    Return: 
        Mean Absolute Error, Mean Squared Error, Root Mean Square Error, and R-squared calculated and saved in excel-file: error_output.xlsx
    """

    # We extract y_test values from the 
    y_test = y_test.values

    # We calculate errors 
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)

    # We put the errors into a dataframe
    error_output = {'Prediction Horizon': [prediction_horizon], 'Prediction Model Name': [model_name], 'Mean Absolute Error': [MAE], 'Mean Squared Error': [MSE], 'Root Mean Square Error': [RMSE], 'R Squared': [R2]}
    
    df_error_output = pd.DataFrame(error_output)

    # We saving the data into an excisting datafile without overriding
    existing = pd.read_excel('outputs/error_output.xlsx')
    updated = pd.concat([existing, df_error_output], ignore_index=True)
    updated.to_excel('outputs/error_output.xlsx', index=False)

    return print(f"Errors calculated and saved in error_output.xlsx for model: {model_name}")
