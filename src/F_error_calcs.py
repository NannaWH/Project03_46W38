################################################################################## Creating functions to run predictions and error measures ################### 
################################################################################

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

## We define a function that calculates errors to test the performance of the prediction model
def error_measures(y_pred, y_test, model_name):
    """Calculating errors 
    
    y_pred = predicted power output
    y_pred = actual power output from test data
    model_name = the prediction model name (persistence, random_forest, neural_network)

    """

    y_test = y_test.values

    # Calculating errors 
    MAE = mean_absolute_error(y_pred, y_test)
    MSE = mean_squared_error(y_pred, y_test)
    RMSE = root_mean_squared_error(y_pred, y_test)
    R2 = r2_score(y_pred, y_test)

    # Putting the errors into a dataframe
    error_output = {'Prediction Model Name': [model_name],
        'Mean Absolute Error': [MAE],
        'Mean Squared Error': [MSE],
        'Root Mean Square Error': [RMSE],
        'R Squared': [R2]}
    
    df_error_output = pd.DataFrame(error_output)

    # Saving the data
    existing = pd.read_excel('outputs/error_output.xlsx')
    updated = pd.concat([existing, df_error_output], ignore_index=True)
    updated.to_excel('outputs/error_output.xlsx', index=False)

    return f"Errors calculated and saved in error_output.xlsx for model: {model_name}"

    
