###############################################################################
## Creating a Machine Learning Forecasting Model using Random Forest Model ###
###############################################################################

# Importing packages
from sklearn.ensemble import RandomForestRegressor


def random_forest_model_1hour(X_train, y_train, X_test):
    """ Creating a random forest prediction model for 1 hour ahead prediction

    Args:
        X_train: training data - input variables (meterological and wind variables).
        y_train: training data - output variable (power output).
        X_test: test data - input variables (meterological and wind variables).
        y_test: test data - output variable (power output).
    
    Returns:
        y_pred: predicted power outcome.
    """

    # We drop columns that are not relevant for the model
    X_train = X_train.drop(['Time', 'date', 'time', 'year', 'day', 'hour',  'relativehumidity_2m', 'windgusts_10m', 'winddirection_10m', 'winddirection_100m', 'wdir_radians_10m', 'wdir_radians_100m', 'Power_l3', 'Power_l4', 'Power_l5', 'Power_l6', 'power_change_l1', 'power_change_l2', 'power_change_momentum', 'windspeed_change_l1', 'windspeed_change_l2', 'windspeed_change_momentum', 'std_power_l0-l2', 'std_windspeed_l0-l2'], axis=1)
    
    X_test  = X_test.drop(['Time', 'date', 'time', 'year', 'day', 'hour', 'relativehumidity_2m', 'windgusts_10m', 'winddirection_10m', 'winddirection_100m', 'wdir_radians_10m', 'wdir_radians_100m', 'Power_l3', 'Power_l4', 'Power_l5', 'Power_l6', 'power_change_l1', 'power_change_l2', 'power_change_momentum', 'windspeed_change_l1', 'windspeed_change_l2', 'windspeed_change_momentum', 'std_power_l0-l2', 'std_windspeed_l0-l2'], axis=1)

    # We make a Random Forest Prediction Model
    RF_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=12)
    
    # We train the model using our training data
    RF_model.fit(X_train, y_train)

    # We make a prediction of y
    y_pred = RF_model.predict(X_test)

    print("Random forest model is done")

    return y_pred


def random_forest_model_6hour(X_train, y_train, X_test):
    """ Creating a random forest prediction model for 1 hour ahead prediction

    Args:
        X_train: training data - input variables (meterological and wind variables) 
        y_train: training data - output variable (power output) 
        X_test: test data - input variables (meterological and wind variables) 
        y_test: test data - output variable (power output) 
    
    Returns:
        y_pred: predicted power outcome 
    """

    # We drop columns that are not relevant for the model
    X_train = X_train.drop(['Time', 'date', 'time', 'year', 'day', 'hour', 'windgusts_10m', 'relativehumidity_2m', 'winddirection_10m', 'winddirection_100m', 'wdir_radians_10m', 'wdir_radians_100m', 'windspeed_100m_l1', 'windspeed_change_l1', 'windspeed_change_momentum', 'Power_l5', 'Power_l4', 'Power_l1', 'windspeed_change_l2', 'std_windspeed_l0-l2'], axis=1)
    
    X_test  = X_test.drop(['Time', 'date', 'time', 'year', 'day', 'hour', 'windgusts_10m', 'relativehumidity_2m', 'winddirection_10m', 'winddirection_100m', 'wdir_radians_10m', 'wdir_radians_100m', 'windspeed_100m_l1', 'windspeed_change_l1', 'windspeed_change_momentum', 'Power_l5', 'Power_l4', 'Power_l1', 'windspeed_change_l2', 'std_windspeed_l0-l2'], axis=1)

    # We make a Random Forest Prediction Model
    RF_model = RandomForestRegressor(n_estimators=400, max_depth=50, min_samples_split=3, min_samples_leaf=2, random_state=12)
    
    # We train the model using our training data
    RF_model.fit(X_train, y_train)

    # We make a prediction of y
    y_pred = RF_model.predict(X_test)

    print("Random forest model is done")

    return y_pred




