################################################################################## Creating a Machine Learning Forecasting Model using Random Forest Model #####
################################################################################

#Importing packages
from sklearn.ensemble import RandomForestRegressor


def random_forest_model_1hour(X_train, y_train, X_test):
    """ Creating a random forest prediction model for 1 hour ahead prediction

    X_train = input variables (meterological and wind variables) in training data
    y_train = output variable (power output) in training data
    X_test = input variables (meterological and wind variables) in test data
    y_train = output variable (power output) in testing data
    """

    #We drop the time column (Timestamp)
    X_train = X_train.drop(['Time','date', 'time', 'year', 'day', 'relativehumidity_2m','windgusts_10m','winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)
    
    X_test  = X_test.drop(['Time','date', 'time', 'year', 'day','relativehumidity_2m', 'windgusts_10m', 'winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)

    # We make a Random Forest Prediction Model
    model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=12)
    model.fit(X_train,y_train)

    # We make a prediction of y
    y_pred = model.predict(X_test)

    return y_pred



