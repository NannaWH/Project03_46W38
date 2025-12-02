################################################################################## Creating a simple persistence forecasting model ############################ 
################################################################################


def persistence_model(y_train, y_test, lag):
    """ Creating a persistence prediction model i.e. power output t = power output t-1

    Args:
        y_train: actual power output from train data
        y_test: actual power output from test data

    Returns:
        y_pred: predicted power outcome 
    """

    prev = y_train.iloc[-lag:]          # last value from training set
    y_pred = y_test.shift(lag)          # lag 
    y_pred.iloc[:lag] = prev            # first prediction
    y_pred = y_pred.values

    print("Persistence model done")
    
    return y_pred




