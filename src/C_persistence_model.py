################################################################################## Creating a simple persistence forecasting model ############################ 
################################################################################


def persistence_model(y_train, y_test):
    """ Creating a persistence prediction model i.e. power output t = power output t-1

    y_train = actual power output from train data
    y_test = actual power output from test data
    """

    prev = y_train.iloc[-1]          # last value from training set
    y_pred = y_test.shift(1)         # lag 1
    y_pred.iloc[0] = prev            # first prediction
    y_pred = y_pred.values

    return y_pred



