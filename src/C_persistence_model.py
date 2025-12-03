################################################################################## Creating a simple persistence forecasting model ############################ 
################################################################################


def persistence_model(y_train, y_test, lag):
    """ Creating a persistence prediction model i.e. power output t = power output t-1.

    Args:
        y_train: actual power output from train data.
        y_test: actual power output from test data.

    Returns:
        y_pred: predicted power outcome .
    """

    # We extract the last x (lag) values from the training data
    prev = y_train.iloc[-lag:]

    # We define y_pred to be y_test shifted by x (lag) period(s)
    y_pred = y_test.shift(lag)

    # We take the first x (lag) from the training
    y_pred.iloc[:lag] = prev

    # We extract the values from y_pred
    y_pred = y_pred.values

    print("Persistence model done")
    
    return y_pred
