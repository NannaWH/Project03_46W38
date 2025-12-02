################################################################################## Creating functions to run results vizualizations ############################
################################################################################

#Importing packages
import matplotlib.pyplot as plt

def scatter_actualvspred(y_pred, y_test, model_name):
    """Creating a scatter plot of the predicted and actutual power output
    
    Args:
        y_pred: predicted power output
        y_test: actual power output from test data
        model_name: the prediction model name (persistence, random_forest, neural_network)
    
    Return: 
        Saved scatterplot of predicted power output vs. acutal power output.
    """
    y_test = y_test.values

    plt.figure()
    plt.scatter(y_pred, y_test)
    plt.xlabel("Power Predicted")
    plt.ylabel("Power Actual")
    plt.title(f"Scatter Plot: Power predicted vs actual")

    plt.savefig(f"outputs/{model_name}_model/scatter_actualvspred.png")
    
    return print(f"Scatter plot actual vs. predicted power output saved for model: {model_name}")


def plot_actualvspred(y_pred, y_test, model_name, subset_start, subset_end):
    """Creating a line plot of the predicted and actutual power output
    
    Args:
        y_pred: predicted power output
        y_pred: actual power output from test data
        model_name: the prediction model name (persistence, random_forest, neural_network)
        subset_start: from what data point the plot subset should start
        subset_end: from what data point the plot subset should end

    Return:
        Two time series plot showing the predicted power output vs. the actual power output. One for the whole test time series data and one for a subset of the data.
    """

    y_test = y_test.values

    fig, ax = plt.subplots()
    fig.suptitle("Predictions vs. Actual Power")
    ax.plot(y_test, color = 'black', linestyle = '-', label="Actual Power")
    ax.plot(y_pred, color = 'orange', linestyle = ':', label="Predicted Power")
    plt.savefig(f"outputs/{model_name}_model/actualvspred.png")


    # We make a subset of the predicted and actual power data
    y_test_subset = y_test[subset_start:subset_end]
    y_pred_subset = y_pred[subset_start:subset_end]
    fig, ax = plt.subplots()
    fig.suptitle("Predictions vs. Actual Power")
    ax.plot(y_test_subset, color = 'black', linestyle = '-', label="Actual Power")
    ax.plot(y_pred_subset, color = 'orange', linestyle = ':', label="Predicted Power")
    plt.savefig(f"outputs/{model_name}_model/actualvspred_subset_{subset_start}_{subset_end}.png")

    return print(f"Time Series plot actual vs. predicted power output saved for model: {model_name}")