################################################################################## Creating functions to run results visualisations ############################
################################################################################

# Importing packages
import matplotlib.pyplot as plt


def scatter_actualvspred(y_pred, y_test, model_name, prediction_horizon,):
    """Creating a scatter plot of the predicted and actutual power output
    
    Args:
        y_pred: predicted power output
        y_test: actual power output from test data
        model_name: the prediction model name (persistence, random_forest, neural_network)
    
    Return: 
        Saved scatterplot of predicted power output vs. acutal power output.
    """

    # We extract the values from the y_test variable
    y_test = y_test.values

    # We set up the plot
    plt.figure()
    plt.scatter(y_pred, y_test)
    plt.xlabel("Power Predicted (percentage of maximum potential output)")
    plt.ylabel("Power Actual (percentage of maximum potential output)")
    plt.title("Scatter Plot: Predicted Power vs. Actual Power")

    # We save the plot
    plt.savefig(f"outputs/{model_name}_model/scatter_actualvspred_hour{prediction_horizon}.png")
    plt.close()
    
    return print(f"Scatter plot actual vs. predicted power output saved for model: {model_name}")


def plot_actualvspred(y_pred, y_test, model_name, subset_start, subset_end, prediction_horizon):
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

    # We extract the values from the y_test variable
    y_test = y_test.values

    # We set up the plot
    fig, ax = plt.subplots()

    # We add labels and title
    fig.suptitle("Predicted Power vs. Actual Power", fontsize=11)
    ax.set_xlabel('Data Point', fontsize=8)
    ax.set_ylabel('Power Output (percentage of maximum potential output)', fontsize=8)

    # We define the graph lines
    line1, = ax.plot(y_test, color='black', linestyle='-', label="Actual Power")
    line2, = ax.plot(y_pred, color='orange', linestyle=':', label="Predicted Power")

    # We set the size of the tick labelse
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)

    # We combine legends and add a legend box to the right of the graph 
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    fig.subplots_adjust(right=0.75)
    ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.03, 0.5), borderaxespad=0, fontsize=8)

    # We save the plot
    plt.savefig(f"outputs/{model_name}_model/actualvspred_hour{prediction_horizon}.png")
    plt.close()

    # We make a subset of the predicted and actual power data
    # We define the subset
    y_test_subset = y_test[subset_start: subset_end]
    y_pred_subset = y_pred[subset_start: subset_end]

    # We set up the plot
    fig, ax = plt.subplots()

    # We add labels and title
    fig.suptitle("Predictions vs. Actual Power for Sample Subset", fontsize=11)
    ax.set_xlabel('Data Point', fontsize=8)
    ax.set_ylabel('Power Output (percentage of maximum potential output)', fontsize=8)

    # We define the graph lines
    line1, =ax.plot(y_test_subset, color='black', linestyle='-', label="Actual Power")
    line2, = ax.plot(y_pred_subset, color='orange', linestyle=':', label="Predicted Power")

    # We set the size of the tick labelse
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)

    # We combine legends and add a box to the right of the graph with the legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    fig.subplots_adjust(right=0.75)
    ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.03, 0.5), borderaxespad=0, fontsize=8)

    # We save the plot
    plt.savefig(f"outputs/{model_name}_model/actualvspred_subset_{subset_start}_{subset_end}_hour{prediction_horizon}.png")
    plt.close()

    return print(f"Time Series plot actual vs. predicted power output saved for model: {model_name}")