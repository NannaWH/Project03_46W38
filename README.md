# Project03_46W38
This is the final project in the course 46W38 Scientific Programming in Wind Energy, Fall 2025 at DTU. 

This project develops three short-term wind power forecasting models:\
    1) **A persistence model** \
    2) **A random forest machine learning model** \
    3) **A neural network machine learning model** 

Using historical meterological and power data, the models are trained to predict wind power generation one hour and six hours ahead of the current hour. Furthermore, the models' performances are tested using graphs and error measures.

## Model Structure

The overall python model is structured into seven python files (located in the src folder) defining functions and classes as well as one main python file (located in the example folder) executing the functions and classes in a structured manner. The the model leverages data located in the input folder and generates and saves output into the output folder.

The model is divided into the following folders, modules, and packages:
- **examples**
    - ***main.py:*** this file executes the final machine learning models as well as calculating errors and saving plots used predefined functions. 
    
- **inputs**
    - ***cleaned_data.csv:*** this file includes the raw data cleaned and with additional explanatory variables.

    - ***LocationX.csv:*** historical power and meterologicsl data for location X (1-4).

- **outputs**
    - **neural_network_model:** this folder includes the timeseries plot and scatterplots comparing the neural network machine learning model's predicted power output and the actual power output.

    - **persistence_model:** this folder includes the timeseries plot and scatterplots comparing the persistence model's predicted power output and the actual power output.

    - **random_forest_model:** this folder includes the timeseries plot and scatterplots comparing the random forest machine learning model's predicted power output and the actual power output.

    - **scatterplots:** this folder includes scatterplots showing the correlation between explanatory variables and the wind power output.

    - **timeseries:** this folder includes a plot of the power output, and wind speeds for a specified limited time period of the historical data.

    - **windrose:** this folder includes two wind rose plots showing the windspeeds for different wind directions.

    - **error output:** this is a file that shows the calculated errors of the three different forecasting models and for different forecasting time horizons. 

- **src**
    - ***__init__.py:*** Empty file required for package initialization.
            
    - ***A_data_cleaning.py:*** the file establishes functions that can load the location data from the inputs folder, clean the data, and add lagged terms and other relevant explanatory variables.

    - ***B_data_visualisation.py:*** the file establishes functions that visualises the raw data. 

    - ***C_persistence_model.py:*** the file establishes functions that develop a persistence forecasting model to predict power output 1 hour and 6 hours into the future.

    - ***D_random_forest_model.py:*** the file establishes functions that develop random forest machine learning models to predict power output 1 hour and 6 hours into the future.

    - ***E_neural_network_model.py:*** the file establishes classes that develop neural network machine learning models to predict power output 1 hour and 6 hours into the future.

    - ***F_error_calcs.py:*** the file establishes functions that calculates the error measures between the predicted and actual power outputs.

    - ***G_results_visualisation.py:*** the file establishes functions that make plots of predicted vs. actual power outputs.

## Model Architecture Design

Underneath diagram shows the architecture of the model. 

The blue box represents the external raw data with historical meteorological observations and wind power generation data. The white boxes are data preparation steps. In these steps, the data from the blue box is loaded into the model, cleaned, updated with new variables, and split into four data sets (X_train, y_train, X_test, and y_test). The four data sets are used in the forecasting models to train and test the performance of the models. 

The coloured boxes (green and orange) are the actual forecasting models. The cleaned and split data from the white boxes go into these models to be trained and tested. The green boxes are forecasting models defined by functions. The persistence model is a simple and naive forecasting method, for which it is assumed that the current power production will be the power production in, e.g., one or six hours. The random forest model is a machine learning (ML) model using decision trees to find patterns in the data and forecast the power production. 

The orange box is a forecasting model defined by a class. In this class, I have defined a model for forecasting the power output using a neural network ML model that learns patterns from interconnected neurons. In the class, the class variables are defined, and a function is built to first drop irrelevant variables and second define and train the ML model. For the neural network model, both Keras and the MLPRegressor from scikit-learn can be used to train the model. The class is defined in the ***src/E_neural_network_model.py*** file.

The grey boxes indicate graph plotting functions. These make graph plots of either the input data or the predicted outputs from the forecasting models. The three boxes connected to one of the white boxes are functions plotting the model input data. The graphs produced by these functions have been used in the preliminary phases of the ML models’ development to get a sense of the relationships between power output and explanatory variables. Additionally, functions are plotting the predicted power output against the actual power output. These plots can give a visual understanding of the models’ performance. Lastly, the pink box indicates a function calculating the error measures of the forecasting models. These error measures are a powerful way to compare the performance of the different models.

![Image](https://github.com/user-attachments/assets/972795ce-a645-4704-8c11-36c3365a57f2)
