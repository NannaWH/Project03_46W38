# Project03_46W38
This is the final project in the course 46W38 Scientific Programming in Wind Energy, Fall 2025 at DTU. 

This project develops a short-term wind power forecasting model. In the model, I predict future power output using current and previous power and weather data. I use three different machine learning models namely:
    1) Persistence model
    2) Random forest model
    3) Neural network model

    The model is divided into the following modules and packages:
        examples
            main.py: this file executes the final machine learning models as well as calculating errors and saving plots used predefined functions. 
        
        inputs
            cleaned_data.csv: this file includes the rawdata cleaned by adding more explanatory variables

            Location1.csv-Location.csv: historical power and weather data for location 1-4

        outputs
            neural_network_model: this folder includes the timeseries plot and scatterplots comparing the neural network machine learning model's predicted power output and the actual power output.

            persistence_model: this folder includes the timeseries plot and scatterplots comparing the persistence model's predicted power output and the actual power output.

            random_forest_model: this folder includes the timeseries plot and scatterplots comparing the random forest machine learning model's predicted power output and the actual power output.

            scatterplots: this folder includes scatterplots showing the correlation between explanatory variables and the wind power output.

            timeseries: this folder includes a plot of the power output, and wind speeds for a specified limited timeperiod of the historical data.

            windrose: this folder includes two wind rose plots showing the power output for different wind directions.

            error output: this is a file that shows the calculated errors of the different trained machine learning models. 

        src
            __init__.py: Empty file required for package initialization
            
            A_data_cleaning.py: the file establishes functions that can load the location data from the inputs folder, clean the data and add lagged terms and other relevant explanatory variables.

            B_data_visualisation.py: the file establishes functions that can visualises the raw data. 

            C_persistence_model.py: the file establishes a function that develops a persistence predicting model to predict power output 1 hour and 6 hours into the future.

            D_random_forest_model.py: the file establishes function that develops random forest machine learnings models to predict power output 1 hour and 6 hours into the future.

            E_neural_network_model.py: the file establishes function that develops neural network machine learnings models to predict power output 1 hour and 6 hours into the future.

            F_error_calcs.py: this file establishes functions that calculates the error measures between the predicted and actual power outputs.

            G_results_visualisation.py: this file establishes functions that make plots of predicted vs. actual power outputs.


- A description of the module/package architecture, with at least one diagram.
- A description of the class(es) you have implemented in your package, with clear reference to the file name of the code inside src.

