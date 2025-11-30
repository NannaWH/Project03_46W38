################################################################################## Creating a Machine Learning Forecasting Model using Neural Networks #########
################################################################################

#Importing packages
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import keras
from keras import layers

def neural_network_model_1hour(X_train, y_train, X_test, model_regressor):
    """Creating a neural prediction model for 1 hour ahead prediction that can be calculated both using MLPregressor (sickit-learn) and Keras
    
    """

    #We drop columns that are irrelevant to our prediction
    X_train = X_train.drop(['Time','date', 'time', 'year', 'day', 'relativehumidity_2m','windgusts_10m','winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)
    X_test = X_test.drop(['Time','date', 'time', 'year', 'day', 'relativehumidity_2m', 'windgusts_10m', 'winddirection_10m', 'winddirection_100m','wdir_radians_10m', 'wdir_radians_100m'], axis=1)

    ### Sickit-learn (MLPregressor) Neural network model 
    if model_regressor == "MLPRegressor":
        # We make a Neural Network Prediction Model
        model = MLPRegressor(hidden_layer_sizes=(5, 5),
                        activation='relu',
                        solver='adam',
                        max_iter=100,
                        random_state=14) #14

        # We make a prediction of y
        model.fit(X_train, y_train)

        # We predict y 
        y_pred = model.predict(X_test)



    ### Keras Neural network model 
    if model_regressor == "Keras":

        # X_train has shape (number of samples, number of input variables)
        n_features = X_train.shape[1]

        #Set global random seed number
        keras.utils.set_random_seed(14)

        model = keras.Sequential(
            [
                layers.Dense(5, activation="relu", input_shape=(n_features,)),
                layers.Dense(5, activation="relu"),
                layers.Dense(1, activation="relu")
            ]
        )

        # Call model on a test input
        model.compile(optimizer='adam', 
                    loss='mse', 
                    metrics=['mae'])

        model.fit(X_train, y_train, 
                epochs=100,        # similar to max_iter=100
                batch_size=32,
                verbose=1)
        
        # We predict y 
        y_pred = model.predict(X_test)

        return y_pred



