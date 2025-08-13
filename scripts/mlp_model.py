import numpy as np
import pandas as pd
import time
import pickle as pc
import os

import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU, PReLU
import tensorflow.keras.backend as K

from sklearn.metrics import mean_squared_error, mean_absolute_error



class MLPModel(object):

    """Basic MLP model based on keras and tensorflow. 
    
    The model can be used standalone to train and predict a MLP using its fit/predict methods.
    However, it is intended to be used within the :class:`hyperparameter_optimizer` method
    and the :class:`DNN` class. The former obtains a set of best hyperparameter using the :class:`DNNModel` class. 
    The latter employes the set of best hyperparameters to recalibrate a :class:`DNNModel` object
    and make predictions.
    
    Parameters
    ----------
    neurons : list
        List containing the number of neurons in each hidden layer. E.g. if ``len(neurons)`` is 2,
        the DNN model has an input layer of size ``n_features``, two hidden layers, and an output 
        layer of size ``outputShape``.
    n_features : int
        Number of input features in the model. This number defines the size of the input layer.
    outputShape : int, optional
        Default number of output neurons. It is 24 as it is the default in most day-ahead markets.
    dropout : float, optional
        Number between [0, 1] that selects the percentage of dropout. A value of 0 indicates
        no dropout.
    batch_normalization : bool, optional
        Boolean that selects whether batch normalization is considered.
    lr : float, optional
        Learning rate for optimizer algorithm. If none provided, the default one is employed
        (see the `keras documentation <https://keras.io/>`_ for the default learning rates of each algorithm).
    verbose : bool, optional
        Boolean that controls the logs. If set to true, a minimum amount of information is 
        displayed.
    epochs_early_stopping : int, optional
        Number of epochs used in early stopping to stop training. When no improvement is observed
        in the validation dataset after ``epochs_early_stopping`` epochs, the training stops.
    scaler : :class:`epftoolbox.data.DataScaler`, optional
        Scaler object to invert-scale the output of the neural network if the neural network
        is trained with scaled outputs.
    loss : str, optional
        Loss to be used when training the neural network. Any of the regression losses defined in 
        keras can be used.
    optimizer : str, optional
        Name of the optimizer when training the DNN. See the `keras documentation <https://keras.io/>`_ 
        for a list of optimizers.
    activation : str, optional
        Name of the activation function in the hidden layers. See the `keras documentation <https://keras.io/>`_ for a list
        of activation function.
    initializer : str, optional
        Name of the initializer function for the weights of the neural network. See the 
        `keras documentation <https://keras.io/>`_ for a list of initializer functions.
    regularization : None, optional
        Name of the regularization technique. It can can have three values ``'l2'`` for l2-norm
        regularization, ``'l1'`` for l1-norm regularization, or ``None`` for no regularization .
    lambda_reg : int, optional
        The weight for regulization if ``regularization`` is ``'l2'`` or ``'l1'``.
    """
    


    def __init__(self, neurons, n_features, outputShape=24, dropout=0, batch_normalization=False, lr=None,
                 verbose=False, epochs_early_stopping=40, scaler=None, loss='mse',
                 optimizer='adam', activation='relu', initializer='glorot_uniform',
                 regularization=None, lambda_reg=0):
        self.neurons = neurons
        self.dropout = dropout

        if self.dropout > 1 or self.dropout < 0:
            raise ValueError('Dropout parameter must be between 0 and 1')

        self.batch_normalization = batch_normalization
        self.verbose = verbose
        self.epochs_early_stopping = epochs_early_stopping
        self.n_features = n_features
        self.scaler = scaler
        self.outputShape = outputShape
        self.activation = activation
        self.initializer = initializer
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.model = self._build_model()

        if lr is None:
            opt = 'adam'
        else:
            if optimizer == 'adam':
                opt = kr.optimizers.Adam(lr=lr, clipvalue=10000)
            if optimizer == 'RMSprop':
                opt = kr.optimizers.RMSprop(lr=lr, clipvalue=10000)
            if optimizer == 'adagrad':
                opt = kr.optimizers.Adagrad(lr=lr, clipvalue=10000)
            if optimizer == 'adadelta':
                opt = kr.optimizers.Adadelta(lr=lr, clipvalue=10000)

        self.model.compile(loss=loss, optimizer=opt)


     def _reg(self, lambda_reg):
        """Internal method to build an l1 or l2 regularizer for the DNN
        
        Parameters
        ----------
        lambda_reg : float
            Weight of the regularization
        
        Returns
        -------
        tensorflow.keras.regularizers.L1L2
            The regularizer object
        """
        if self.regularization == 'l2':
            return l2(lambda_reg)
        if self.regularization == 'l1':
            return l1(lambda_reg)
        else:
            return None
        

    def _build_model(self):
        """Internal method that defines the structure of the MLP
        
        Returns
        -------
        tensorflow.keras.models.Model
            A neural network model using keras and tensorflow
        """
        inputShape = (None, self.n_features)

        past_data = Input(batch_shape=inputShape)

        past_Dense = past_data
        if self.activation == 'selu':
            self.initializer = 'lecun_normal'

        for k, neurons in enumerate(self.neurons):

            if self.activation == 'LeakyReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = LeakyReLU(alpha=.001)(past_Dense)

            elif self.activation == 'PReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = PReLU()(past_Dense)

            else:
                past_Dense = Dense(neurons, activation=self.activation,
                                   batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)

            if self.batch_normalization:
                past_Dense = BatchNormalization()(past_Dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    past_Dense = AlphaDropout(self.dropout)(past_Dense)
                else:
                    past_Dense = Dropout(self.dropout)(past_Dense)

        output_layer = Dense(self.outputShape, kernel_initializer=self.initializer,
                             kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
        model = Model(inputs=[past_data], outputs=[output_layer])

        return model
    
    def _obtain_metrics(self, X, Y):
        """Internal method to update the metrics used to train the network
        
        Parameters
        ----------
        X : numpy.array
            Input array for evaluating the model
        Y : numpy.array
            Output array for evaluating the model
        
        Returns
        -------
        list
            A list containing the metric based on the loss of the neural network and a second metric
            representing the RMSE and MAE of the MLP
        """
        error = self.model.evaluate(X, Y, verbose=0)
        Ybar = self.model.predict(X, verbose=0)

        if self.scaler is not None:
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
                Ybar = Ybar.reshape(-1, 1)

            Y = self.scaler.inverse_transform(Y)
            Ybar = self.scaler.inverse_transform(Ybar)

        rmse = np.sqrt(mean_squared_error(Y, Ybar))
        mae = mean_absolute_error(Y, Ybar)

        return error, rmse, mae
    
    def _display_info_training(self, bestError, bestRMSE, bestMAE, countNoImprovement):
        """Internal method that displays useful information during training
        
        Parameters
        ----------
        bestError : float
            Loss of the neural network in the validation dataset
        bestRMSE : float
            RMSE of the neural network in the validation dataset
        bestMAE : float
            MAE of the neural network in the validation dataset
        countNoImprovement : int
            Number of epochs in the validation dataset without improvements
        """
        print(" Best error:\t\t{:.1e}".format(bestError))
        print(" Best RMSE:\t\t{:.2f}".format(bestRMSE))                
        print(" Best MAE:\t\t{:.2f}".format(bestMAE))                
        print(" Epochs without improvement:\t{}\n".format(countNoImprovement))


    def fit(self, trainX, trainY, valX, valY):
        """Method to estimate the DNN model.
        
        Parameters
        ----------
        trainX : numpy.array
            Inputs fo the training dataset.
        trainY : numpy.array
            Outputs fo the training dataset.
        valX : numpy.array
            Inputs fo the validation dataset used for early-stopping.
        valY : numpy.array
            Outputs fo the validation dataset used for early-stopping.
        """

        # Variables to control training improvement
        bestError = 1e20
        bestMAE = 1e20

        countNoImprovement = 0

        bestWeights = self.model.get_weights()

        for epoch in range(1000):
            start_time = time.time()

            self.model.fit(trainX, trainY, batch_size=32,
                           epochs=1, verbose=False, shuffle=True)

            # Updating epoch metrics and displaying useful information
            if self.verbose:
                print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, 1000,
                                                             time.time() - start_time))

            # Calculating relevant metrics to perform early-stopping
            valError, valMAE = self._obtain_metrics(valX, valY)

            # Early-stopping
            # Checking if current validation metrics are better than best so far metrics.
            # If the network does not improve, we stop.
            # If it improves, the optimal weights are saved
            if valError < bestError:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()

                bestError = valError
                bestMAE = valMAE
                if valMAE < bestMAE:
                    bestMAE = valMAE

            elif valMAE < bestMAE:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()
                bestMAE = valMAE
            else:
                countNoImprovement += 1

            if countNoImprovement >= self.epochs_early_stopping:
                if self.verbose:
                    self._display_info_training(bestError, bestMAE, countNoImprovement)
                break

            # Displaying final information
            if self.verbose:
                self._display_info_training(bestError, bestMAE, countNoImprovement)

        # After early-stopping, the best weights are set in the model
        self.model.set_weights(bestWeights)

    
    def predict(self, X):
        """Method to make a prediction after the DNN is trained.
        
        Parameters
        ----------
        X : numpy.array
            Input to the DNN. It has to be of size *[n, n_features]* where *n* can be any 
            integer, and *n_features* is the attribute of the DNN representing the number of
            input features.
        
        Returns
        -------
        numpy.array
            Output of the DNN after making the prediction.
        """

        Ybar = self.model.predict(X, verbose=0)
        return Ybar

    def clear_session(self):
        """Method to clear the tensorflow session. 

        It is used in the :class:`DNN` class during recalibration to avoid RAM memory leakages.
        In particular, if the DNN is retrained continuosly, at each step tensorflow slightly increases 
        the total RAM usage.

        """

        K.clear_session()