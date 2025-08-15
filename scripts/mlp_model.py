"""MLP model implementation for power price forecasting."""

import numpy as np
import datetime
import json
import os
import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Dropout, AlphaDropout,
    BatchNormalization, LeakyReLU, PReLU
)
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define paths
from scripts.config import LOGS_PATH, MODEL_PATH

class MLPModel:
    """Basic MLP model based on keras and tensorflow.

    The model can be used standalone to train and predict a MLP using its
    fit/predict methods. However, it is intended to be used within the
    hyperparameter_optimizer method and the DNN class. The former obtains a set
    of best hyperparameter using the DNNModel class. The latter employes the set
    of best hyperparameters to recalibrate a DNNModel object and make predictions.

    Parameters
    ----------
    neurons : list
        List containing the number of neurons in each hidden layer. E.g. if
        len(neurons) is 2, the DNN model has an input layer of size n_features,
        two hidden layers, and an output layer of size outputShape.
    n_features : int
        Number of input features in the model. This number defines the size of
        the input layer.
    outputShape : int, optional
        Default number of output neurons. It is 24 as it is the default in most
        day-ahead markets.
    dropout : float, optional
        Number between [0, 1] that selects the percentage of dropout. A value of
        0 indicates no dropout.
    batch_normalization : bool, optional
        Boolean that selects whether batch normalization is considered.
    lr : float, optional
        Learning rate for optimizer algorithm. If none provided, the default one
        is employed (see keras documentation for default learning rates).
    verbose : bool, optional
        Boolean that controls the logs. If set to true, a minimum amount of
        information is displayed.
    epochs_early_stopping : int, optional
        Number of epochs used in early stopping to stop training. When no
        improvement is observed in validation dataset after epochs_early_stopping
        epochs, training stops.
    scaler : object, optional
        Scaler object to invert-scale the output of the neural network if the
        neural network is trained with scaled outputs.
    loss : str, optional
        Loss to be used when training the neural network. Any of the regression
        losses defined in keras can be used.
    optimizer : str, optional
        Name of the optimizer when training the DNN. See keras documentation for
        list of optimizers.
    activation : str, optional
        Name of the activation function in hidden layers. See keras documentation
        for list of activation functions.
    initializer : str, optional
        Name of the initializer function for weights. See keras documentation for
        list of initializer functions.
    regularization : None, optional
        Name of regularization technique. Can be 'l2' for l2-norm regularization,
        'l1' for l1-norm regularization, or None for no regularization.
    lambda_reg : int, optional
        Weight for regulization if regularization is 'l2' or 'l1'.
    """

    def __init__(
        self,
        neurons,
        n_features,
        outputShape=24,
        dropout=0,
        batch_normalization=False,
        lr=None,
        verbose=False,
        epochs_early_stopping=40,
        scaler=None,
        loss='mse',
        optimizer='adam',
        activation='relu',
        initializer='glorot_uniform',
        regularization=None,
        lambda_reg=0
    ):
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
                opt = kr.optimizers.Adam(learning_rate=lr, clipvalue=10000)
            elif optimizer == 'rmsprop':
                opt = kr.optimizers.RMSprop(learning_rate=lr, clipvalue=10000)
            elif optimizer == 'adagrad':
                opt = kr.optimizers.Adagrad(learning_rate=lr, clipvalue=10000)
            elif optimizer == 'adadelta':
                opt = kr.optimizers.Adadelta(learning_rate=lr, clipvalue=10000)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")

        self.model.compile(loss=loss, optimizer=opt)

    def _reg(self, lambda_reg):
        """Build an l1 or l2 regularizer for the DNN.

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
        return None

    def _build_model(self):
        """Define the structure of the MLP.

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

        for neurons in self.neurons:
            if self.activation == 'LeakyReLU':
                past_Dense = Dense(
                    neurons,
                    activation='linear',
                    kernel_initializer=self.initializer,
                    kernel_regularizer=self._reg(self.lambda_reg)
                )(past_Dense)
                past_Dense = LeakyReLU(alpha=.001)(past_Dense)

            elif self.activation == 'PReLU':
                past_Dense = Dense(
                    neurons,
                    activation='linear',
                    kernel_initializer=self.initializer,
                    kernel_regularizer=self._reg(self.lambda_reg)
                )(past_Dense)
                past_Dense = PReLU()(past_Dense)

            else:
                past_Dense = Dense(
                    neurons,
                    activation=self.activation,
                    kernel_initializer=self.initializer,
                    kernel_regularizer=self._reg(self.lambda_reg)
                )(past_Dense)

            if self.batch_normalization:
                past_Dense = BatchNormalization()(past_Dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    past_Dense = AlphaDropout(self.dropout)(past_Dense)
                else:
                    past_Dense = Dropout(self.dropout)(past_Dense)

        output_layer = Dense(
            self.outputShape,
            kernel_initializer=self.initializer,
            kernel_regularizer=self._reg(self.lambda_reg)
        )(past_Dense)

        return Model(inputs=[past_data], outputs=[output_layer])

    def _obtain_metrics(self, X, Y):
        """Calculate training metrics.

        Parameters
        ----------
        X : numpy.array
            Input array for evaluating the model
        Y : numpy.array
            Output array for evaluating the model

        Returns
        -------
        tuple
            (error, rmse, mae) metrics for the model
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

    def fit(self, X_train, y_train, X_val, y_val):
        """Train the MLP model using single validation.
        
        Parameters
        ----------
        X_train : numpy.array
            Training input data
        y_train : numpy.array
            Training target data
        X_val : numpy.array
            Validation input data
        y_val : numpy.array
            Validation target data

        Returns
        -------
        history : tensorflow.keras.callbacks.History
            Training history containing metrics for each epoch
        """
        # Create log directory for TensorBoard
        model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = LOGS_PATH / "V1" / "fit" / model_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters
        params = {k: str(v) if isinstance(v, (np.ndarray, list)) else v 
                 for k, v in self.__dict__.items() if k != 'model'}
        with open(log_dir / "parameters.json", "w") as f:
            json.dump(params, f, indent=4)

        # Setup callbacks
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.epochs_early_stopping,
            restore_best_weights=True,
            verbose=self.verbose
        )

        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=1000,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, tensorboard_callback],
            verbose=self.verbose
        )

        # Calculate and log final metrics
        train_error, train_rmse, train_mae = self._obtain_metrics(X_train, y_train)
        val_error, val_rmse, val_mae = self._obtain_metrics(X_val, y_val)

        if self.verbose:
            print("\nFinal Training Metrics:")
            print(f"Error: {train_error:.4f}")
            print(f"RMSE: {train_rmse:.4f}")
            print(f"MAE: {train_mae:.4f}")
            print("\nFinal Validation Metrics:")
            print(f"Error: {val_error:.4f}")
            print(f"RMSE: {val_rmse:.4f}")
            print(f"MAE: {val_mae:.4f}")

        # Save the model
        self.model.save(os.path.join(MODEL_PATH, "V1", f"{model_name}.keras"))

        return history

    def predict(self, X):
        """Make predictions with the trained model.

        Parameters
        ----------
        X : numpy.array
            Input to the DNN. Must be size [n, n_features] where n can be any
            integer, and n_features matches the model's input features.

        Returns
        -------
        numpy.array
            Model predictions
        """
        return self.model.predict(X, verbose=0)

    def clear_session(self):
        """Clear the tensorflow session.

        Used during recalibration to avoid RAM memory leakages. If the DNN is
        retrained continuously, tensorflow slightly increases RAM usage at each
        step.
        """
        K.clear_session()