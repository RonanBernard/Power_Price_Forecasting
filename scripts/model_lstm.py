"""Attention-based model implementation for power price forecasting."""

import numpy as np
import datetime
import json
import os
import matplotlib.pyplot as plt
import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Dropout, BatchNormalization,
    LSTM, Concatenate, TimeDistributed, Reshape,
    RepeatVector
)
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow.keras.backend as K

# Define paths
from scripts.config import LOGS_PATH, MODELS_PATH

class LSTMModel:
    """Simple LSTM model for power price forecasting.
    
    The model architecture consists of:
    1. Past sequence processing:
        - LSTM layer with return_sequences
        - Optional BatchNormalization and Dropout
    2. Future sequence processing:
        - Dense projection to match LSTM dimension
        - Optional BatchNormalization and Dropout
    3. Combination layer:
        - Concatenate LSTM and future features
    4. Final processing:
        - Dense layer with ReLU activation
        - TimeDistributed Dense for output

    Parameters
    ----------
    lstm_units : int
        Number of units in LSTM layer
    dense_units : int
        Number of units in intermediate dense layer
    n_past_features : int
        Number of features in past input sequence
    n_future_features : int
        Number of features in future input sequence
    past_seq_len : int
        Length of past input sequence
    future_seq_len : int
        Length of future sequence to predict (default 24 for day-ahead)
    dropout : float, optional
        Dropout rate between [0, 1]
    batch_normalization : bool, optional
        Whether to use batch normalization
    learning_rate : float, optional
        Learning rate for optimizer
    verbose : bool, optional
        Whether to print training progress
    epochs_early_stopping : int, optional
        Number of epochs for early stopping
    scaler : object, optional
        Scaler for output denormalization
    loss : str, optional
        Loss function name
    metrics : list, optional
        List of metric names
    optimizer : str, optional
        Optimizer name
    regularization : str, optional
        Type of regularization ('l1', 'l2', or 'l1_l2')
    lambda_reg : float, optional
        Weight of the regularization
    """

    def __init__(
        self,
        lstm_units,
        dense_units,
        n_past_features,
        n_future_features,
        past_seq_len,
        future_seq_len=24,
        dropout=0,
        batch_normalization=False,
        learning_rate=None,
        verbose=False,
        epochs_early_stopping=40,
        scaler=None,
        loss='mse',
        metrics=['mae'],
        optimizer='adam',
        regularization=None,
        lambda_reg=0
    ):
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.n_past_features = n_past_features
        self.n_future_features = n_future_features
        self.past_seq_len = past_seq_len
        self.future_seq_len = future_seq_len
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.verbose = verbose
        self.epochs_early_stopping = epochs_early_stopping
        self.scaler = scaler
        self.loss = loss
        self.metrics = metrics
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        if self.dropout > 1 or self.dropout < 0:
            raise ValueError('Dropout parameter must be between 0 and 1')

        self.model = self._build_model()

        if learning_rate is None:
            opt = 'adam'
        else:
            if optimizer == 'adam':
                opt = kr.optimizers.Adam(
                    learning_rate=learning_rate, clipvalue=10000
                )
            elif optimizer == 'rmsprop':
                opt = kr.optimizers.RMSprop(
                    learning_rate=learning_rate, clipvalue=10000
                )
            elif optimizer == 'adagrad':
                opt = kr.optimizers.Adagrad(
                    learning_rate=learning_rate, clipvalue=10000
                )
            elif optimizer == 'adadelta':
                opt = kr.optimizers.Adadelta(
                    learning_rate=learning_rate, clipvalue=10000
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")

        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)

    def _reg(self, lambda_reg):
        """Build an l1 or l2 regularizer for the model.

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
        if self.regularization == 'l1_l2':
            return l1_l2(lambda_reg)
        return None

    def _dilated_causal_conv_block(self, x, dilation_rate):
        """Create a dilated causal convolution block with residual connection.

        Parameters
        ----------
        x : tensorflow.Tensor
            Input tensor
        dilation_rate : int
            Dilation rate for the convolution

        Returns
        -------
        tensorflow.Tensor
            Output tensor after convolution and residual connection
        """
        # Dilated convolution with causal padding
        conv = Conv1D(
            filters=self.cnn_filters,
            kernel_size=5,
            dilation_rate=dilation_rate,
            padding='causal',  # Use causal padding
            kernel_regularizer=self._reg(self.lambda_reg)
        )(x)
        
        # Adjust input channels if needed
        if x.shape[-1] != self.cnn_filters:
            x = Conv1D(
                filters=self.cnn_filters,
                kernel_size=1,
                padding='same',
                kernel_regularizer=self._reg(self.lambda_reg)
            )(x)
        
        # Residual connection
        out = Add()([x, conv])
        
        if self.batch_normalization:
            out = BatchNormalization()(out)
            
        if self.dropout > 0:
            out = Dropout(self.dropout)(out)
            
        return out

    def _build_model(self):
        """Define the structure of a simple LSTM model.

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network model using keras and tensorflow
        """
        # Input layers
        past_input = Input(shape=(self.past_seq_len, self.n_past_features))
        future_input = Input(shape=(self.future_seq_len, self.n_future_features))

        # Process past sequence with LSTM
        x = LSTM(
            self.lstm_units,
            return_sequences=False,  # Get final state
            kernel_regularizer=self._reg(self.lambda_reg)
        )(past_input)

        if self.batch_normalization:
            x = BatchNormalization()(x)

        if self.dropout > 0:
            x = Dropout(self.dropout)(x)

        # Repeat LSTM output for each future timestep
        x = RepeatVector(self.future_seq_len)(x)

        # Process future inputs
        future_dense = Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=self._reg(self.lambda_reg)
        )(future_input)

        if self.batch_normalization:
            future_dense = BatchNormalization()(future_dense)

        if self.dropout > 0:
            future_dense = Dropout(self.dropout)(future_dense)

        # Combine past and future information
        combined = Concatenate(axis=2)([x, future_dense])

        # Process combined information
        x = TimeDistributed(
            Dense(self.dense_units, activation='relu',
                 kernel_regularizer=self._reg(self.lambda_reg))
        )(combined)

        if self.batch_normalization:
            x = BatchNormalization()(x)

        if self.dropout > 0:
            x = Dropout(self.dropout)(x)

        # Output layer
        outputs = TimeDistributed(
            Dense(1, kernel_regularizer=self._reg(self.lambda_reg))
        )(x)
        outputs = Reshape((self.future_seq_len,))(outputs)

        return Model(inputs=[past_input, future_input], outputs=outputs)

    def fit(self, X_past_train, X_future_train, y_train, 
            X_past_val, X_future_val, y_val):
        """Train the attention model using single validation.
        
        Parameters
        ----------
        X_past_train : numpy.array
            Past sequence training input data
        X_future_train : numpy.array
            Future sequence training input data
        y_train : numpy.array
            Training target data
        X_past_val : numpy.array
            Past sequence validation input data
        X_future_val : numpy.array
            Future sequence validation input data
        y_val : numpy.array
            Validation target data

        Returns
        -------
        history : tensorflow.keras.callbacks.History
            Training history containing metrics for each epoch
        """
        # Create log directory for TensorBoard
        model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = LOGS_PATH / "LSTM" / "fit" / model_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters
        params = {k: str(v) if isinstance(v, (np.ndarray, list)) else v 
                 for k, v in self.__dict__.items() if k != 'model'}
        
        # Save parameters in logs directory for TensorBoard
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

        # Add learning rate scheduler
        lr_scheduler = kr.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,        # Reduce LR by half when plateauing
            patience=10,       # Wait 10 epochs before reducing LR
            min_lr=1e-6,      # Don't reduce LR below this value
            verbose=self.verbose
        )

        # Train the model
        history = self.model.fit(
            [X_past_train, X_future_train],
            y_train,
            epochs=1000,
            batch_size=32,
            validation_data=([X_past_val, X_future_val], y_val),
            callbacks=[early_stopping, tensorboard_callback, lr_scheduler],
            verbose=self.verbose
        )

        self.history = history

        self.plot_history(history)

        # Calculate and log final metrics
        val_results = self.model.evaluate(
            [X_past_val, X_future_val],
            y_val
        )
        val_loss = val_results[0]
        val_metrics = val_results[1:]

        train_results = self.model.evaluate(
            [X_past_train, X_future_train],
            y_train
        )
        train_loss = train_results[0]
        train_metrics = train_results[1:]

        if self.verbose:
            print("\nFinal Training Metrics:")
            if self.loss == 'mse':
                print(f"RMSE: {np.sqrt(train_loss):.4f}")
            else:
                print(f"{self.loss}: {train_loss:.4f}")
            print(f"{self.metrics}: {train_metrics}")
            print("\nFinal Validation Metrics:")
            if self.loss == 'mse':
                print(f"RMSE: {np.sqrt(val_loss):.4f}")
            else:
                print(f"{self.loss}: {val_loss:.4f}")
            print(f"{self.metrics}: {val_metrics}")

        # Save final results
        final_results = {
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }

        param_results = {
            'parameters': params,
            'model_name': model_name,
            'final_results': final_results
        }

        # Save parameters and final results alongside model file
        model_dir = os.path.join(MODELS_PATH, "LSTM")
        os.makedirs(model_dir, exist_ok=True)
        param_results_path = os.path.join(
            model_dir, f"{model_name}_param_results.json"
        )
        with open(param_results_path, "w") as f:
            json.dump(param_results, f, indent=4)

        # Save the model
        model_path = os.path.join(model_dir, f"{model_name}.keras")
        self.model.save(model_path)

        return history
    
    def plot_history(self, history):
        """Plot training history showing loss and metrics.
        
        Parameters
        ----------
        history : tensorflow.keras.callbacks.History
            Training history containing metrics for each epoch
        """
        n_metrics = len(self.metrics) + 1  # +1 for loss
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
            
        # Plot loss
        if self.loss == 'mse':
            axes[0].plot(
                np.sqrt(history.history['loss']), 
                label='Train RMSE', 
                linestyle='--'
            )
            axes[0].plot(
                np.sqrt(history.history['val_loss']), 
                label='Val RMSE', 
                linestyle='--'
            )
            axes[0].set_title('Loss RMSE')
        else:
            axes[0].plot(history.history['loss'], label='Train loss')
            axes[0].plot(history.history['val_loss'], label='Val loss')
            axes[0].set_title(f'Loss ({self.loss})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot additional metrics
        for i, metric in enumerate(self.metrics, 1):
            metric_name = metric if isinstance(metric, str) else metric.__name__
            axes[i].plot(
                history.history[metric_name], 
                label=f'Train {metric_name}'
            )
            axes[i].plot(
                history.history[f'val_{metric_name}'], 
                label=f'Val {metric_name}'
            )
            axes[i].set_title(f'Metric: {metric_name}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric_name)
            axes[i].legend()
            axes[i].grid(True)
            
        plt.tight_layout()
        plt.show()

    def predict(self, X_past, X_future):
        """Make predictions with the trained model.

        Parameters
        ----------
        X_past : numpy.array
            Past sequence input data
        X_future : numpy.array
            Future sequence input data

        Returns
        -------
        numpy.array
            Model predictions
        """
        return self.model.predict([X_past, X_future], verbose=0)

    def plot_hourly_averages(self, y_true, y_pred):
        """Plot average values for each hour comparing predictions and actual values.

        Parameters
        ----------
        y_true : numpy.array
            True target values with shape (n_samples, future_seq_len)
        y_pred : numpy.array
            Predicted values with shape (n_samples, future_seq_len)
        """
        # Ensure inputs have correct shape
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        if y_true.shape[1] != self.future_seq_len:
            raise ValueError(
                f"Expected sequence length {self.future_seq_len}, "
                f"got {y_true.shape[1]}"
            )

        # Calculate hourly averages
        hours = range(self.future_seq_len)
        avg_true = np.mean(y_true, axis=0)
        avg_pred = np.mean(y_pred, axis=0)

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(hours, avg_true, 'b-', label='Actual', marker='o')
        plt.plot(hours, avg_pred, 'r--', label='Predicted', marker='o')

        # Add gap visualization
        plt.fill_between(
            hours, avg_true, avg_pred, alpha=0.2, color='gray', label='Gap'
        )

        # Customize the plot
        plt.title('Average Hourly Values: Predicted vs Actual')
        plt.xlabel('Hour')
        plt.ylabel('Average Value')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add hour markers
        plt.xticks(hours)

        # Calculate and display metrics
        mae_hourly = np.mean(np.abs(avg_true - avg_pred))
        rmse_hourly = np.sqrt(np.mean((avg_true - avg_pred)**2))
        plt.text(
            0.02, 0.98,
            f'Hourly MAE: {mae_hourly:.4f}\nHourly RMSE: {rmse_hourly:.4f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        plt.tight_layout()
        plt.show()

    @classmethod
    def from_saved_model(cls, model_name):
        """Create a LSTMModel instance from a saved model.

        This class method loads both the model weights and its parameters
        from saved files.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g. "20240315-123456"). The method will look
            for both .keras and parameters.json files in MODELS_PATH/LSTM.

        Returns
        -------
        LSTMModel
            A new instance initialized with the correct parameters
        """
        # Construct paths
        model_dir = os.path.join(MODELS_PATH, "LSTM")
        model_path = os.path.join(model_dir, f"{model_name}.keras")
        
        # Try to find parameters file (check both locations)
        params_path = os.path.join(model_dir, f"{model_name}_parameters.json")
        if not os.path.exists(params_path):
            # Try the logs directory
            params_path = os.path.join(
                LOGS_PATH, "LSTM", "fit", model_name, "parameters.json"
            )

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(params_path):
            param_results_path = os.path.join(
                model_dir, f'{model_name}_param_results.json'
            )
            logs_path = os.path.join(
                LOGS_PATH, 'LSTM', 'fit', model_name, 'parameters.json'
            )
            raise FileNotFoundError(
                "Parameters file not found in either:\n"
                f"1. {param_results_path}\n"
                f"2. {logs_path}"
            )

        # Load parameters
        with open(params_path, 'r') as f:
            params = json.load(f)['parameters']

        # Create instance with loaded parameters
        instance = cls(
            lstm_units=params['lstm_units'],
            dense_units=params['dense_units'],
            n_past_features=params['n_past_features'],
            n_future_features=params['n_future_features'],
            past_seq_len=params['past_seq_len'],
            future_seq_len=params['future_seq_len'],
            dropout=params.get('dropout', 0),
            batch_normalization=params.get('batch_normalization', False),
            regularization=params.get('regularization', None),
            lambda_reg=params.get('lambda_reg', 0)
        )

        # Load the model weights
        instance.model = kr.models.load_model(model_path)
        return instance

    def load_model(self, model_path):
        """Load a previously saved model.

        It's recommended to use from_saved_model() class method instead,
        which will properly initialize all parameters.

        Parameters
        ----------
        model_path : str or Path
            Path to the saved model file. Can be either absolute path or
            relative to the MODELS_PATH/LSTM directory.

        Returns
        -------
        self : LSTMModel
            The instance with loaded model for method chaining
        """
        # Handle relative paths
        if not os.path.isabs(model_path):
            model_path = os.path.join(MODELS_PATH, "LSTM", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the model
        self.model = kr.models.load_model(model_path)

        # Update model parameters from the loaded model
        self.future_seq_len = self.model.output_shape[1]
        input_shapes = [layer.input_shape for layer in self.model.inputs]
        self.past_seq_len = input_shapes[0][1]
        self.n_past_features = input_shapes[0][2]
        self.n_future_features = input_shapes[1][2]

        return self

    def explain_predictions(
        self,
        X_past,
        X_future,
        sample_size=100,
        target_hour=12,  # Default to explaining noon predictions
        feature_names=None
    ):
        """Calculate and visualize SHAP values for model predictions.
        
        Parameters
        ----------
        X_past : numpy.array
            Past sequence input data
        X_future : numpy.array
            Future sequence input data
        sample_size : int, optional
            Number of samples to use for SHAP calculation (default: 100)
        target_hour : int, optional
            Which hour of the 24-hour prediction to explain (default: 12)
        feature_names : dict, optional
            Dictionary with 'past' and 'future' lists of feature names
            
        Returns
        -------
        tuple
            (past_shap_values, future_shap_values, past_expected_value, future_expected_value)
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap package is required for model explanations.\n"
                "Install it with: pip install shap"
            )

        if not 0 <= target_hour < self.future_seq_len:
            raise ValueError(
                f"target_hour must be between 0 and {self.future_seq_len-1}"
            )

        # Use a subset of data if sample_size is smaller than dataset
        if sample_size and sample_size < len(X_past):
            indices = np.random.choice(
                len(X_past), sample_size, replace=False
            )
            X_past = X_past[indices]
            X_future = X_future[indices]

        # Create feature names if not provided
        if feature_names is None:
            feature_names = {
                'past': [f'past_feature_{i}' 
                        for i in range(self.n_past_features)],
                'future': [f'future_feature_{i}' 
                          for i in range(self.n_future_features)]
            }

        # Create wrapper functions to explain past and future features separately
        def f_past(X):
            # Create dummy future input (mean of training data)
            X_future_dummy = np.tile(
                X_future.mean(axis=0),
                (len(X), 1, 1)
            )
            return self.model.predict(
                [X, X_future_dummy]
            )[:, target_hour]

        def f_future(X):
            # Create dummy past input (mean of training data)
            X_past_dummy = np.tile(
                X_past.mean(axis=0),
                (len(X), 1, 1)
            )
            return self.model.predict(
                [X_past_dummy, X]
            )[:, target_hour]

        # Calculate SHAP values for past features
        past_explainer = shap.KernelExplainer(
            f_past,
            X_past[:sample_size]
        )
        past_shap_values = past_explainer.shap_values(
            X_past[:sample_size],
            nsamples=50
        )

        # Calculate SHAP values for future features
        future_explainer = shap.KernelExplainer(
            f_future,
            X_future[:sample_size]
        )
        future_shap_values = future_explainer.shap_values(
            X_future[:sample_size],
            nsamples=50
        )

        # Create summary plots
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        shap.summary_plot(
            past_shap_values,
            X_past[:sample_size],
            feature_names=feature_names['past'],
            show=False,
            plot_size=(6, 8)
        )
        plt.title(f'SHAP Values - Past Features (Hour {target_hour})')
        
        plt.subplot(1, 2, 2)
        shap.summary_plot(
            future_shap_values,
            X_future[:sample_size],
            feature_names=feature_names['future'],
            show=False,
            plot_size=(6, 8)
        )
        plt.title(f'SHAP Values - Future Features (Hour {target_hour})')
        
        plt.tight_layout()
        plt.show()

        return (
            past_shap_values,
            future_shap_values,
            past_explainer.expected_value,
            future_explainer.expected_value
        )

    def clear_session(self):
        """Clear the tensorflow session.

        Used during recalibration to avoid RAM memory leakages. If the model is
        retrained continuously, tensorflow slightly increases RAM usage at each
        step.
        """
        K.clear_session()
