"""Attention-based model implementation for power price forecasting."""

import numpy as np
import datetime
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Dropout, BatchNormalization, Conv1D, Add,
    Bidirectional, LSTM, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate, TimeDistributed, Reshape,
    RepeatVector
)
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow.keras.backend as K

# Define paths
from scripts.config import LOGS_PATH, MODELS_PATH

class AttentionModel:
    """Advanced attention-based model combining CNN, BiLSTM, and attention mechanisms.
    
    The model architecture consists of:
    1. Encoder:
        - Causal dilated Conv1D layers with residuals
        - Bidirectional LSTM
        - Multi-Head Self-Attention
    2. Context:
        - GlobalAveragePooling + Dense
    3. Decoder:
        - Cross-attention between encoder outputs and future inputs
        - TimeDistributed Dense for multi-horizon prediction

    Parameters
    ----------
    cnn_filters : int
        Number of filters in CNN layers
    lstm_units : int
        Number of units in LSTM layers
    attention_heads : int
        Number of attention heads
    attention_key_dim : int
        Dimension of attention keys
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
    """

    def __init__(
        self,
        preprocess_version,
        cnn_filters,
        lstm_units,
        attention_heads,
        attention_key_dim,
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
        self.preprocess_version = preprocess_version
        self.model_version = "v1"
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.attention_key_dim = attention_key_dim
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
                opt = kr.optimizers.Adam(learning_rate=learning_rate, clipvalue=10000)
            elif optimizer == 'rmsprop':
                opt = kr.optimizers.RMSprop(learning_rate=learning_rate, clipvalue=10000)
            elif optimizer == 'adagrad':
                opt = kr.optimizers.Adagrad(learning_rate=learning_rate, clipvalue=10000)
            elif optimizer == 'adadelta':
                opt = kr.optimizers.Adadelta(learning_rate=learning_rate, clipvalue=10000)
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
        """Define the structure of the attention-based model.

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network model using keras and tensorflow
        """
        # Input layers
        past_input = Input(shape=(self.past_seq_len, self.n_past_features))
        future_input = Input(shape=(self.future_seq_len, self.n_future_features))

        # Encoder
        # 1. Dilated Causal CNN
        x = past_input
        dilations = [1, 2, 4, 8]
        for d in dilations:
            x = self._dilated_causal_conv_block(x, d)

        # 2. Bidirectional LSTM
        x = Bidirectional(
            LSTM(
                self.lstm_units,
                return_sequences=True,
                kernel_regularizer=self._reg(self.lambda_reg)
            )
        )(x)

        if self.batch_normalization:
            x = BatchNormalization()(x)

        # 3. Self-Attention
        x = LayerNormalization()(x)
        self_attention = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.attention_key_dim
        )(x, x)
        encoder_output = Add()([x, self_attention])
        encoder_output = LayerNormalization()(encoder_output)

        # Context vector
        context = GlobalAveragePooling1D()(encoder_output)
        context = Dense(
            self.lstm_units * 2,
            kernel_regularizer=self._reg(self.lambda_reg)
        )(context)

        # Process context using RepeatVector instead of custom layer
        context_repeated = RepeatVector(self.future_seq_len)(context)

        # Combine context with future inputs
        decoder_input = Concatenate()([context_repeated, future_input])

        # Cross-attention between decoder input and encoder output
        cross_attention = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.attention_key_dim
        )(decoder_input, encoder_output)
        
        decoder_output = Add()([decoder_input, cross_attention])
        decoder_output = LayerNormalization()(decoder_output)

        if self.dropout > 0:
            decoder_output = Dropout(self.dropout)(decoder_output)

        # Final prediction
        outputs = TimeDistributed(
            Dense(1, kernel_regularizer=self._reg(self.lambda_reg))
        )(decoder_output)
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
        model_name = "ATT_v1_preproc_" + self.preprocess_version + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = MODELS_PATH / "logs" / "fit" / model_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters
        params = {k: str(v) if isinstance(v, (np.ndarray, list)) else v 
                 for k, v in self.__dict__.items() if k != 'model'}

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

        history_results = {
                'history_loss': history.history['loss'],
                'history_val_loss': history.history['val_loss'],
            }

        for metric in self.metrics:
            history_results[f'history_{metric}'] = history.history[metric]
            history_results[f'history_val_{metric}'] = history.history[f'val_{metric}']

        param_results = {
            'parameters': params,
            'rolling_horizon': False,
            'model_name': model_name,
            'final_results': final_results,
            'history_results': history_results
            }

        # Save parameters and final results alongside model file
        model_dir = MODELS_PATH
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, f"{model_name}_param_results.json"), "w") as f:
            json.dump(param_results, f, indent=4)

        # Save the model
        self.model.save(os.path.join(model_dir, f"{model_name}.keras"))

        return history
    
    def fit_rolling_horizon(self, cv, data_model_dir):
        """Train the attention model using rolling horizon validation.
        
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

        time_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        results = {}
        results['history_loss'] = []
        results['history_metrics'] = []
        results['history_val_loss'] = []
        results['history_val_metrics'] = []
        results['train_loss'] = []
        results['train_metrics'] = []
        results['val_loss'] = []
        results['val_metrics'] = []

        # Save model parameters
        params = {k: str(v) if isinstance(v, (np.ndarray, list)) else v 
                 for k, v in self.__dict__.items() if k != 'model' 
                 and k != 'history'}
        

        for i in range(cv):
            # Reinitialize model with fresh weights for each fold
            self.model = self._build_model()
            # Use the same optimizer configuration as initial model
            if hasattr(self.model.optimizer, 'get_config'):
                optimizer_config = self.model.optimizer.get_config()
                optimizer = self.model.optimizer.__class__.from_config(optimizer_config)
            else:
                optimizer = 'adam'  # fallback to default
            self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)

            print(f"Fitting fold {i+1} of {cv}")

            X_past_train = np.load(data_model_dir / f"X_past_train_fold_{i+1}.npy")
            X_future_train = np.load(data_model_dir / f"X_future_train_fold_{i+1}.npy")
            y_train = np.load(data_model_dir / f"y_train_fold_{i+1}.npy")
            X_past_val = np.load(data_model_dir / f"X_past_val_fold_{i+1}.npy")
            X_future_val = np.load(data_model_dir / f"X_future_val_fold_{i+1}.npy")
            y_val = np.load(data_model_dir / f"y_val_fold_{i+1}.npy")

            train_past_times = pd.read_pickle(data_model_dir / f"train_past_times_fold_{i+1}.pkl")
            train_future_times = pd.read_pickle(data_model_dir / f"train_future_times_fold_{i+1}.pkl")
            val_past_times = pd.read_pickle(data_model_dir / f"val_past_times_fold_{i+1}.pkl")
            val_future_times = pd.read_pickle(data_model_dir / f"val_future_times_fold_{i+1}.pkl")

            print(f"Training period: {train_past_times[0][0]} to {train_future_times[-1][-1]}")
            print(f"Validation period: {val_past_times[0][0]} to {val_future_times[-1][-1]}")

            model_name = time_start + f"_fold_{i+1}"

            history, train_results, val_results = self.fit(X_past_train, 
                                                           X_future_train, 
                                                           y_train, 
                                                           X_past_val, 
                                                           X_future_val, 
                                                           y_val, 
                                                           model_name = model_name, 
                                                           rolling_horizon = True)

            # Store results, converting numpy arrays to lists where needed
            results['history_loss'].append([float(x) for x in history.history['loss']])
            # Store each metric separately
            for metric in self.metrics:
                metric_name = metric if isinstance(metric, str) else metric.__name__
                results['history_metrics'].append([float(x) for x in history.history[metric_name]])
                results['history_val_metrics'].append([float(x) for x in history.history[f'val_{metric_name}']])
            results['history_val_loss'].append([float(x) for x in history.history['val_loss']])
            results['train_loss'].append(float(train_results[0]))
            results['train_metrics'].append([float(x) for x in train_results[1:]])
            results['val_loss'].append(float(val_results[0]))
            results['val_metrics'].append([float(x) for x in val_results[1:]])

        
        # Save final results, converting numpy arrays to lists
        final_results = {
            'train_loss': float(np.mean(results['train_loss'])),
            'train_metrics': [float(x) for x in np.mean(results['train_metrics'], axis=0)],
            'val_loss': float(np.mean(results['val_loss'])),
            'val_metrics': [float(x) for x in np.mean(results['val_metrics'], axis=0)]
        }

        self.history = history

        # Print final results
        if self.verbose:
            print("\nFinal Training Metrics:")
            if self.loss == 'mse':
                print(f"RMSE: {np.sqrt(final_results['train_loss']):.4f}")
            else:
                print(f"{self.loss}: {final_results['train_loss']:.4f}")
            print(f"{self.metrics}: {final_results['train_metrics']}")
            print("\nFinal Validation Metrics:")
            if self.loss == 'mse':
                print(f"RMSE: {np.sqrt(final_results['val_loss']):.4f}")
            else:
                print(f"{self.loss}: {final_results['val_loss']:.4f}")
            print(f"{self.metrics}: {final_results['val_metrics']}")

        param_results = {
            'parameters': params,
            'rolling_horizon': True,
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

        return results
    
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
        """Create an AttentionModel instance from a saved model.

        This class method loads both the model weights and its parameters
        from saved files.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g. "20240315-123456"). The method will look
            for both .keras and parameters.json files in MODELS_PATH/ATT.

        Returns
        -------
        AttentionModel
            A new instance initialized with the correct parameters
        """
        # Construct paths
        model_dir = MODELS_PATH

        model_path = os.path.join(model_dir, f"{model_name}.keras")
        
        # Try to find parameters file (check both locations)
        params_path = os.path.join(model_dir, f"{model_name}_param_results.json")

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(params_path):
            param_results_path = os.path.join(
                model_dir, f'{model_name}_param_results.json'
            )

            raise FileNotFoundError(
                "Parameters file not found in either:\n"
                f"{param_results_path}"
            )

        # Load parameters
        with open(params_path, 'r') as f:
            params = json.load(f)['parameters']

        # Create instance with loaded parameters
        instance = cls(
            preprocess_version=params['preprocess_version'],
            model_version=params['model_version'],
            cnn_filters=params['cnn_filters'],
            lstm_units=params['lstm_units'],
            attention_heads=params['attention_heads'],
            attention_key_dim=params['attention_key_dim'],
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
            relative to the MODELS_PATH/ATT directory.

        Returns
        -------
        self : AttentionModel
            The instance with loaded model for method chaining
        """
        # Handle relative paths
        if not os.path.isabs(model_path):
            model_path = os.path.join(MODELS_PATH, "ATT", model_path)

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

    def clear_session(self):
        """Clear the tensorflow session.

        Used during recalibration to avoid RAM memory leakages. If the model is
        retrained continuously, tensorflow slightly increases RAM usage at each
        step.
        """
        K.clear_session()
