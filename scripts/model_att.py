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
        model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = LOGS_PATH / "ATT" / "fit" / model_name
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
            [X_past_train, X_future_train],
            y_train,
            epochs=1000,
            batch_size=32,
            validation_data=([X_past_val, X_future_val], y_val),
            callbacks=[early_stopping, tensorboard_callback],
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

        # Save the model
        self.model.save(os.path.join(MODELS_PATH, "ATT", f"{model_name}.keras"))

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

    def clear_session(self):
        """Clear the tensorflow session.

        Used during recalibration to avoid RAM memory leakages. If the model is
        retrained continuously, tensorflow slightly increases RAM usage at each
        step.
        """
        K.clear_session()
