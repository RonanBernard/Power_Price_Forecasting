"""Service for data preprocessing using saved sklearn pipelines."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List
import joblib
from datetime import datetime, timedelta

class PreprocessingService:
    def __init__(self, models_dir: str = "api/models"):
        """
        Initialize preprocessing service with saved sklearn pipelines.
        
        Args:
            models_dir: Directory containing saved models and pipelines
        """
        self.models_dir = models_dir
        
        # Load preprocessing pipelines
        self.load_pipelines()
        
    def load_pipelines(self):
        """Load all necessary preprocessing pipelines."""
        try:
            # Load the feature preprocessing pipeline
            self.feature_pipeline = joblib.load(
                os.path.join(self.models_dir, "feature_pipeline.joblib"))
            
            # Load the target preprocessing pipeline
            self.target_pipeline = joblib.load(
                os.path.join(self.models_dir, "target_pipeline.joblib"))
            
        except Exception as e:
            raise Exception(f"Error loading preprocessing pipelines: {e}")
    
    def create_sequences(self, data: pd.DataFrame, 
                        target_date: datetime,
                        past_seq_len: int = 168,  # 7 days * 24 hours
                        future_seq_len: int = 24   # 1 day ahead
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create past and future sequences for the ATT model.
        
        Args:
            data: Processed DataFrame with all features
            target_date: The date for which to make predictions
            past_seq_len: Length of past sequence
            future_seq_len: Length of future sequence
            
        Returns:
            Tuple of (past_sequence, future_sequence)
        """
        # Ensure data is sorted by time
        data = data.sort_index()
        
        # Get the sequence end points
        sequence_end = target_date
        sequence_start = sequence_end - timedelta(hours=past_seq_len)
        future_end = sequence_end + timedelta(hours=future_seq_len)
        
        # Extract sequences
        past_data = data[sequence_start:sequence_end]
        future_data = data[sequence_end:future_end]
        
        # Convert to numpy arrays and reshape for the model
        past_sequence = past_data.values.reshape(1, past_seq_len, -1)
        future_sequence = future_data.values.reshape(1, future_seq_len, -1)
        
        return past_sequence, future_sequence
    
    def preprocess_data(self, 
                       data: pd.DataFrame,
                       target_date: datetime
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data using saved pipelines and create sequences.
        
        Args:
            data: Raw data DataFrame
            target_date: Target date for prediction
            
        Returns:
            Tuple of (past_sequence, future_sequence) ready for the model
        """
        try:
            # Transform features using the saved pipeline
            transformed_features = self.feature_pipeline.transform(data)
            
            # Create sequences
            past_seq, future_seq = self.create_sequences(
                transformed_features, target_date)
            
            return past_seq, future_seq
            
        except Exception as e:
            raise Exception(f"Error in data preprocessing: {e}")
    
    def postprocess_predictions(self, predictions: np.ndarray) -> List[float]:
        """
        Convert model predictions back to original scale.
        
        Args:
            predictions: Model predictions
            
        Returns:
            List of price predictions in original scale
        """
        try:
            # Inverse transform predictions
            original_scale = self.target_pipeline.inverse_transform(
                predictions.reshape(-1, 1))
            
            return original_scale.flatten().tolist()
            
        except Exception as e:
            raise Exception(f"Error in prediction postprocessing: {e}")
