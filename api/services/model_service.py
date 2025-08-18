"""Service for loading and running the ATT model."""

import os
import numpy as np
from typing import List
import tensorflow as tf

class ModelService:
    def __init__(self, models_dir: str = "api/models"):
        """
        Initialize model service.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = models_dir
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the ATT model from disk."""
        try:
            model_path = os.path.join(self.models_dir, "att_model.keras")
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise Exception(f"Error loading ATT model: {e}")
    
    def predict(self, 
                past_sequence: np.ndarray,
                future_sequence: np.ndarray
                ) -> np.ndarray:
        """
        Make predictions using the ATT model.
        
        Args:
            past_sequence: Preprocessed past sequence data
            future_sequence: Preprocessed future sequence data
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise Exception("Model not loaded")
            
        try:
            predictions = self.model.predict(
                [past_sequence, future_sequence],
                verbose=0
            )
            return predictions
            
        except Exception as e:
            raise Exception(f"Error making predictions: {e}")
