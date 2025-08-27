"""Service for validating models and pipelines."""

import os
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple, List

class ValidationService:
    def __init__(self, models_dir: str = "api/models"):
        self.models_dir = models_dir
        
    def validate_att_model(self, model: tf.keras.Model) -> Tuple[bool, List[str]]:
        """
        Validate the ATT model structure and configuration.
        
        Args:
            model: Loaded ATT model
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check if model has two inputs (past and future)
            if len(model.inputs) != 2:
                issues.append(f"Expected 2 inputs, got {len(model.inputs)}")
            
            # Check input shapes
            past_input = model.inputs[0]
            future_input = model.inputs[1]
            
            if len(past_input.shape) != 3:  # (batch, sequence_length, features)
                issues.append(f"Past input shape should be 3D, got {len(past_input.shape)}D")
            
            if len(future_input.shape) != 3:
                issues.append(f"Future input shape should be 3D, got {len(future_input.shape)}D")
            
            # Check output shape (should be 24 for day-ahead predictions)
            output = model.outputs[0]
            if output.shape[-1] != 24:
                issues.append(f"Output should have 24 values, got {output.shape[-1]}")
            
            # Test with dummy data
            batch_size = 1
            past_seq = np.zeros((batch_size,) + tuple(past_input.shape[1:]))
            future_seq = np.zeros((batch_size,) + tuple(future_input.shape[1:]))
            
            try:
                _ = model.predict([past_seq, future_seq], verbose=0)
            except Exception as e:
                issues.append(f"Model prediction test failed: {str(e)}")
                
        except Exception as e:
            issues.append(f"Model validation failed: {str(e)}")
            
        return len(issues) == 0, issues
    
    def validate_feature_pipeline(self, pipeline: Pipeline) -> Tuple[bool, List[str]]:
        """
        Validate the feature preprocessing pipeline.
        
        Args:
            pipeline: Loaded sklearn pipeline
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check if pipeline is a sklearn Pipeline
            if not isinstance(pipeline, Pipeline):
                issues.append("Feature pipeline is not a sklearn Pipeline")
                return False, issues
            
            # Check expected steps in pipeline
            expected_steps = ['imputer', 'scaler']  # Add your expected steps
            for step in expected_steps:
                if not any(s.startswith(step) for s in pipeline.named_steps):
                    issues.append(f"Missing expected step: {step}")
            
            # Check if pipeline has get_feature_names_out method (for column transformers)
            if hasattr(pipeline, 'get_feature_names_out'):
                try:
                    feature_names = pipeline.get_feature_names_out()
                    if len(feature_names) == 0:
                        issues.append("Pipeline returns empty feature names")
                except Exception as e:
                    issues.append(f"Error getting feature names: {str(e)}")
            
        except Exception as e:
            issues.append(f"Pipeline validation failed: {str(e)}")
            
        return len(issues) == 0, issues
    
    def validate_target_pipeline(self, pipeline: Pipeline) -> Tuple[bool, List[str]]:
        """
        Validate the target preprocessing pipeline.
        
        Args:
            pipeline: Loaded sklearn pipeline
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check if pipeline is a sklearn Pipeline
            if not isinstance(pipeline, Pipeline):
                issues.append("Target pipeline is not a sklearn Pipeline")
                return False, issues
            
            # Test inverse transform
            test_data = np.array([[100.0], [200.0], [300.0]])
            try:
                transformed = pipeline.transform(test_data)
                original = pipeline.inverse_transform(transformed)
                
                if not np.allclose(test_data, original, rtol=1e-10):
                    issues.append("Pipeline inverse transform test failed")
                    
            except Exception as e:
                issues.append(f"Pipeline transform test failed: {str(e)}")
            
        except Exception as e:
            issues.append(f"Pipeline validation failed: {str(e)}")
            
        return len(issues) == 0, issues
    
    def validate_all(self) -> Tuple[bool, dict]:
        """
        Validate all models and pipelines.
        
        Returns:
            Tuple of (all_valid, validation_results)
        """
        results = {}
        
        # Load and validate ATT model
        try:
            model = tf.keras.models.load_model(
                os.path.join(self.models_dir, "att_model.keras"))
            model_valid, model_issues = self.validate_att_model(model)
            results['att_model'] = {
                'valid': model_valid,
                'issues': model_issues
            }
        except Exception as e:
            results['att_model'] = {
                'valid': False,
                'issues': [f"Failed to load model: {str(e)}"]
            }
        
        # Load and validate feature pipeline
        try:
            feature_pipeline = joblib.load(
                os.path.join(self.models_dir, "feature_pipeline.joblib"))
            pipeline_valid, pipeline_issues = self.validate_feature_pipeline(feature_pipeline)
            results['feature_pipeline'] = {
                'valid': pipeline_valid,
                'issues': pipeline_issues
            }
        except Exception as e:
            results['feature_pipeline'] = {
                'valid': False,
                'issues': [f"Failed to load pipeline: {str(e)}"]
            }
        
        # Load and validate target pipeline
        try:
            target_pipeline = joblib.load(
                os.path.join(self.models_dir, "target_pipeline.joblib"))
            target_valid, target_issues = self.validate_target_pipeline(target_pipeline)
            results['target_pipeline'] = {
                'valid': target_valid,
                'issues': target_issues
            }
        except Exception as e:
            results['target_pipeline'] = {
                'valid': False,
                'issues': [f"Failed to load pipeline: {str(e)}"]
            }
        
        # Check if all components are valid
        all_valid = all(r['valid'] for r in results.values())
        
        return all_valid, results
