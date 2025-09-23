from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import logging
import keras as kr

from api.services.preprocessing_service import main as preprocess_data
from api.config import API_MODELS_PATH


# Configure logging
logger = logging.getLogger(__name__)


router = APIRouter()


class PricePredictionRequest(BaseModel):
    """Request model for price prediction"""
    date: str  # Format: DD/MM/YYYY
    entsoe_api_key: str  # ENTSOE API key for data fetching

    @field_validator('date')
    def validate_date_format(cls, v: str) -> pd.Timestamp:
        try:
            # Parse the date string
            day, month, year = map(int, v.split('/'))
            # Convert to pandas Timestamp with Paris timezone
            ts = pd.Timestamp(f"{year}-{month:02d}-{day:02d}", tz="Europe/Paris")
            return ts
        except Exception as e:
            raise ValueError("Date must be in format DD/MM/YYYY") from e


class PricePredictionResponse(BaseModel):
    """Response model for price prediction"""
    target_date: datetime
    predicted_prices: List[float]
    actual_prices: List[float]
    prediction_time: datetime = datetime.now()


@router.post("/predict", response_model=PricePredictionResponse)
async def predict_prices(request: PricePredictionRequest):
    """
    Predict day-ahead electricity prices for a given target date.
    
    The process involves:
    1. Preprocessing the data using the main preprocessing pipeline
    2. Loading the LSTM model
    3. Making predictions
    4. Returning the results

    Example request:
        {
            "date": "19/08/2025"
        }
    """
    try:
        # 1. Preprocess data
        logger.info(f"Preprocessing data for date: {request.date}")
        data_past_transformed, data_future_transformed, data_target = (
            preprocess_data(request.date, request.entsoe_api_key)
        )
        
        # 2. Load the model using the proper class method
        logger.info("Loading model")
        model = kr.models.load_model(API_MODELS_PATH / "model.keras")
        
        # 3. Make predictions
        # Ensure data is in the right shape (samples, sequence_length, features)
        if len(data_past_transformed.shape) == 2:
            data_past_transformed = np.expand_dims(data_past_transformed, axis=0)
        if len(data_future_transformed.shape) == 2:
            data_future_transformed = np.expand_dims(data_future_transformed, axis=0)
        
        logger.info("Making predictions")
        predictions_log = model.predict([data_past_transformed, data_future_transformed], verbose=1)

        predictions = np.sign(predictions_log) * np.expm1(np.abs(predictions_log))
        
        # 4. Convert predictions to list and return
        predictions_list = predictions.flatten().tolist()
        predictions_list = [round(p, 2) for p in predictions_list]
        actual_prices = data_target.tolist() if data_target is not None else [None] * len(predictions_list)
        
        logger.info("Prediction successful")
        return PricePredictionResponse(
            target_date=request.date,
            predicted_prices=predictions_list,
            actual_prices=actual_prices,
            prediction_time=datetime.now()
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model or parameter file not found: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Model files not found. Please ensure the model is properly installed."
        )
    except ValueError as e:
        logger.error(f"Invalid input data: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )