from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from api.services.entsoe_service import EntsoeService
from api.services.preprocessing_service import PreprocessingService
from api.services.model_service import ModelService

router = APIRouter()

# Service instances
entsoe_service = EntsoeService()
preprocessing_service = PreprocessingService()
model_service = ModelService()

class PricePredictionRequest(BaseModel):
    """Request model for price prediction"""
    target_date: datetime

class PricePredictionResponse(BaseModel):
    """Response model for price prediction"""
    target_date: datetime
    predicted_prices: List[float]
    prediction_time: datetime = datetime.now()

@router.post("/predict", response_model=PricePredictionResponse)
async def predict_prices(request: PricePredictionRequest):
    """
    Predict electricity prices for a given date using the ATT model.
    
    The process involves:
    1. Downloading required data from ENTSOE
    2. Preprocessing the data using saved pipelines
    3. Running the ATT model
    4. Returning the predictions
    """
    try:
        # 1. Get data from ENTSOE
        raw_data = entsoe_service.get_data_for_date(request.target_date)
        
        # 2. Process raw data into a DataFrame
        processed_df = entsoe_service.process_raw_data(raw_data)
        
        # 3. Preprocess data for the model
        past_sequence, future_sequence = preprocessing_service.preprocess_data(
            processed_df, request.target_date)
        
        # 4. Make predictions
        predictions = model_service.predict(past_sequence, future_sequence)
        
        # 5. Postprocess predictions
        final_predictions = preprocessing_service.postprocess_predictions(predictions)
        
        return PricePredictionResponse(
            target_date=request.target_date,
            predicted_prices=final_predictions,
            prediction_time=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
