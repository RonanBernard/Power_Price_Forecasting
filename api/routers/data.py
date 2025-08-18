from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

router = APIRouter()

class FeatureRequest(BaseModel):
    """Request model for feature extraction"""
    target_date: datetime
    lookback_days: Optional[int] = 7
    include_weather: Optional[bool] = True
    include_load: Optional[bool] = True

class FeatureResponse(BaseModel):
    """Response model for feature extraction"""
    features: Dict
    feature_descriptions: Dict[str, str]
    timestamp: datetime

@router.post("/extract-features", response_model=FeatureResponse)
async def extract_features(request: FeatureRequest):
    """Extract features for a given target date"""
    try:
        # TODO: Implement feature extraction logic
        return FeatureResponse(
            features={},
            feature_descriptions={},
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-info")
async def get_feature_info():
    """Get information about available features"""
    try:
        return {
            "price_features": [
                "price_lag_24h",
                "price_lag_48h",
                "price_rolling_mean_7d"
            ],
            "load_features": [
                "load_forecast",
                "load_actual"
            ],
            "generation_features": [
                "wind_forecast",
                "solar_forecast",
                "nuclear_actual"
            ],
            "time_features": [
                "hour_of_day",
                "day_of_week",
                "is_weekend"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
