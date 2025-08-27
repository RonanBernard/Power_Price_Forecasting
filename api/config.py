from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
from scripts.config import ENTSOE_API_KEY

class Settings(BaseSettings):
    """API Configuration Settings"""
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Power Price Forecasting API"
    
    # ENTSOE Settings
    ENTSOE_API_KEY: str = ENTSOE_API_KEY
    
    # Model Settings
    MODEL_PATH: Optional[Path] = None
    
    # Deployment Settings
    ENVIRONMENT: str = "development"
    
    class Config:
        case_sensitive = True

settings = Settings()