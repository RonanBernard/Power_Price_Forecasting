from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """API Configuration Settings"""
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Power Price Forecasting API"
    
    # Model Settings
    MODEL_PATH: Optional[Path] = None
    
    # Deployment Settings
    ENVIRONMENT: str = "development"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
