"""Router for model and pipeline validation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from api.services.validation_service import ValidationService

router = APIRouter()
validation_service = ValidationService()

class ValidationResponse(BaseModel):
    """Response model for validation results"""
    all_valid: bool
    components: Dict[str, Dict[str, Optional[List[str]]]]

@router.get("/validate", response_model=ValidationResponse)
async def validate_system():
    """
    Validate all system components (model and pipelines).
    
    Returns:
        Validation results for each component
    """
    try:
        all_valid, results = validation_service.validate_all()
        
        return ValidationResponse(
            all_valid=all_valid,
            components=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )
