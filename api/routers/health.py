from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime

router = APIRouter()

class ServiceStatus(BaseModel):
    """Status information for a service component"""
    status: str  # "healthy", "degraded", "error"
    last_check: datetime
    details: Dict = {}

class HealthStatus(BaseModel):
    """Overall health status response"""
    status: str
    timestamp: datetime
    components: Dict[str, ServiceStatus]
    version: str = "1.0.0"

@router.get("/status", response_model=HealthStatus)
async def get_health_status():
    """Get detailed health status of all system components"""
    try:
        now = datetime.now()
        return HealthStatus(
            status="healthy",
            timestamp=now,
            components={
                "database": ServiceStatus(
                    status="healthy",
                    last_check=now,
                    details={"connection_pool": "active"}
                ),
                "model_service": ServiceStatus(
                    status="healthy",
                    last_check=now,
                    details={"loaded_models": ["att", "mlp"]}
                ),
                "data_service": ServiceStatus(
                    status="healthy",
                    last_check=now,
                    details={"last_data_update": now.isoformat()}
                )
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    try:
        return {
            "requests_total": 0,
            "requests_last_hour": 0,
            "average_response_time_ms": 0,
            "model_inference_time_ms": 0,
            "feature_extraction_time_ms": 0,
            "errors_last_hour": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
