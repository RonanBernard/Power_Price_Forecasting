from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.config import settings
from api.routers import predictions, models, data, health, validation

app = FastAPI(
    title="Power Price Forecasting API",
    description="API for predicting electricity prices using machine learning models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint returning API status"""
    return {"status": "online", "message": "Power Price Forecasting API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Include routers
app.include_router(
    predictions.router,
    prefix=f"{settings.API_V1_STR}/predictions",
    tags=["predictions"]
)

app.include_router(
    models.router,
    prefix=f"{settings.API_V1_STR}/models",
    tags=["models"]
)

app.include_router(
    data.router,
    prefix=f"{settings.API_V1_STR}/data",
    tags=["data"]
)

app.include_router(
    health.router,
    prefix=f"{settings.API_V1_STR}/health",
    tags=["health"]
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
