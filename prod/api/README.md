# Power Price Forecasting API

This is the API service for the Power Price Forecasting project. It provides endpoints for making electricity price predictions using trained machine learning models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the following content:
```env
ENVIRONMENT=development
MODEL_PATH=/path/to/your/model
```

3. Run the API:
```bash
python api/main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the API is running, you can access:
- Interactive API documentation (Swagger UI) at `http://localhost:8000/docs`
- Alternative API documentation (ReDoc) at `http://localhost:8000/redoc`

## Endpoints

### Health Check
- `GET /health`: Check if the API is running

### Predictions
- `POST /api/v1/predictions/predict`: Get price predictions for a specific date

## Development

The API is built using FastAPI and follows a modular structure:
- `main.py`: Application entry point and configuration
- `config.py`: Configuration settings
- `routers/`: API route handlers
  - `predictions.py`: Prediction-related endpoints
