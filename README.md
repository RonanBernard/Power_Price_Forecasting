# Power Price Forecasting

A comprehensive machine learning platform for forecasting day-ahead electricity prices on European power markets, with a focus on the French EPEX market. The project includes data collection, preprocessing, model training, and production deployment capabilities.

## 🎯 Project Overview

This project provides an end-to-end solution for electricity price forecasting, combining:
- **Data Collection**: Automated data gathering from ENTSO-E Transparency Platform
- **Feature Engineering**: Advanced preprocessing with market-specific indicators
- **Model Training**: Deep learning models with attention mechanisms
- **Experiment Tracking**: MLflow integration for model management
- **Production Deployment**: API service for real-time predictions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ML Platform   │    │   Production    │
│                 │    │                 │    │                 │
│ • ENTSO-E API   │───▶│ • MLflow (GCP)  │───▶│ • FastAPI       │
│ • Fuel Prices   │    │ • Cloud SQL     │    │ • Docker        │
│ • Weather Data  │    │ • Cloud Storage │    │ • Cloud Run     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.12+**: Primary development language
- **TensorFlow/Keras**: Deep learning framework
- **Pandas/NumPy**: Data manipulation and analysis
- **SQLite/PostgreSQL**: Data storage and management

### Cloud Infrastructure (GCP)
- **Cloud Run**: Serverless container deployment
- **Cloud SQL**: PostgreSQL database for MLflow
- **Cloud Storage**: Artifact and data storage
- **Artifact Registry**: Container image management

### ML/AI Stack
- **MLflow**: Experiment tracking and model registry
- **TensorBoard**: Training visualization
- **Scikit-learn**: Traditional ML algorithms
- **Jupyter**: Interactive development

### Production Stack
- **FastAPI**: High-performance API framework
- **Docker**: Containerization
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## 📁 Project Structure

```
Power_Price_Forecasting/
├── research/                    # Research and development
│   ├── data/                   # Data storage
│   │   ├── entsoe_data.sqlite  # Historical data database
│   │   ├── fuel_prices.csv     # Fuel price data
│   │   └── v5/                 # Processed datasets
│   ├── scripts/                # Data processing and modeling
│   │   ├── download_entsoe_data.py    # Data collection
│   │   ├── preprocess_V1.py          # Data preprocessing
│   │   ├── model_att_v2log.py        # Attention model with MLflow
│   │   └── config.py                 # Configuration
│   ├── mlflow/                 # MLflow server deployment
│   │   ├── Dockerfile          # MLflow container
│   │   └── README.md           # MLflow setup guide
│   ├── notebooks/              # Jupyter notebooks
│   └── README.md               # Research documentation
├── prod/                       # Production API
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API entry point
│   │   ├── routers/           # API endpoints
│   │   └── models/            # Data models
│   ├── Dockerfile             # Production container
│   └── README.md              # Production documentation
├── front/                      # Frontend (optional)
│   ├── app.py                 # Streamlit dashboard
│   └── utils.py               # Frontend utilities
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker
- Google Cloud SDK (`gcloud`)
- ENTSO-E API key

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Power_Price_Forecasting

# Create virtual environment
python -m venv power_env
source power_env/bin/activate  # Linux/Mac
# or
power_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
# ENTSO-E API
ENTSOE_API_KEY=your_entsoe_api_key

# GCP Configuration
GCP_PROJECT=your-project-id
GCP_REGION=europe-west1
GCP_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com

# MLflow Configuration
MLFLOW_TRACKING_URI=https://your-mlflow-url.run.app
MLFLOW_DB_PW=your-strong-password

# Cloud Storage
CS_BUCKET=your-mlflow-bucket
INSTANCE_CONN=project:region:instance-name
```

### 3. Data Collection

```bash
cd research
python scripts/download_entsoe_data.py
```

### 4. Model Training

```python
from scripts.model_att_v2log import AttentionModel

# Create and train model with MLflow tracking
model = AttentionModel(
    preprocess_version="v5",
    cnn_filters=64,
    lstm_units=128,
    attention_heads=8,
    attention_key_dim=32,
    n_past_features=10,
    n_future_features=5,
    past_seq_len=168,
    future_seq_len=24,
    mlflow_enabled=True,
    mlflow_nested=False
)

# Train the model
history = model.fit(X_past_train, X_future_train, y_train, 
                   X_past_val, X_future_val, y_val)
```

### 5. Production Deployment

```bash
cd prod
docker build -t power-price-api .
docker run -p 8000:8000 power-price-api
```

## 📊 Data Sources

### ENTSO-E Transparency Platform
- **Day-ahead prices**: Historical and forecast prices
- **Load data**: Actual and forecasted electricity demand
- **Generation data**: Production by source (nuclear, wind, solar, etc.)
- **Cross-border flows**: Electricity imports/exports

### Market Coverage
- **France** (EPEX Paris)
- **Germany** (EPEX Frankfurt)
- **Belgium** (EPEX Brussels)
- **Switzerland** (EPEX Zurich)
- **Spain** (OMIE)
- **Italy** (North zone)

### Additional Data
- **Fuel prices**: Natural gas (TTF), coal (ARA), carbon (EUA)
- **Weather forecasts**: Wind and solar generation forecasts

## 🤖 Model Architecture

### Attention-Based Model (v2log)
The primary model combines multiple neural network components:

1. **Encoder**: Causal dilated convolutions + BiLSTM + self-attention
2. **Context Processing**: Global pooling + dense transformation
3. **Decoder**: Cross-attention + time-distributed output
4. **AR-MIMO Head**: Linear autoregressive component
5. **Gated Blending**: Learned combination of neural and AR outputs
6. **Target Transformation**: Signed log transformation for stability

### Key Features
- **Multi-horizon prediction**: 24-hour ahead forecasts
- **Attention mechanisms**: Captures complex temporal dependencies
- **Autoregressive component**: Leverages recent price patterns
- **Log transformation**: Handles negative prices and improves stability
- **MLflow integration**: Automatic experiment tracking

## 🔬 Experiment Tracking

### MLflow Setup
- **Cloud deployment**: MLflow server on Google Cloud Run
- **Database backend**: Cloud SQL PostgreSQL
- **Artifact storage**: Google Cloud Storage
- **Authentication**: Google Cloud IAM

### Tracking Features
- **Parameters**: Model hyperparameters and configuration
- **Metrics**: Training and validation metrics
- **Artifacts**: Model files, plots, and data
- **Model registry**: Version control and production deployment

### Accessing MLflow UI
```bash
gcloud run services proxy mlflow --region=${GCP_REGION} --port=8080
# Open http://localhost:8080
```

## 🚀 Production API

### FastAPI Service
- **Real-time predictions**: RESTful API for price forecasts
- **Model loading**: Automatic loading from MLflow Model Registry
- **Health checks**: Service monitoring and status endpoints
- **Docker deployment**: Containerized for easy deployment

### API Endpoints
- `GET /health`: Service health check
- `POST /predict`: Generate price forecasts
- `GET /models`: List available models
- `GET /metrics`: Model performance metrics

### Deployment
```bash
# Build and deploy to Cloud Run
gcloud run deploy power-price-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

## 📈 Performance

### Model Performance (French Market)
- **Training RMSE**: ~31 EUR/MWh (original scale)
- **Validation RMSE**: ~32 EUR/MWh (original scale)
- **Training MAE**: ~18 EUR/MWh (original scale)
- **Validation MAE**: ~25 EUR/MWh (original scale)

### Infrastructure Performance
- **Data collection**: Automated daily updates
- **Training time**: ~30-60 minutes per model
- **Prediction latency**: <100ms for 24-hour forecasts
- **API availability**: 99.9% uptime target

## 🔧 Development

### Local Development
```bash
# Start development environment
cd research
jupyter lab

# Run tests
pytest tests/

# Code formatting
black scripts/
isort scripts/
```

### Adding New Features
1. **Data sources**: Add new data collection scripts in `research/scripts/`
2. **Models**: Implement new models following the `AttentionModel` pattern
3. **API endpoints**: Add new routes in `prod/api/routers/`
4. **Frontend**: Extend the Streamlit dashboard in `front/`

## 📚 Documentation

- **[Research Guide](research/README.md)**: Data collection, preprocessing, and modeling
- **[MLflow Setup](research/mlflow/README.md)**: Experiment tracking deployment
- **[Production Guide](prod/README.md)**: API deployment and usage
- **[Frontend Guide](front/README.md)**: Dashboard setup and usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ENTSO-E**: For providing the transparency platform API
- **EPEX**: For electricity market data
- **Google Cloud**: For cloud infrastructure services
- **MLflow**: For experiment tracking capabilities

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the documentation in each module
- Review the troubleshooting guides

---

**Built with ❤️ for the energy transition**
