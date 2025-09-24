# MLflow Server on Google Cloud Platform

This directory contains the MLflow server setup for tracking machine learning experiments and managing model artifacts on Google Cloud Platform.

## Architecture

- **MLflow Server**: Deployed on Cloud Run (serverless)
- **Backend Store**: Cloud SQL PostgreSQL database
- **Artifact Store**: Google Cloud Storage bucket
- **Authentication**: Google Cloud IAM with identity tokens

## Files

- `Dockerfile`: Container image for the MLflow server
- `README.md`: This documentation file

## Prerequisites

- Google Cloud Project with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed
- Required APIs enabled:
  - Cloud Run API
  - Cloud SQL Admin API
  - Artifact Registry API
  - Cloud Storage API

## Environment Variables

Set these in your `.env` file:

```bash
# GCP Configuration
GCP_PROJECT=your-project-id
GCP_REGION=europe-west1
GCP_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com

# MLflow Configuration
MLFLOW_TRACKING_URI=https://your-mlflow-url.run.app
MLFLOW_DB_PW=your-strong-password

# Cloud Storage
CS_BUCKET=your-mlflow-bucket

# Cloud SQL
INSTANCE_CONN=project:region:instance-name
```

## Deployment

### 1. Create Infrastructure

```bash
# Set project and region
gcloud config set project ${GCP_PROJECT}
gcloud config set run/region ${GCP_REGION}

# Enable required APIs
gcloud services enable run.googleapis.com sqladmin.googleapis.com artifactregistry.googleapis.com

# Create GCS bucket
gsutil mb -l ${GCP_REGION} gs://${CS_BUCKET}
gsutil versioning set on gs://${CS_BUCKET}

# Create Cloud SQL PostgreSQL
gcloud sql instances create mlflow-pg --database-version=POSTGRES_14 --cpu=1 --memory=3840MiB --region=${GCP_REGION} --storage-auto-increase
gcloud sql databases create mlflowdb --instance=mlflow-pg
gcloud sql users create mlflow --instance=mlflow-pg --password=${MLFLOW_DB_PW}
```

### 2. Set Up Service Account

```bash
# Grant bucket permissions
gsutil iam ch serviceAccount:${GCP_SERVICE_ACCOUNT}:roles/storage.objectAdmin gs://${CS_BUCKET}

# Grant Cloud SQL permissions
gcloud projects add-iam-policy-binding ${GCP_PROJECT} \
  --member="serviceAccount:${GCP_SERVICE_ACCOUNT}" \
  --role="roles/cloudsql.client"
```

### 3. Build and Deploy

```bash
# Create Artifact Registry repository
gcloud artifacts repositories create mlflow-repo --repository-format=docker --location=${GCP_REGION}

# Build and push Docker image
docker build -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/mlflow-repo/mlflow:latest .
docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/mlflow-repo/mlflow:latest

# Deploy to Cloud Run
gcloud run deploy mlflow \
  --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/mlflow-repo/mlflow:latest \
  --service-account ${GCP_SERVICE_ACCOUNT} \
  --add-cloudsql-instances ${INSTANCE_CONN} \
  --set-env-vars BACKEND_STORE_URI="postgresql+psycopg2://mlflow:${MLFLOW_DB_PW}@/mlflowdb?host=/cloudsql/${INSTANCE_CONN}",DEFAULT_ARTIFACT_ROOT="gs://${CS_BUCKET}/mlflow-artifacts" \
  --no-allow-unauthenticated \
  --memory=1Gi --cpu=1 --min-instances=1 \
  --timeout=900 \
  --port=5000
```

### 4. Grant Access

```bash
# Grant yourself access to the service
gcloud run services add-iam-policy-binding mlflow \
  --member="user:your-email@gmail.com" \
  --role="roles/run.invoker"
```

## Usage

### Accessing the MLflow UI

Since the service requires authentication, use the proxy method:

```bash
gcloud run services proxy mlflow --region=${GCP_REGION} --port=8080
```

Then open http://localhost:8080 in your browser.

### Using in Training Code

The `AttentionModel` class automatically handles MLflow integration:

```python
from scripts.model_att_v2log import AttentionModel

# Create model with MLflow enabled
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
    mlflow_enabled=True,  # Enable MLflow tracking
    mlflow_nested=False
)

# Train the model (automatically logs to MLflow)
history = model.fit(X_past_train, X_future_train, y_train, 
                   X_past_val, X_future_val, y_val)
```

### Loading Models in Production

```python
import mlflow
import mlflow.tensorflow

# Set tracking URI
mlflow.set_tracking_uri("https://your-mlflow-url.run.app")

# Load the latest Production model
model = mlflow.tensorflow.load_model("models:/PowerPriceV2/Production")
```

### Promoting Models to Production

```bash
# Transition a model version to Production stage
mlflow models transition-version-stage \
  -m "PowerPriceV2" -v 1 -s Production \
  --archive-existing-versions
```

## Configuration

### Experiment Name

The default experiment name is set in `scripts/config.py`:

```python
MLFLOW_EXPERIMENT = 'French_DAM_PowerPriceForecasting'
```

### Tracking URI

Set in your `.env` file:

```bash
MLFLOW_TRACKING_URI="https://your-mlflow-url.run.app"
```

## Troubleshooting

### Authentication Issues

If you get 403 Forbidden errors:

1. Ensure you're using the correct Google account
2. Grant yourself access: `gcloud run services add-iam-policy-binding mlflow --member="user:your-email" --role="roles/run.invoker"`
3. Use the proxy method: `gcloud run services proxy mlflow --region=${GCP_REGION} --port=8080`

### Database Connection Issues

1. Verify Cloud SQL instance is running
2. Check the connection string format
3. Ensure the service account has `cloudsql.client` role

### Storage Issues

1. Verify the GCS bucket exists and is accessible
2. Check service account has `storage.objectAdmin` role on the bucket
3. Ensure bucket versioning is enabled

## Security Notes

- The MLflow server requires authentication (no public access)
- All data is stored in your GCP project (PostgreSQL + GCS)
- Service account has minimal required permissions
- Identity tokens are used for authentication

## Cost Optimization

- Cloud Run scales to zero when not in use
- Set `min-instances=1` only if you need low-latency access
- Use appropriate Cloud SQL machine types
- GCS storage is cost-effective for artifacts

## Monitoring

- Monitor Cloud Run service health in the GCP Console
- Set up alerts for Cloud SQL and Cloud Run
- Check MLflow logs: `gcloud run services logs read mlflow --region=${GCP_REGION}`
