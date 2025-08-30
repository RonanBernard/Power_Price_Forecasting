# Power Price Forecasting API - Production Deployment

This guide explains how to deploy the Power Price Forecasting API to Google Cloud Platform (GCP) using Google Artifact Registry and Cloud Run.

## Prerequisites

1. Install and set up the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Enable the following GCP services:
   - Cloud Run
   - Artifact Registry
   - BigQuery (for data storage)

## Environment Setup

1. Create a `.env.yaml` file with the following structure:
   ```yaml
   ENVIRONMENT: "production"
   ENTSOE_API_KEY: "your-entsoe-api-key"
   GCP_PROJECT: "your-project-id"
   GCP_REGION: "europe-west1"
   BQ_REGION: "europe-west1"
   BQ_DATASET: "entsoe_data"
   BQ_TABLE_PRICES: "day_ahead_prices"
   BQ_TABLE_FLOWS: "crossborder_flows"
   BQ_TABLE_GENERATION: "generation_data"
   BQ_TABLE_LOAD: "load_data"
   BQ_TABLE_WIND_SOLAR: "wind_solar_forecast"
   AR_REPO_NAME: "power-da-price"
   AR_REPO_REGION: "europe-west1"
   ```

2. Set up environment variables in your shell:
   ```bash
   export AR_REPO_REGION="europe-west1"
   export GCP_PROJECT="your-project-id"
   export AR_REPO_NAME="power-da-price"
   export DOCKER_IMAGE="power-da-price"
   export GCP_REGION="europe-west1"
   export AR_MEMORY="1Gi"
   ```

## Docker Build and Push

1. Build the Docker image:
   ```bash
   docker build -t "${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest" .
   ```

2. Push to Google Artifact Registry:
   ```bash
   docker push "${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest"
   ```

## Cloud Run Deployment

Deploy to Cloud Run using the following command:

```bash
gcloud run deploy power-da-price \
  --image=${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest \
  --platform=managed \
  --region=${GCP_REGION} \
  --memory=${AR_MEMORY} \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=2 \
  --env-vars-file=.env.yaml \
  --allow-unauthenticated
```

## API Access

Your API is deployed and accessible at:
**https://power-da-price-1040927723543.europe-west1.run.app**

### Test Endpoints

```bash
# Root endpoint
curl https://power-da-price-1040927723543.europe-west1.run.app/

# Health check
curl https://power-da-price-1040927723543.europe-west1.run.app/api/v1/health

# Predictions endpoint
curl https://power-da-price-1040927723543.europe-west1.run.app/api/v1/predictions
```

## Configuration Details

- **Service Name**: `power-da-price`
- **Region**: `europe-west1`
- **Memory**: `1Gi`
- **CPU**: `1`
- **Port**: `8080` (Cloud Run default)
- **Authentication**: Public (--allow-unauthenticated)
- **Scaling**: 0-2 instances

## Troubleshooting

### Check Cloud Run Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=power-da-price" --limit=50
```

### Check Service Status
```bash
gcloud run services describe power-da-price --region=europe-west1
```

### List Revisions
```bash
gcloud run revisions list --service=power-da-price --region=europe-west1
```

## Important Notes

1. **Security**:
   - API is currently public (--allow-unauthenticated)
   - Consider implementing authentication for production use
   - Review CORS settings in `main.py`

2. **Monitoring**:
   - Set up Cloud Monitoring for metrics and alerts
   - Configure logging to track API usage and errors

3. **Scaling**:
   - Cloud Run automatically scales based on traffic
   - Configured with 0-2 instances for cost optimization

4. **Cost Optimization**:
   - Monitor usage and adjust resources accordingly
   - Set budget alerts in GCP
   - Min instances set to 0 for cost savings

## Maintenance

### Updating the API
```bash
# Build and push new version
docker build -t "${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest" .
docker push "${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest"

# Deploy update
gcloud run deploy power-da-price \
  --image=${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest
```

### Rolling Back
```bash
# List revisions
gcloud run revisions list --service=power-da-price --region=europe-west1

# Roll back to specific revision
gcloud run services update-traffic power-da-price \
  --to-revision=power-da-price-00001-abc
```

## Complete Deployment Flow

1. **Build**: `docker build -t [IMAGE_TAG] .`
2. **Push**: `docker push [IMAGE_TAG]`
3. **Deploy**: `gcloud run deploy [SERVICE_NAME] --image=[IMAGE_TAG] [OPTIONS]`
4. **Test**: `curl [SERVICE_URL]`

Your API is now live and ready to handle requests! 🚀
