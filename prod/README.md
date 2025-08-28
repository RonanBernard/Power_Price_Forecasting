# Power Price Forecasting API - Production Deployment

This guide explains how to deploy the Power Price Forecasting API to Google Cloud Platform (GCP).

## Prerequisites

1. Install and set up the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Enable the following GCP services:
   - Cloud Run
   - Container Registry
   - Secret Manager

## Environment Setup

1. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

2. Fill in the required environment variables in `.env`:
   - `ENTSOE_API_KEY`: Your ENTSOE API key
   - `ENVIRONMENT`: Set to "production"
   - `MODEL_PATH`: Path to your trained model

## Local Testing

1. Build the Docker image:
   ```bash
   docker build -t power-price-api .
   ```

2. Run the container locally:
   ```bash
   docker run -p 8000:8000 --env-file .env power-price-api
   ```

3. Test the API at `http://localhost:8000`

## GCP Deployment

1. Set up GCP project and authentication:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Store sensitive environment variables in Secret Manager:
   ```bash
   echo -n "your_entsoe_api_key" | gcloud secrets create entsoe-api-key --data-file=-
   ```

3. Build and push the Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/power-price-api
   ```

4. Deploy to Cloud Run:
   ```bash
   gcloud run deploy power-price-api \
     --image gcr.io/YOUR_PROJECT_ID/power-price-api \
     --platform managed \
     --region europe-west1 \
     --allow-unauthenticated \
     --set-secrets=ENTSOE_API_KEY=entsoe-api-key:latest \
     --set-env-vars="ENVIRONMENT=production"
   ```

## Important Notes

1. Security:
   - Always use Secret Manager for sensitive data
   - Consider implementing authentication for the API
   - Review and restrict CORS settings in `main.py`

2. Monitoring:
   - Set up Cloud Monitoring for metrics and alerts
   - Configure logging to track API usage and errors

3. Scaling:
   - Cloud Run automatically scales based on traffic
   - Configure memory and CPU limits as needed

4. Cost Optimization:
   - Monitor usage and adjust resources accordingly
   - Set budget alerts in GCP
   - Consider using preemptible instances for non-critical workloads

## Maintenance

1. Updating the API:
   ```bash
   # Build and push new version
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/power-price-api
   
   # Deploy update
   gcloud run deploy power-price-api \
     --image gcr.io/YOUR_PROJECT_ID/power-price-api
   ```

2. Rolling back:
   ```bash
   # List revisions
   gcloud run revisions list --service power-price-api
   
   # Roll back to specific revision
   gcloud run services update-traffic power-price-api \
     --to-revision=power-price-api-00001-abc
   ```
