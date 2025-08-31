docker build -t "${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest" .
docker push "${AR_REPO_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO_NAME}/${DOCKER_IMAGE}:latest"

# Deploy to Cloud Run
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