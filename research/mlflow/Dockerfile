FROM python:3.11-slim

WORKDIR /app

# Keep runtime lightweight and include DB + GCS deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "mlflow[extras]" \
        psycopg2-binary \
        google-cloud-storage

EXPOSE 5000
# MLflow will use env vars at runtime
CMD mlflow server \
  --backend-store-uri ${BACKEND_STORE_URI} \
  --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
  --host 0.0.0.0 --port 5000