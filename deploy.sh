#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ID:-}" || -z "${REGION:-}" ]]; then
  echo "ERROR: You must export PROJECT_ID and REGION before running this script."
  echo "Example:"
  echo "  export PROJECT_ID=my-gcp-project"
  echo "  export REGION=us-central1"
  exit 1
fi

SERVICE_NAME="${SERVICE_NAME:-financial-intel-demo}"

echo "Deploying service '${SERVICE_NAME}' to project '${PROJECT_ID}' in region '${REGION}'..."

gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --set-env-vars "GCP_REGION=${REGION}"

echo "Deployment initiated. Once complete, you can invoke the service with:"
echo "  curl -X POST \"\$(gcloud run services describe ${SERVICE_NAME} --project ${PROJECT_ID} --region ${REGION} --format='value(status.url)')/invoke\" \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"ticker\": \"PYPL\"}'"

