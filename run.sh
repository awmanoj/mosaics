#!/usr/bin/env bash
#

set -euo pipefail

# --- Config -----------------------------------------------------------------
DOCKERHUB_USER="${DOCKERHUB_USER:?Set DOCKERHUB_USER to your Docker Hub username}"
IMAGE_NAME="${IMAGE_NAME:-mosaics}"
TAG="${1:-latest}"

REPO="${DOCKERHUB_USER}/${IMAGE_NAME}"
CONTAINER_NAME="${CONTAINER_NAME:-mosaics}"
PORT=8765

echo ">> Pulling ${REPO}:${TAG}"
docker pull "${REPO}:${TAG}"

# --- Replace any existing container -----------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo ">> Removing existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# --- Run --------------------------------------------------------------------
# APP_PORT makes uvicorn (and the image's healthcheck) bind to ${PORT} inside
# the container; -p publishes that same port on the host.
echo ">> Starting ${CONTAINER_NAME} on port ${PORT}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  -e "APP_PORT=${PORT}" \
  -e "GA_MEASUREMENT_ID=${GA_MEASUREMENT_ID:-}" \
  -p "${PORT}:${PORT}" \
  "${REPO}:${TAG}"

echo ">> Running. Available at http://<server-ip>:${PORT}/"
echo ">> Logs: docker logs -f ${CONTAINER_NAME}"
