#!/bin/bash

# Load environment variables
set -a
source .env
set +a

IMAGE_NAME=nesy
CONTAINER_NAME=nesy_container
MODEL_CACHE_DIR=./models
DATA_DIR=./data

# --------------------------------------------------
# Remove existing container if it exists
# --------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already exists. Removing it..."
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
fi

# --------------------------------------------------
# GPU is optional — this setup is CPU-only by design,
# but falls back to using a GPU if one is visible and
# DEVICE is set in .env, in case someone runs it on a
# machine that has one.
# --------------------------------------------------
GPU_FLAG=""
if [ -n "$DEVICE" ] && command -v nvidia-smi &> /dev/null; then
    echo "GPU detected and DEVICE=$DEVICE set — enabling GPU passthrough."
    GPU_FLAG="--gpus device=$DEVICE"
else
    echo "Running CPU-only (no DEVICE set or no GPU available)."
fi

# --------------------------------------------------
# Run container
# --------------------------------------------------
docker run --rm \
  $GPU_FLAG \
  --name $CONTAINER_NAME \
  --memory=16g \
  --network host \
  -v ${MODEL_CACHE_DIR}:/workdir/models \
  -v ${DATA_DIR}:/workdir/data \
  -v ./scripts:/workdir/scripts \
  -v ./src:/workdir/src \
  --env-file .env \
  -it $IMAGE_NAME:latest