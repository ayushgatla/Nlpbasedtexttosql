#!/usr/bin/env bash
set -e

echo "==================================================="
echo "Starting NeuroSQL Studio via Docker Compose"
echo "==================================================="

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Building and launching containers in detached mode..."
docker compose up --build -d

echo ""
echo "[SUCCESS] NeuroSQL Studio is running!"
echo "Access the visual studio at: http://localhost:8501"
echo "To view logs, run: docker compose logs -f"
echo "To stop the container, run: docker compose down"
