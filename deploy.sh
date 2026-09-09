#!/bin/bash

echo "🚀 Starting Deployment for AI Marketing Material Library..."

# 1. Ensure Files Exist to prevent Docker directory creation issues
# (Docker creates a directory if the mount source file doesn't exist)
[ ! -f users.json ] && echo "[]" > users.json
[ ! -f known_ids.json ] && echo "[]" > known_ids.json
[ ! -f simulation_results.csv ] && touch simulation_results.csv

mkdir -p AI素材库
mkdir -p local_qdrant_db
mkdir -p pending_ingest

# 2. Check for .env
if [ ! -f .env ]; then
    echo "⚠️  .env file not found! Please create one with your API keys."
    # Optional: Create a dummy one if needed, but better to warn
else
    echo "✅ .env file found."
fi

# 3. Build and Run
echo "🐳 Building and starting containers..."
# Check if docker-compose exists, otherwise try docker compose
if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
else
    docker compose up -d --build
fi

echo "✅ Deployment Complete!"
echo "👉 Access the app at: http://localhost:8501"
