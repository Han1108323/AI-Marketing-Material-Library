#!/bin/bash

echo "🚀 Starting Setup..."

# 1. Check if python3 exists
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 could not be found."
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate and Install
echo "⬇️  Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "🎉 Setup Complete!"
echo "👉 Run 'source .venv/bin/activate' to start working."
