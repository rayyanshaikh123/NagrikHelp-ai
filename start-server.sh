#!/bin/bash
# Startup script for NagrikHelp AI server with proper configuration

cd "$(dirname "$0")"

# Check if port 8001 is already in use and kill it
if lsof -ti:8001 > /dev/null 2>&1; then
    echo "⚠️  Port 8001 is already in use. Stopping existing process..."
    lsof -ti:8001 | xargs kill -9 2>/dev/null
    sleep 1
    echo "✓ Existing process stopped"
fi

# Load environment variables from .env.local (filter comments and empty lines)
if [ -f .env.local ]; then
    export $(grep -v '^#' .env.local | grep -v '^$' | sed 's/#.*//' | xargs)
fi

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r local_vision_requirements.txt"
    exit 1
fi

echo "🚀 Starting NagrikHelp AI Validation Server"
echo "   Confidence Threshold: ${CONFIDENCE_THRESHOLD:-0.6}"
echo "   YOLO Detection: ${ENABLE_YOLO:-true}"
echo "   CLIP Zero-shot: ${ENABLE_CLIP:-true}"
echo "   Port: ${LOCAL_VISION_PORT:-8001}"
echo ""

# Start the server
python -m uvicorn local_vision_server:app \
    --host ${LOCAL_VISION_HOST:-0.0.0.0} \
    --port ${LOCAL_VISION_PORT:-8001}
