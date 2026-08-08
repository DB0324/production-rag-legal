#!/bin/bash
# Wrapper: sources .env (LLM_PROVIDER/LLM_MODEL) before launching uvicorn,
# so the API server never accidentally starts with the wrong/missing
# environment variables (which silently defaults to Gemini and crashes
# on every generation call).

set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env not found, using shell defaults"
fi

echo "Starting API with LLM_PROVIDER=$LLM_PROVIDER, LLM_MODEL=$LLM_MODEL"

# Kill any existing uvicorn instance on port 8000 first
EXISTING_PID=$(pgrep -f "uvicorn src.api.main:app" || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Killing existing API process: $EXISTING_PID"
    kill -9 $EXISTING_PID
    sleep 2
fi

mkdir -p logs
nohup python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > logs/api_server.log 2>&1 &
disown
NEW_PID=$!
echo "API started, PID: $NEW_PID"
echo "Waiting for startup..."
sleep 15
curl -m 15 http://127.0.0.1:8000/health
