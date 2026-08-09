#!/bin/bash
# Wrapper: sources .env before launching uvicorn, kills any stale instance,
# and POLLS /health until the server is actually ready instead of guessing
# with a fixed sleep -- this is the permanent fix for the repeated
# "connection timed out" issue caused by startup taking longer than
# whatever fixed wait time we guessed.

set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env not found, using shell defaults"
fi

echo "Starting API with LLM_PROVIDER=$LLM_PROVIDER, LLM_MODEL=$LLM_MODEL"

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

echo "Waiting for API to become ready (polling /health, up to 350s (extended due to observed 5+ min cold-start under shared HPC disk contention))..."
for i in $(seq 1 70); do
    if curl -s -m 3 http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "API is ready after ~$((i * 5))s"
        curl -m 10 http://127.0.0.1:8000/health
        echo ""
        exit 0
    fi
    sleep 5
done

echo "ERROR: API did not become ready within 350s. Check logs/api_server.log"
tail -30 logs/api_server.log
exit 1
