#!/bin/bash
# Script to start the FastAPI job server

# Change to the multimode_expts directory
cd "$(dirname "$0")/.."

# Check if server is already running on port 8000
if netstat -ano | grep -q ":8000.*LISTENING"; then
    echo "Error: A server is already running on port 8000"
    echo "Kill it first with: netstat -ano | grep :8000 to find PID, then kill the process"
    exit 1
fi

# Start the server (runs in foreground)
echo "Starting FastAPI server..."
echo "Access at: http://127.0.0.1:8000"
echo "API docs: http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to stop"
echo ""

pixi run python -m uvicorn job_server.server:app --host 0.0.0.0 --port 8000
