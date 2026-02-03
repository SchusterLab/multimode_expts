#!/bin/bash
# Script to start the job worker

# Change to the multimode_expts directory
cd "$(dirname "$0")/.."

echo "Starting job worker..."
echo "Press Ctrl+C to cancel current job (worker continues), or Ctrl+C while idle to stop"
echo ""

# Start the worker with unbuffered output (runs in foreground)
pixi run python -u -m job_server.worker
