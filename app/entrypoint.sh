#!/bin/bash
# Start Ollama server in the background
ollama start &

# Wait a few seconds for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 5

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
