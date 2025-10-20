# Stage 1: Build stage
FROM python:3.11.9-slim AS builder
WORKDIR /app
COPY requirements.txt .
# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app
# Expose the FastAPI default port
EXPOSE 8000
# Command to start the server
CMD [ "uvicorn", "test:app", "--host", "0.0.0.0", "--port", "8000","--workers", "1"]


# COPY /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages # buildkit

