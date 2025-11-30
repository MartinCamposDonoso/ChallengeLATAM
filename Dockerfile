# syntax=docker/dockerfile:1.2
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies if needed for opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
COPY requirements-test.txt requirements-test.txt

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-test.txt

# Copy application code
COPY challenge/ ./challenge/
COPY tests/ ./tests/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]