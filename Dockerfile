# Dockerfile
# Builds the Sound Limiter RL Environment for Hugging Face Spaces
# Run locally:
#   docker build -t sound-limiter-env .
#   docker run -p 7860:7860 -e HF_TOKEN=... -e MODEL_NAME=... -e API_BASE_URL=... sound-limiter-env

FROM python:3.11-slim

# Non-interactive, no .pyc files, unbuffered stdout (critical for [STEP] log streaming)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create __init__ files if missing
RUN touch environment/__init__.py agent/__init__.py 2>/dev/null || true

# Expose HF Spaces port
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "server.py"]