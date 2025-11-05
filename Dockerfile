# -----------------------------
# Base image
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Environment settings
# -----------------------------
# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -----------------------------
# Install system dependencies
# -----------------------------
# Install minimal dependencies for headless OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# Use production requirements with opencv-python-headless
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy the application code
# -----------------------------
COPY . .

# -----------------------------
# Environment variables
# -----------------------------
ENV LOCAL_VISION_HOST=0.0.0.0 \
    LOCAL_VISION_PORT=8001

# -----------------------------
# Expose the app port
# -----------------------------
EXPOSE 8001

# -----------------------------
# Start the FastAPI app
# -----------------------------
CMD ["uvicorn", "local_vision_server:app", "--host", "0.0.0.0", "--port", "8001"]
