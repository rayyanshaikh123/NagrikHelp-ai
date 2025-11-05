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
# Minimal dependencies - NO OpenCV libs needed (using lightweight build)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# Use LIGHTWEIGHT requirements (no YOLO, no OpenCV)
COPY requirements-light.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-light.txt

# -----------------------------
# Copy the application code
# -----------------------------
COPY . .

# -----------------------------
# Environment variables
# -----------------------------
ENV LOCAL_VISION_HOST=0.0.0.0 \
    LOCAL_VISION_PORT=8001 \
    ENABLE_YOLO=false \
    CONFIDENCE_THRESHOLD=0.45

# -----------------------------
# Expose the app port
# -----------------------------
EXPOSE 8001

# -----------------------------
# Start the FastAPI app
# -----------------------------
CMD ["uvicorn", "local_vision_server:app", "--host", "0.0.0.0", "--port", "8001"]
