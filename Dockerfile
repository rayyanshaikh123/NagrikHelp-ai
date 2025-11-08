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
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
COPY local_vision_requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r local_vision_requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


# -----------------------------
# Optional: install CPU-only PyTorch
# Uncomment if needed
# -----------------------------
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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
CMD ["uvicorn", "render_server:app", "--host", "0.0.0.0", "--port", "8001"]
