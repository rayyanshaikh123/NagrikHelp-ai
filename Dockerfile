FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files to disk and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY local_vision_requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r local_vision_requirements.txt

# Optional: If you want CPU-only PyTorch, uncomment the following line and rebuild.
# It's commented out because many projects pin torch to specific wheels which can be large.
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . /app

# Default environment values; override at `docker run` time if needed
ENV LOCAL_VISION_HOST=0.0.0.0
ENV LOCAL_VISION_PORT=8001

EXPOSE 8001

# Use uvicorn to serve the FastAPI app. This expects `local_vision_server:app` to be importable.
CMD ["uvicorn", "local_vision_server:app", "--host", "0.0.0.0", "--port", "8001"]
