# NagrikHelp AI - Render Deployment (Lightweight ResNet-50)
FROM python:3.11-slim

WORKDIR /app

# Copy and install ultra-light dependencies
COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-render.txt

# Copy application code
COPY render_server.py .

# Environment variables
ENV CONFIDENCE_THRESHOLD=0.5
ENV PORT=8001

# Expose port
EXPOSE 8001

# Run the lightweight server
CMD ["uvicorn", "render_server:app", "--host", "0.0.0.0", "--port", "8001"]
