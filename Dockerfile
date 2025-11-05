# NagrikHelp AI - BLIP Vision Model
FROM python:3.11-slim

WORKDIR /app

# Copy and install ultra-light dependencies (no torch)
COPY requirements-hf.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-hf.txt

# Copy application code
COPY hf_caption_server.py .

# Environment variables
ENV CONFIDENCE_THRESHOLD=0.5
ENV PORT=8001

# Expose port
EXPOSE 8001

# Run the server
CMD ["uvicorn", "hf_caption_server:app", "--host", "0.0.0.0", "--port", "8001"]
