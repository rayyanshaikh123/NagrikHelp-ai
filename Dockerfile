# NagrikHelp AI - BLIP Vision Model
FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements-gemini.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-gemini.txt

# Copy application code
COPY gemini_vision_server.py .

# Environment variables
ENV CONFIDENCE_THRESHOLD=0.5
ENV PORT=8001

# Expose port
EXPOSE 8001

# Run the server
CMD ["uvicorn", "gemini_vision_server:app", "--host", "0.0.0.0", "--port", "8001"]
