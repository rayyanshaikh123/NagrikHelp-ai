# NagrikHelp AI Servers

This directory contains **two separate AI servers** for different deployment scenarios:

## 1. Render Server (Production - Free Tier) 🚀

**File:** `render_server.py`  
**Requirements:** `requirements-render.txt`  
**Model:** `microsoft/resnet-50` via Hugging Face Inference API  
**Memory:** < 512MB (fits Render free tier)

### Features:
- Ultra-lightweight, no local ML models
- Uses Hugging Face Inference API for image classification
- Maps ImageNet labels to civic categories (POTHOLE, GARBAGE, STREETLIGHT, WATER)
- Health check endpoint at `GET /`
- Validation endpoint at `POST /validate`
- Legacy compatibility at `POST /classify`

### Deploy to Render:
```bash
# Dockerfile automatically uses render_server.py
# Environment variables needed:
HUGGINGFACE_API_TOKEN=hf_xxx  # Get from https://huggingface.co/settings/tokens
CONFIDENCE_THRESHOLD=0.5
```

---

## 2. Local Server (Development - Full Features) 💪

**File:** `local_vision_server.py`  
**Requirements:** `requirements-local.txt`  
**Models:** YOLOv8 + ResNet-50 + CLIP (multi-model pipeline)  
**Memory:** ~2-4GB

### Features:
- **YOLOv8**: Object detection with bounding boxes
- **ResNet-50**: Image classification (ImageNet labels)
- **CLIP**: Zero-shot classification for maximum accuracy
- Multi-model voting system
- Configurable model enablement via env vars

### Run Locally:
```bash
# Install dependencies
pip install -r requirements-local.txt

# Set environment variables (optional)
export ENABLE_YOLO=true
export ENABLE_CLIP=true
export CONFIDENCE_THRESHOLD=0.6
export HUGGINGFACE_HUB_TOKEN=hf_xxx

# Run server
uvicorn local_vision_server:app --host 0.0.0.0 --port 8001 --reload
```

### Configure Models:
```bash
# Enable/disable specific models
export ENABLE_YOLO=true   # Object detection (heavier)
export ENABLE_CLIP=true   # Zero-shot (best accuracy, heavier)

# Both disabled = ResNet-50 only (lightweight)
export ENABLE_YOLO=false
export ENABLE_CLIP=false
```

---

## API Endpoints (Both Servers)

### Health Check
```bash
GET /
```
Returns service info, model configuration

### Validate Issue (Primary)
```bash
POST /validate
Content-Type: application/json

{
  "image": "data:image/png;base64,iVBORw0KG...",  # Base64 image
  "description": "streetlight not working"        # Optional hint
}
```

**Response:**
```json
{
  "isIssue": true,
  "category": "STREETLIGHT",
  "confidence": 0.75,
  "bbox": [x1, y1, x2, y2],  # YOLO only
  "modelUsed": "resnet-50",
  "message": "Detected STREETLIGHT issue...",
  "latencyMs": 850,
  "rawLabels": [...],
  "debug": {...}
}
```

### Classify (Legacy)
```bash
POST /classify
Content-Type: application/octet-stream
(raw image bytes)
```

---

## Quick Comparison

| Feature | Render Server | Local Server |
|---------|---------------|--------------|
| **Memory** | < 512MB | ~2-4GB |
| **Speed** | 500-1000ms | 300-800ms |
| **Accuracy** | Good (70-80%) | Excellent (85-95%) |
| **Models** | ResNet-50 (API) | YOLO + ResNet + CLIP |
| **Cost** | Free (HF API) | Local compute |
| **Setup** | Zero install | pip install heavy deps |

---

## Testing

```bash
# Test Render deployment
curl -X POST https://nagrikhelp-ai.onrender.com/validate \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/png;base64,...","description":"pothole"}'

# Test local server
curl -X POST http://localhost:8001/validate \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/png;base64,...","description":"garbage"}'
```

---

## Files Overview

```
ai/
├── render_server.py           # Lightweight for Render
├── local_vision_server.py     # Full-featured for local
├── requirements-render.txt    # Minimal deps (5 packages)
├── requirements-local.txt     # Full deps (torch, YOLO, etc.)
├── Dockerfile                 # Points to render_server.py
├── .env.local                 # Local dev configuration
└── README.md                  # This file
```
