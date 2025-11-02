"""
Local image classification server using Hugging Face Transformers (PyTorch).

Usage:
  1) Create & activate a virtual environment (recommended).
  2) pip install -r scripts/local_vision_requirements.txt
  3) python -m uvicorn scripts.local_vision_server:app --host 127.0.0.1 --port 8001

Env vars:
  MODEL_NAME   (default: google/vit-base-patch16-224)
  TOP_K        (default: 5)

API:
  GET  /           -> { ok: true, model: str }
  POST /classify   -> body: raw image bytes (image/jpeg|image/png)
                      resp: [{ label: str, score: float }]
"""

import io
import os
from typing import List

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("LOCAL_VISION_MODEL", os.getenv("CIVIC_VISION_MODEL", "google/vit-base-patch16-224")))
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI()

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()


@app.get("/")
def root():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/classify")
async def classify(request: Request):
    try:
        content_type = request.headers.get("content-type", "application/octet-stream")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty body")

        img = Image.open(io.BytesIO(body)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        topk_scores, topk_indices = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))

        id2label = model.config.id2label
        results: List[dict] = []
        for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
            label = id2label.get(int(idx), str(idx))
            results.append({"label": label, "score": float(score)})
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("LOCAL_VISION_HOST", "127.0.0.1")
    port = int(os.getenv("LOCAL_VISION_PORT", "8001"))
    # Run directly without module string to avoid import path issues
    uvicorn.run(app, host=host, port=port)
