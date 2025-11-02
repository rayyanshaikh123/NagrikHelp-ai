"""
Local image classification server using Hugging Face Transformers (PyTorch).

Usage:
    1) Create & activate a virtual environment (recommended).
    2) pip install -r scripts/local_vision_requirements.txt
    3) python -m uvicorn scripts.local_vision_server:app --host 127.0.0.1 --port 8001

Env vars:
    MODEL_NAME   (default: google/vit-base-patch16-224)
    TOP_K        (default: 5)

Recommended public model names (examples):
    - google/vit-base-patch16-224
    - google/vit-large-patch16-224
    - microsoft/resnet-50
    - facebook/convnext-base-224

API:
    GET  /           -> { ok: True, model: str }
    POST /classify   -> body: raw image bytes (image/jpeg|image/png)
                                            resp: [{ label: str, score: float }]
"""

import io
import os

login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Example public models for quick reference inside the module
SUPPORTED_MODELS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "microsoft/resnet-50",
    "facebook/convnext-base-224",
]

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("LOCAL_VISION_MODEL", os.getenv("CIVIC_VISION_MODEL", "google/vit-base-patch16-224")))
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI()

# Lazy-loaded model/processor. This avoids heavy model downloads at import time so CI
# and quick smoke tests can import the app without pulling large HF weights.
processor = None
model = None


def load_model():
    """Load the HF image processor and model into module globals if not already loaded.

    This function imports the transformers classes locally so importing this module
    doesn't trigger large downloads.
    """
    global processor, model
    if model is not None and processor is not None:
        return
    # Import inside the function to avoid module-level side-effects during import
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except Exception as e:
        logger.exception("Failed to import transformers: %s", e)
        raise

    try:
        logger.info("Loading model %s ...", MODEL_NAME)
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        model.eval()
        logger.info("Model %s loaded successfully", MODEL_NAME)
    except Exception as e:
        # Provide a clearer error message for runtime issues (torch/numpy)
        logger.exception("Failed to load model %s: %s", MODEL_NAME, e)
        raise RuntimeError(f"Failed to load model {MODEL_NAME}: {e}") from e


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
        # Ensure the model is loaded on first use
        try:
            load_model()
        except Exception as e:
            # Surface a 503 so orchestration can retry if startup/load fails
            logger.exception("Model load error: %s", e)
            raise HTTPException(status_code=503, detail=str(e))

        # Import torch lazily so the module can be imported in CI or in a
        # lightweight environment without having to install torch/tokenizers.
        try:
            import torch
        except Exception as e:
            logger.exception("Torch import failed: %s", e)
            raise HTTPException(status_code=503, detail=f"Torch not available: {e}")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in classify: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-status")
def model_status():
    """Return whether the model is loaded and basic info.

    Useful for health checks after deployment.
    """
    loaded = (model is not None and processor is not None)
    return {"loaded": loaded, "model": MODEL_NAME if loaded else None}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("LOCAL_VISION_HOST", "127.0.0.1")
    port = int(os.getenv("LOCAL_VISION_PORT", "8001"))
    # Run directly without module string to avoid import path issues
    uvicorn.run(app, host=host, port=port)
