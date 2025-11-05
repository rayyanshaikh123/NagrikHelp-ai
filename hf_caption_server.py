"""
NagrikHelp AI Validation (Ultra-light): uses Hugging Face Inference API
Model: nlpconnect/vit-gpt2-image-captioning (free, no local heavy deps)
"""
import io
import os
import time
import base64
import logging
from typing import Dict, Any

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-caption-server")

# Config
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
HF_MODEL = os.getenv("HF_MODEL", "nlpconnect/vit-gpt2-image-captioning")
# HF Serverless Inference API endpoint
HF_API_URL = os.getenv("HF_API_URL", f"https://api-inference.huggingface.co/models/{HF_MODEL}")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT_MS", "30000")) // 1000 or 30

# Category keywords
CATEGORY_KEYWORDS: Dict[str, list[str]] = {
    "POTHOLE": ["pothole", "hole", "crack", "damage", "road", "asphalt", "pavement"],
    "GARBAGE": ["garbage", "trash", "waste", "litter", "dump", "rubbish", "plastic", "bag"],
    "STREETLIGHT": ["light", "lamp", "pole", "streetlight", "lighting", "bulb"],
    "WATER": ["water", "leak", "pipe", "flood", "puddle", "wet", "drain"],
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_image(data_uri: str) -> Image.Image:
    if data_uri.startswith("data:image"):
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def hf_caption(image: Image.Image) -> str:
    # Encode to PNG bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    headers = {"Accept": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        resp = requests.post(HF_API_URL, data=buf.getvalue(), headers=headers, timeout=HF_TIMEOUT)
        if resp.status_code == 200:
            arr = resp.json()
            if isinstance(arr, list) and arr and "generated_text" in arr[0]:
                return str(arr[0]["generated_text"]).strip()
            # Some models return dict
            if isinstance(arr, dict) and "generated_text" in arr:
                return str(arr["generated_text"]).strip()
            raise HTTPException(status_code=502, detail=f"Unexpected HF response: {arr}")
        elif resp.status_code in (503, 524):
            raise HTTPException(status_code=503, detail="HF model loading/warming up. Try again.")
        else:
            raise HTTPException(status_code=502, detail=f"HF API error {resp.status_code}: {resp.text[:140]}")
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="HF API timeout")
    except Exception as e:
        logger.exception("HF API call failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


def score_caption(caption: str, description: str = "") -> Dict[str, Any]:
    cap = caption.lower()
    scores: Dict[str, float] = {}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        s = sum(1 for k in keywords if k in cap)
        if s > 0:
            scores[cat] = s / len(keywords)

    if description:
        desc = description.lower()
        for cat, keywords in CATEGORY_KEYWORDS.items():
            s = sum(1 for k in keywords if k in desc)
            if s > 0:
                scores[cat] = scores.get(cat, 0) + (s / len(keywords)) * 0.5

    if scores:
        best_cat, best_score = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_score * 1.5, 0.95)
        is_issue = confidence >= CONFIDENCE_THRESHOLD
    else:
        # Fallback
        civic_indicators = ["outdoor", "street", "road", "building", "city", "urban", "infrastructure"]
        has_civic = any(w in cap for w in civic_indicators)
        best_cat = "OTHER"
        confidence = 0.4 if has_civic else 0.2
        is_issue = has_civic

    reasoning = f"Image caption: {caption}. " + (f"Detected {best_cat} indicators." if is_issue else "No clear civic issue detected.")

    return {
        "isIssue": is_issue,
        "category": best_cat,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
    }


@app.get("/")
@app.head("/")
def root():
    return {
        "ok": True,
        "service": "NagrikHelp AI Validation (HF Inference)",
        "model": HF_MODEL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "uses_hf_api": True,
        "hf_api_url": HF_API_URL,
    }


@app.post("/validate")
async def validate_issue(req: Request):
    t0 = time.time()
    body = await req.json()
    img_data = body.get("image")
    if not img_data:
        raise HTTPException(status_code=400, detail="'image' field required")
    description = body.get("description", "")

    image = parse_image(img_data)
    caption = hf_caption(image)
    result = score_caption(caption, description)

    return JSONResponse({
        "isIssue": result["isIssue"],
        "category": result["category"],
        "confidence": result["confidence"],
        "bbox": None,
        "modelUsed": "HF-vit-gpt2",
        "message": result["reasoning"],
        "latencyMs": int((time.time() - t0) * 1000),
        "rawLabels": [{"label": caption, "score": result["confidence"]}],
        "debug": {"caption": caption}
    })


@app.post("/classify")
async def classify_legacy(req: Request):
    body = await req.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    # turn bytes into data uri
    b64 = base64.b64encode(body).decode("utf-8")
    img = parse_image(f"data:image/png;base64,{b64}")
    caption = hf_caption(img)
    result = score_caption(caption, "")
    return JSONResponse([{ "label": result["category"], "score": result["confidence"] }])
