"""
NagrikHelp AI Validation (Render Free Tier)
Lightweight server using microsoft/resnet-50 via HF Inference API
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
logger = logging.getLogger("render-server")

# Config
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
HF_MODEL = os.getenv("HF_MODEL", "microsoft/resnet-50")
# Use HF Inference API (correct endpoint)
HF_API_URL = os.getenv("HF_API_URL", f"https://api-inference.huggingface.co/models/{HF_MODEL}")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT_MS", "30000")) // 1000 or 30

# Category keyword mapping for ResNet ImageNet labels
CATEGORY_KEYWORDS: Dict[str, list[str]] = {
    "POTHOLE": ["pothole", "hole", "crack", "asphalt", "pavement", "road", "crater", "depression", "cavity", "damaged"],
    "GARBAGE": ["garbage", "trash", "waste", "litter", "dump", "rubbish", "plastic", "bag", "dumpster", "bin", "debris"],
    "STREETLIGHT": ["street light", "lamp", "pole", "light", "streetlight", "lighting", "bulb", "lantern", "lamppost"],
    "WATER": ["water", "leak", "pipe", "flood", "puddle", "drain", "geyser", "fountain", "hose", "waterlog"],
    "ELECTRICITY": ["wire", "electric", "electricity", "cable", "power line", "transformer", "pole", "exposed"],
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
    """Parse base64 image data"""
    if data_uri.startswith("data:image"):
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def classify_image_hf(image: Image.Image, retry_count: int = 0) -> list:
    """Call HF Inference API for image classification"""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        logger.info(f"Calling HF API: {HF_API_URL} (attempt {retry_count + 1})")
        resp = requests.post(HF_API_URL, data=buf.getvalue(), headers=headers, timeout=HF_TIMEOUT)
        
        logger.info(f"HF API response: {resp.status_code}")
        
        if resp.status_code == 200:
            result = resp.json()
            # Handle both list and dict responses
            if isinstance(result, list):
                logger.info(f"HF API success: {len(result)} predictions")
                return result
            elif isinstance(result, dict) and "error" in result:
                logger.error(f"HF API returned error dict: {result}")
                # Model might be loading, retry once
                if retry_count < 1 and "loading" in str(result.get('error', '')).lower():
                    logger.info("Model loading, waiting 10s before retry...")
                    time.sleep(10)
                    return classify_image_hf(image, retry_count + 1)
                raise HTTPException(status_code=502, detail=result['error'])
            logger.warning(f"Unexpected result format: {type(result)}")
            return result
        elif resp.status_code == 503:
            # Model loading, retry once
            if retry_count < 1:
                logger.info("Model loading (503), waiting 15s before retry...")
                time.sleep(15)
                return classify_image_hf(image, retry_count + 1)
            raise HTTPException(status_code=503, detail="Model loading. Please retry in 20-30 seconds.")
        elif resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        elif resp.status_code in [404, 410]:
            error_text = resp.text[:500]
            logger.error(f"HF API endpoint error ({resp.status_code}): {error_text}")
            logger.error(f"Attempted URL: {HF_API_URL}")
            raise HTTPException(status_code=502, detail=f"Inference API unavailable ({resp.status_code}). Endpoint may have changed.")
        else:
            error_text = resp.text[:500]
            logger.error(f"HF API error {resp.status_code}: {error_text}")
            raise HTTPException(status_code=502, detail=f"HF API error {resp.status_code}")
            
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="HF API timeout")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("HF API call failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


def map_labels_to_category(labels: list, description: str = "") -> Dict[str, Any]:
    """Map ImageNet labels to civic issue categories"""
    scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORY_KEYWORDS.keys()}
    scores["OTHER"] = 0.0
    
    # Score based on label matching with weighted importance
    for idx, item in enumerate(labels[:10]):  # Top 10 labels
        label = item["label"].lower()
        score = item["score"]
        
        # Weight: top predictions matter more
        weight = 1.0 if idx < 3 else 0.7
        
        matched = False
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in label:
                    scores[cat] += score * weight * 1.2  # 1.2x boost for direct matches
                    matched = True
                    break
            if matched:
                break
        
        # Civic-related visual terms boost relevant categories
        if not matched:
            civic_visual = {
                "outdoor": ["POTHOLE", "GARBAGE", "STREETLIGHT", "WATER"],
                "street": ["POTHOLE", "STREETLIGHT"],
                "road": ["POTHOLE"],
                "pavement": ["POTHOLE"],
                "ground": ["POTHOLE", "GARBAGE"],
                "urban": ["POTHOLE", "GARBAGE", "STREETLIGHT"],
                "container": ["GARBAGE"],
                "bucket": ["GARBAGE", "WATER"],
            }
            for term, cats in civic_visual.items():
                if term in label:
                    for cat in cats:
                        if cat in scores:
                            scores[cat] += score * 0.4
    
    # Boost from description
    if description:
        desc_lower = description.lower()
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    scores[cat] += 0.4
    
    # Find best category
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]
    
    # Normalize confidence (cap at 0.95)
    confidence = min(best_score, 0.95)
    
    # More lenient validation: if ANY civic category has decent score, consider it valid
    any_civic_score = max(scores[cat] for cat in CATEGORY_KEYWORDS.keys())
    is_issue = confidence >= CONFIDENCE_THRESHOLD or any_civic_score >= CONFIDENCE_THRESHOLD
    
    # Generate reasoning
    top_labels = [f"{l['label']} ({l['score']:.2f})" for l in labels[:3]]
    reasoning = f"Top predictions: {', '.join(top_labels)}. "
    if is_issue:
        reasoning += f"Detected {best_cat} issue with {confidence:.1%} confidence."
    else:
        reasoning += "Low confidence - no clear civic issue detected."
    
    return {
        "isIssue": is_issue,
        "category": best_cat,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
        "rawLabels": labels[:5]
    }


@app.get("/")
@app.head("/")
def root():
    return {
        "ok": True,
        "service": "NagrikHelp AI Validation (Render)",
        "model": HF_MODEL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "deployment": "render-free-tier"
    }


@app.post("/validate")
async def validate_issue(req: Request):
    """Primary validation endpoint"""
    t0 = time.time()
    body = await req.json()
    
    img_data = body.get("image")
    if not img_data:
        raise HTTPException(status_code=400, detail="'image' field required")
    
    description = body.get("description", "")
    
    # Parse and classify
    image = parse_image(img_data)
    labels = classify_image_hf(image)
    result = map_labels_to_category(labels, description)
    
    return JSONResponse({
        "isIssue": result["isIssue"],
        "category": result["category"],
        "confidence": result["confidence"],
        "bbox": None,
        "modelUsed": "resnet-50",
        "message": result["reasoning"],
        "latencyMs": int((time.time() - t0) * 1000),
        "rawLabels": result["rawLabels"],
        "debug": {"top_labels": [l["label"] for l in labels[:3]]}
    })


@app.post("/classify")
async def classify_legacy(req: Request):
    """Legacy endpoint for backward compatibility"""
    body = await req.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    
    # Convert raw bytes to base64
    b64 = base64.b64encode(body).decode("utf-8")
    img = parse_image(f"data:image/png;base64,{b64}")
    
    labels = classify_image_hf(img)
    result = map_labels_to_category(labels, "")
    
    return JSONResponse([{
        "label": result["category"],
        "score": result["confidence"]
    }])
