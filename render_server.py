#!/usr/bin/env python3
"""
Render production server using Hugging Face Transformers (PyTorch)
with optional Gemini/text-model based civic confidence scoring.

Endpoints:
  GET  /                -> { ok: True, model: str }
  HEAD /                -> health check
  GET  /model-status    -> {"loaded": bool, "model": str|None}
  GET  /gemini-status   -> {"reachable": bool, "last_error": str|null}
  POST /classify        -> send image bytes (raw body) OR multipart form 'file'
                           returns predictions + civic_confidence_pct + top_label
  POST /validate        -> accepts {image: base64} for backward compatibility
"""

import io
import os
import json
import logging
import re
import base64
import time
from typing import List, Optional

import urllib.request
import urllib.error

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("render-server")

# -------- Configuration (via env vars) --------
MODEL_NAME = os.getenv("MODEL_NAME", "google/vit-base-patch16-224")
TOP_K = int(os.getenv("TOP_K", "5"))
USE_TEXT_MODEL = os.getenv("USE_TEXT_MODEL", "0").lower() in ("1", "true", "yes")
TEXT_MODEL_PROVIDER = os.getenv("TEXT_MODEL_PROVIDER", "hf")  # 'hf' or 'gemini'
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "facebook/bart-large-mnli")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-mini")
CIVIC_CONFIDENCE_THRESHOLD = float(os.getenv("CIVIC_CONFIDENCE_THRESHOLD", "30"))
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# Example labels/keywords for civic issue detection
CIVIC_KEYWORDS = [
    "pothole", "potholes", "graffiti", "garbage", "trash", "litter", "dump", "dumping",
    "illegal", "flood", "flooding", "standing water", "sewage", "overflow",
    "blocked", "blocked drain", "sinkhole", "road", "street", "traffic light",
    "streetlight", "lamp post", "sign", "broken", "damaged", "collapsed",
    "debris", "fallen", "fire", "smoke", "accident", "vandalism", "construction",
    "hole", "crack", "leak", "spill",
]

app = FastAPI(title="NagrikHelp AI Validation (Render)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded
processor = None
model = None

# For tracking last Gemini health error
_gemini_last_error: Optional[str] = None


# ------------------- Model loading -------------------
def _try_from_pretrained(name: str, token: Optional[str] = None):
    """Central place to call from_pretrained with optional token."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    kwargs = {}
    if token:
        kwargs["token"] = token
    logger.info("Calling AutoImageProcessor.from_pretrained(%s)", name)
    proc = AutoImageProcessor.from_pretrained(name, **kwargs)
    logger.info("Calling AutoModelForImageClassification.from_pretrained(%s)", name)
    mod = AutoModelForImageClassification.from_pretrained(name, **kwargs)
    return proc, mod


def load_model():
    """Load model and processor into globals. Raises on failure with helpful message."""
    global processor, model
    if processor is not None and model is not None:
        return

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: F401
    except Exception as e:
        logger.exception("transformers import failed: %s", e)
        raise RuntimeError(f"transformers import failed: {e}")

    # Attempt load; try given name, then attempt auto-fix replacement '-' -> '/'
    first_exc = None
    try:
        logger.info("Loading model %s ...", MODEL_NAME)
        processor, model = _try_from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model.eval()
        logger.info("Model %s loaded successfully", MODEL_NAME)
        return
    except Exception as e:
        first_exc = e
        logger.debug("Initial load failed: %s", e)

    # try auto-fix if looks like owner-name instead of owner/name
    alt_name = None
    if "/" not in MODEL_NAME and "-" in MODEL_NAME:
        alt_name = MODEL_NAME.replace("-", "/", 1)
        try:
            logger.info("Trying alternative model id: %s", alt_name)
            processor, model = _try_from_pretrained(alt_name, token=HF_TOKEN)
            model.eval()
            logger.info("Model loaded successfully as %s", alt_name)
            return
        except Exception as e:
            logger.debug("Alt load failed: %s", e)

    # Build diagnostics
    guidance = (
        "Model identifier not found or access denied. If this is a private model, set HUGGINGFACE_HUB_TOKEN "
        "in the environment. Also verify MODEL_NAME uses owner/repo format (e.g., 'microsoft/resnet-50')."
    )
    detail_msg = f"initial_error: {first_exc!r}"
    logger.exception("Failed to load model %s: %s. %s", MODEL_NAME, detail_msg, guidance)
    # Raise an informative RuntimeError
    raise RuntimeError(f"Failed to load model {MODEL_NAME}: {detail_msg}. {guidance}")


# ------------------- Helpers -------------------
def is_civic_issue(labels: List[str]) -> bool:
    if not labels:
        return False
    lowered = [l.lower() for l in labels if isinstance(l, str)]
    for kw in CIVIC_KEYWORDS:
        for lab in lowered:
            if kw in lab:
                return True
    return False


def _fallback_confidence(results_list: List[dict]) -> float:
    if not results_list:
        return 0.0
    return max(r.get("score", 0.0) for r in results_list) * 100.0


# ------------------- Gemini integration -------------------
def call_gemini_confidence(labels: List[str], required_labels: List[str]) -> float:
    """
    Ask Gemini to return a numeric civic_confidence (0-100). Returns 0.0 on any failure.
    """
    global _gemini_last_error
    if not GEMINI_API_KEY:
        _gemini_last_error = "GEMINI_API_KEY not set"
        logger.debug(_gemini_last_error)
        return 0.0

    labels_text = "; ".join(labels) if labels else ""
    required_text = ", ".join(required_labels[:30])

    prompt_text = (
        "You are a strict JSON-only classifier.\n"
        "Given the following detected image labels: '" + labels_text + "'.\n"
        "Also consider these civic-related labels for reference: '" + required_text + "'.\n"
        "Return ONLY a JSON object with a single numeric field 'civic_confidence' whose value is a number between 0 and 100 representing the percent likelihood that the detected labels indicate a civic/municipal issue (pothole, flooding, graffiti, illegal dumping, broken streetlight, etc.).\n"
        "Do NOT include any extra text."
    )

    endpoint = f"https://generativelanguage.googleapis.com/v1beta2/{GEMINI_MODEL}:generate?key={GEMINI_API_KEY}"
    body = {"prompt": {"text": prompt_text}, "temperature": 0.0, "max_output_tokens": 128}

    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_text = resp.read().decode("utf-8")
            logger.info("Gemini raw response: %s", resp_text)
    except urllib.error.HTTPError as he:
        _gemini_last_error = f"HTTPError: {he}"
        logger.exception("Gemini HTTP error: %s", he)
        return 0.0
    except Exception as ex:
        _gemini_last_error = f"Request failed: {ex}"
        logger.exception("Gemini request failed: %s", ex)
        return 0.0

    # Try parsing JSON first
    try:
        parsed = json.loads(resp_text)
    except Exception:
        parsed = None

    cand_text = ""
    if isinstance(parsed, dict):
        if "candidates" in parsed and parsed["candidates"]:
            first = parsed["candidates"][0]
            if isinstance(first, dict):
                cand_text = first.get("content") or first.get("output") or json.dumps(first)
            else:
                cand_text = str(first)
        elif "output" in parsed and parsed["output"]:
            first = parsed["output"][0]
            if isinstance(first, dict):
                cand_text = first.get("content") or json.dumps(first)
            else:
                cand_text = str(first)
        elif "civic_confidence" in parsed:
            cand_text = str(parsed["civic_confidence"])
        else:
            cand_text = json.dumps(parsed)
    else:
        cand_text = resp_text

    # Extract first numeric (0-100)
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", cand_text)
    if m:
        try:
            val = float(m.group(1))
            val = max(0.0, min(100.0, val))
            _gemini_last_error = None
            return val
        except Exception as ex:
            _gemini_last_error = f"parse error: {ex}"
            logger.exception("Gemini parse error: %s", ex)
            return 0.0

    _gemini_last_error = "no numeric found in response"
    logger.warning("Gemini response did not contain a numeric confidence: %s", cand_text)
    return 0.0


def parse_image(data_uri: str) -> Image.Image:
    """Parse base64 image data"""
    if data_uri.startswith("data:image"):
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ------------------- FastAPI endpoints -------------------
@app.get("/")
@app.head("/")
def root():
    return {"ok": True, "service": "NagrikHelp AI Validation (Render)", "model": MODEL_NAME}


@app.get("/model-status")
def model_status():
    loaded = model is not None and processor is not None
    return {"loaded": loaded, "model": MODEL_NAME if loaded else None}


@app.get("/gemini-status")
def gemini_status():
    reachable = _gemini_last_error is None if GEMINI_API_KEY else False
    return {"reachable": reachable, "last_error": _gemini_last_error}


@app.post("/validate")
async def validate_issue(req: Request):
    """Primary validation endpoint - accepts {image: base64}"""
    t0 = time.time()
    body = await req.json()
    
    img_data = body.get("image")
    if not img_data:
        raise HTTPException(status_code=400, detail="'image' field required")
    
    description = body.get("description", "")
    
    # Parse image
    try:
        image = parse_image(img_data)
    except Exception as e:
        logger.exception("Failed to parse image: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    # Classify
    try:
        result = await classify_image(image, description)
        
        return JSONResponse({
            "isIssue": result.get("accepted", False),
            "category": result.get("category", "OTHER"),
            "confidence": result.get("civic_confidence_pct", 0.0) / 100.0,
            "bbox": None,
            "modelUsed": MODEL_NAME,
            "message": result.get("message", ""),
            "latencyMs": int((time.time() - t0) * 1000),
            "rawLabels": result.get("predictions", [])[:5],
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Validation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify(request: Request, file: Optional[UploadFile] = File(None)):
    """
    Accepts raw image bytes as body OR multipart/form-data with file field named 'file'.
    Returns JSON with predictions and civic_confidence_pct.
    """
    try:
        # 1) get image bytes
        body_bytes = b""
        try:
            if file is not None:
                body_bytes = await file.read()
            else:
                body_bytes = await request.body()
                if not body_bytes:
                    raise HTTPException(status_code=400, detail="No image data provided")
        except Exception as e:
            logger.exception("Failed to read image bytes: %s", e)
            raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

        # 2) open image
        try:
            img = Image.open(io.BytesIO(body_bytes)).convert("RGB")
        except Exception as e:
            logger.exception("Invalid image: %s", e)
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        
        result = await classify_image(img, "")
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in classify: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def classify_image(img: Image.Image, description: str = "") -> dict:
    """Core classification logic"""
    # 1) ensure model loaded
    try:
        load_model()
    except Exception as e:
        logger.exception("Model load error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    # 2) verify PyTorch present
    try:
        import torch
    except Exception as e:
        logger.exception("Torch import failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Torch not available: {e}")

    # 3) run processor/model
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model output did not contain logits")

    probs = torch.softmax(logits, dim=-1)[0]
    topk = min(TOP_K, probs.shape[-1])
    topk_scores, topk_indices = torch.topk(probs, k=topk)

    id2label = getattr(model.config, "id2label", None) or {}
    results: List[dict] = []
    for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
        label = id2label.get(int(idx), str(idx))
        results.append({"label": label, "score": float(score)})

    top_labels = [r["label"] for r in results]

    # 4) compute civic confidence
    civic_confidence_pct: Optional[float] = None
    if USE_TEXT_MODEL:
        try:
            if TEXT_MODEL_PROVIDER == "hf":
                try:
                    from transformers import pipeline
                except Exception as e:
                    logger.exception("HF text pipeline import failed: %s", e)
                    raise

                seq = "; ".join(top_labels) or ""
                clf = pipeline("zero-shot-classification", model=TEXT_MODEL_NAME, device=-1)
                candidate_labels = ["civic_issue", "not_civic_issue"]
                z = clf(seq, candidate_labels)
                if "labels" in z and "scores" in z:
                    if "civic_issue" in z["labels"]:
                        idx = z["labels"].index("civic_issue")
                        civic_confidence_pct = float(z["scores"][idx]) * 100.0
                    else:
                        civic_confidence_pct = float(z["scores"][0]) * 100.0
                else:
                    civic_confidence_pct = _fallback_confidence(results)
            elif TEXT_MODEL_PROVIDER == "gemini":
                civic_confidence_pct = call_gemini_confidence(top_labels, CIVIC_KEYWORDS)
            else:
                civic_confidence_pct = _fallback_confidence(results)
        except Exception as e:
            logger.exception("Text-model based confidence failed: %s", e)
            civic_confidence_pct = _fallback_confidence(results)
    else:
        civic_confidence_pct = _fallback_confidence(results)

    # 5) decide acceptance
    accepted = False
    try:
        if USE_TEXT_MODEL:
            accepted = civic_confidence_pct is not None and civic_confidence_pct >= CIVIC_CONFIDENCE_THRESHOLD
        else:
            accepted = is_civic_issue(top_labels)
    except Exception:
        accepted = False

    # 6) Determine category from labels
    category = "OTHER"
    for label in top_labels:
        label_lower = label.lower()
        if any(kw in label_lower for kw in ["pothole", "hole", "crack", "road", "pavement"]):
            category = "POTHOLE"
            break
        elif any(kw in label_lower for kw in ["garbage", "trash", "litter", "waste", "dump"]):
            category = "GARBAGE"
            break
        elif any(kw in label_lower for kw in ["street", "lamp", "light", "pole"]):
            category = "STREETLIGHT"
            break
        elif any(kw in label_lower for kw in ["water", "flood", "leak", "drain"]):
            category = "WATER"
            break
        elif any(kw in label_lower for kw in ["wire", "electric", "cable", "power"]):
            category = "ELECTRICITY"
            break

    # 7) build response
    top_label = results[0]["label"] if results else "unknown"
    message = f"Detected '{top_label}' with civic confidence {civic_confidence_pct:.2f}%"
    
    if not accepted:
        message = f"Low confidence ({civic_confidence_pct:.2f}%). Unable to detect clear civic issue."

    return {
        "accepted": accepted,
        "category": category,
        "top_label": top_label,
        "top_label_score": results[0]["score"] if results else 0.0,
        "civic_confidence_pct": civic_confidence_pct,
        "predictions": results,
        "message": message
    }


# Run with `python render_server.py` for local dev
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("render_server:app", host=host, port=port, log_level="info")
