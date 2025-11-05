"""
AI-Powered Civic Issue Image Validation Server using Hugging Face Transformers.
"""

import io
import os
import base64
from typing import List, Dict, Optional, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
ENABLE_YOLO = os.getenv("ENABLE_YOLO", "true").lower() == "true"
ENABLE_CLIP = os.getenv("ENABLE_CLIP", "true").lower() == "true"
TOP_K = int(os.getenv("TOP_K", "5"))

# Civic issue categories mapping
CIVIC_CATEGORIES = {
    "POTHOLE": ["pothole", "road damage", "asphalt", "pavement", "crack", "hole", "street"],
    "GARBAGE": ["garbage", "trash", "litter", "waste", "dump", "rubbish", "debris", "plastic", "bag"],
    "STREETLIGHT": ["streetlight", "lamp", "light pole", "lighting", "bulb", "street lamp"],
    "WATER": ["water", "leak", "pipe", "flood", "sewage", "drain", "waterlogging", "puddle"],
    "ELECTRICITY": ["wire", "electric", "electricity", "cable", "power line", "exposed wire"],
    "OTHER": ["issue", "problem", "civic", "municipal", "road", "street"]
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model = None
resnet_processor = None
resnet_model = None
clip_processor = None
clip_model = None


def load_models():
    global yolo_model, resnet_processor, resnet_model, clip_processor, clip_model
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    kwargs = {"token": hf_token} if hf_token else {}
    
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except ImportError as e:
        logger.error(f"Failed to import transformers/torch: {e}")
        raise HTTPException(status_code=503, detail="Required libraries not installed")
    
    if ENABLE_YOLO and yolo_model is None:
        try:
            logger.info("Loading YOLOv8 detection model...")
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8n.pt")
            logger.info("✓ YOLOv8 base model loaded")
        except Exception as e:
            logger.warning(f"YOLOv8 loading failed: {e}. Detection disabled.")
            
    if resnet_model is None:
        try:
            logger.info("Loading ResNet-50 classification model...")
            resnet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", **kwargs)
            resnet_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50", **kwargs)
            resnet_model.eval()
            logger.info("✓ ResNet-50 loaded")
        except Exception as e:
            logger.error(f"ResNet-50 loading failed: {e}")
            raise HTTPException(status_code=503, detail=f"ResNet model failed: {e}")
    
    if ENABLE_CLIP and clip_model is None:
        try:
            logger.info("Loading CLIP zero-shot model...")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", **kwargs)
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", **kwargs)
            clip_model.eval()
            logger.info("✓ CLIP loaded")
        except Exception as e:
            logger.warning(f"CLIP loading failed: {e}. Zero-shot disabled.")


def detect_with_yolo(image):
    if not ENABLE_YOLO or yolo_model is None:
        return False, None, 0.0
    
    try:
        results = yolo_model(image, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            bbox = boxes.xyxy[best_idx].cpu().numpy().tolist()
            conf = float(confidences[best_idx])
            
            logger.info(f"YOLOv8 detected object with confidence {conf:.3f}")
            return True, bbox, conf
        
        return False, None, 0.0
    except Exception as e:
        logger.warning(f"YOLO detection failed: {e}")
        return False, None, 0.0


def classify_with_resnet(image):
    try:
        import torch
        
        inputs = resnet_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = resnet_model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1)[0]
        topk_scores, topk_indices = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))
        
        id2label = resnet_model.config.id2label
        results = []
        for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
            label = id2label.get(int(idx), str(idx))
            results.append({"label": label, "score": float(score)})
        
        return results
    except Exception as e:
        logger.error(f"ResNet classification failed: {e}")
        return []


def classify_with_clip(image, description=None):
    if not ENABLE_CLIP or clip_model is None:
        return {}
    
    try:
        import torch
        
        text_prompts = []
        category_map = []
        
        for category in CIVIC_CATEGORIES.keys():
            text_prompts.append(f"a photo of a {category.lower()} civic issue")
            category_map.append(category)
        
        if description:
            text_prompts.append(f"a photo showing {description}")
            category_map.append("DESCRIPTION_MATCH")
        
        inputs = clip_processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
        scores = {}
        for idx, category in enumerate(category_map):
            scores[category] = float(probs[idx])
        
        return scores
    except Exception as e:
        logger.warning(f"CLIP classification failed: {e}")
        return {}


def map_labels_to_civic_category(labels):
    if not labels:
        return "OTHER", 0.0
    
    category_scores = {cat: 0.0 for cat in CIVIC_CATEGORIES.keys()}
    
    for item in labels:
        label_lower = item["label"].lower()
        score = item["score"]
        
        for category, keywords in CIVIC_CATEGORIES.items():
            for keyword in keywords:
                if keyword in label_lower:
                    category_scores[category] += score
                    break
    
    best_category = max(category_scores.items(), key=lambda x: x[1])
    
    if best_category[1] < 0.1:
        top_label = labels[0]["label"].lower()
        infrastructure_keywords = ["road", "street", "building", "outdoor", "urban", "city"]
        if any(kw in top_label for kw in infrastructure_keywords):
            return "OTHER", labels[0]["score"] * 0.5
        else:
            return "OTHER", 0.0
    
    return best_category[0], min(best_category[1], 1.0)


def parse_image(data):
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    
    image_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.get("/")
@app.head("/")
def root():
    models_loaded = {
        "yolo": yolo_model is not None,
        "resnet": resnet_model is not None,
        "clip": clip_model is not None,
    }
    return {
        "ok": True,
        "service": "NagrikHelp AI Validation",
        "models": models_loaded,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.post("/validate")
async def validate_issue(request: Request):
    import time
    start_time = time.time()
    
    try:
        body = await request.json()
        
        if not body.get("image"):
            raise HTTPException(status_code=400, detail="'image' field required")
        
        load_models()
        
        image = parse_image(body["image"])
        description = body.get("description", "")
        
        has_detection, bbox, yolo_conf = detect_with_yolo(image)
        resnet_labels = classify_with_resnet(image)
        resnet_category, resnet_conf = map_labels_to_civic_category(resnet_labels)
        clip_scores = classify_with_clip(image, description)
        
        final_scores = {cat: 0.0 for cat in CIVIC_CATEGORIES.keys()}
        
        if has_detection and yolo_conf > 0.5:
            final_scores["POTHOLE"] += yolo_conf * 0.2
            final_scores["GARBAGE"] += yolo_conf * 0.2
            final_scores["OTHER"] += yolo_conf * 0.1
            logger.info(f"YOLO contributed {yolo_conf:.3f} to detection")
        
        if resnet_conf > 0.1:
            final_scores[resnet_category] += resnet_conf * 0.3
            logger.info(f"ResNet contributed {resnet_conf:.3f} to {resnet_category}")
        
        for category, score in clip_scores.items():
            if category in final_scores and score > 0.1:
                final_scores[category] += score * 0.5
                logger.info(f"CLIP contributed {score:.3f} to {category}")
        
        best_category = max(final_scores.items(), key=lambda x: x[1])
        final_category = best_category[0]
        final_confidence = min(best_category[1], 1.0)
        
        is_valid = final_confidence >= CONFIDENCE_THRESHOLD
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        models_used = []
        if has_detection:
            models_used.append("YOLOv8")
        if resnet_labels:
            models_used.append("ResNet-50")
        if clip_scores:
            models_used.append("CLIP")
        
        result = {
            "isIssue": is_valid,
            "category": final_category,
            "confidence": round(final_confidence, 3),
            "bbox": bbox if bbox else None,
            "modelUsed": "+".join(models_used) if models_used else "ResNet-50",
            "message": (
                f"Detected {final_category.lower()} issue with {final_confidence*100:.1f}% confidence"
                if is_valid
                else f"Low confidence ({final_confidence*100:.1f}%). Unable to detect clear civic issue."
            ),
            "latencyMs": elapsed_ms,
            "rawLabels": resnet_labels[:5],
            "debug": {
                "yolo": {"detected": has_detection, "conf": yolo_conf},
                "resnet": {"category": resnet_category, "conf": resnet_conf},
                "clip": dict(list(clip_scores.items())[:5]) if clip_scores else {}
            }
        }
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify_legacy(request: Request):
    try:
        body = await request.body()
        
        if not body:
            raise HTTPException(status_code=400, detail="Empty body")
        
        load_models()
        
        image = Image.open(io.BytesIO(body)).convert("RGB")
        results = classify_with_resnet(image)
        
        return JSONResponse(results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-status")
def model_status():
    loaded = {
        "yolo": yolo_model is not None,
        "resnet": resnet_model is not None,
        "clip": clip_model is not None,
    }
    return {
        "loaded": loaded,
        "all_loaded": all(loaded.values()),
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("LOCAL_VISION_HOST", "0.0.0.0")
    port = int(os.getenv("LOCAL_VISION_PORT", "8001"))
    logger.info(f"Starting NagrikHelp AI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
