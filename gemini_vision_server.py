"""
NagrikHelp AI Validation using HuggingFace BLIP-2 Vision Model
Free, accurate, and works on 512MB RAM - perfect for civic issue detection
"""

import io
import os
import base64
import logging
from typing import Dict, Any
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load vision model
blip_processor = None
blip_model = None

def load_vision_model():
    """Initialize BLIP-2 vision-language model"""
    global blip_processor, blip_model
    
    if blip_model is not None:
        return
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        logger.info("Loading BLIP vision model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.eval()
        logger.info("✓ BLIP vision model loaded")
        
    except ImportError:
        raise HTTPException(status_code=503, detail="transformers not installed")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=503, detail=f"Model error: {str(e)}")


def parse_image(image_data: str) -> Image.Image:
    """Parse base64 image data"""
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


def analyze_with_gemini(image: Image.Image, description: str = "") -> Dict[str, Any]:
    """Analyze civic issue using BLIP vision model"""
    
    load_vision_model()
    
    try:
        import torch
        
        # Generate caption describing the image
        inputs = blip_processor(image, return_tensors="pt")
        
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_length=50)
        
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        caption_lower = caption.lower()
        
        logger.info(f"BLIP caption: {caption}")
        
        # Keyword matching for civic issues
        category_keywords = {
            "POTHOLE": ["pothole", "hole", "crack", "damage", "road", "asphalt", "pavement"],
            "GARBAGE": ["garbage", "trash", "waste", "litter", "dump", "rubbish", "plastic", "bag"],
            "STREETLIGHT": ["light", "lamp", "pole", "streetlight", "lighting", "bulb"],
            "WATER": ["water", "leak", "pipe", "flood", "puddle", "wet", "drain"],
        }
        
        # Score each category
        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in caption_lower)
            if score > 0:
                scores[category] = score / len(keywords)
        
        # Check description too
        if description:
            desc_lower = description.lower()
            for category, keywords in category_keywords.items():
                desc_score = sum(1 for keyword in keywords if keyword in desc_lower)
                if desc_score > 0:
                    scores[category] = scores.get(category, 0) + (desc_score / len(keywords)) * 0.5
        
        # Determine best category
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            category = best_category[0]
            confidence = min(best_category[1] * 1.5, 0.95)  # Scale up confidence
            is_issue = confidence >= 0.3
        else:
            # Default to OTHER if no specific keywords found
            category = "OTHER"
            # Check if it looks like outdoor/civic scene
            civic_indicators = ["outdoor", "street", "road", "building", "city", "urban", "infrastructure"]
            has_civic = any(word in caption_lower for word in civic_indicators)
            confidence = 0.4 if has_civic else 0.2
            is_issue = has_civic
        
        reasoning = f"Image shows: {caption}. "
        reasoning += f"Detected keywords suggest {category} issue." if is_issue else "No clear civic issue detected."
        
        return {
            "isIssue": is_issue,
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning,
            "caption": caption
        }
        
    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")



@app.get("/")
@app.head("/")
def root():
    """Health check endpoint"""
    return {
        "ok": True,
        "service": "NagrikHelp AI Validation (BLIP Vision)",
        "model": "Salesforce/blip-image-captioning-base",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_loaded": blip_model is not None
    }


@app.post("/validate")
async def validate_issue(request: Request):
    """Main validation endpoint using Gemini Vision"""
    import time
    start_time = time.time()
    
    try:
        body = await request.json()
        
        if not body.get("image"):
            raise HTTPException(status_code=400, detail="'image' field required")
        
        # Parse image
        try:
            image = parse_image(body["image"])
        except Exception as e:
            logger.error(f"Image parsing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        description = body.get("description", "")
        
        # Analyze with Gemini
        result = analyze_with_gemini(image, description)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Format response
        is_valid = result["isIssue"] and result["confidence"] >= CONFIDENCE_THRESHOLD
        
        response = {
            "isIssue": is_valid,
            "category": result["category"],
            "confidence": round(result["confidence"], 3),
            "modelUsed": "BLIP-Vision",
            "message": result["reasoning"] if is_valid else f"Low confidence. {result['reasoning']}",
            "latencyMs": elapsed_ms,
            "rawLabels": [{"label": result.get("caption", ""), "score": result["confidence"]}],
            "bbox": None,
            "debug": {
                "blip_caption": result.get("caption", ""),
                "threshold": CONFIDENCE_THRESHOLD
            }
        }
        
        return JSONResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/classify")
async def classify_legacy(request: Request):
    """Legacy endpoint for backward compatibility"""
    try:
        body = await request.body()
        
        if not body:
            raise HTTPException(status_code=400, detail="Empty body")
        
        # Convert raw bytes to base64
        image_b64 = base64.b64encode(body).decode('utf-8')
        image_data = f"data:image/png;base64,{image_b64}"
        
        image = parse_image(image_data)
        result = analyze_with_gemini(image, "")
        
        # Return simplified format
        return JSONResponse([{
            "label": result["category"],
            "score": result["confidence"]
        }])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    logger.info(f"Starting NagrikHelp Gemini AI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
