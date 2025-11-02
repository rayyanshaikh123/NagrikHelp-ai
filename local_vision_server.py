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
from typing import List
import logging
import re
import json
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Example public models for quick reference inside the module
SUPPORTED_MODELS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "microsoft/resnet-50",
    "facebook/convnext-base-224",
]

# Keywords / phrases that indicate an image likely contains a civic issue.
# This is a heuristic list — expand as needed for your use-case or replace with
# a domain-specific model trained to detect civic issues (potholes, flooding,
# graffiti, illegal dumping, broken streetlights, etc.). Matching is case-
# insensitive and checks for substring presence in label names returned by the
# classifier.
CIVIC_KEYWORDS = [
    "pothole",
    "potholes",
    "graffiti",
    "garbage",
    "trash",
    "litter",
    "dump",
    "dumping",
    "illegal",
    "flood",
    "flooding",
    "standing water",
    "sewage",
    "overflow",
    "blocked",
    "blocked drain",
    "sinkhole",
    "road",
    "street",
    "traffic light",
    "streetlight",
    "lamp post",
    "sign",
    "broken",
    "damaged",
    "collapsed",
    "debris",
    "fallen",
    "fire",
    "smoke",
    "accident",
    "vandalism",
    "construction",
    "hole",
    "crack",
    "leak",
    "spill",
]


def is_civic_issue(labels: list) -> bool:
    """Return True if any of the given label strings contains a civic keyword.

    This is intentionally permissive (substring match) to work with labels
    returned by generic image classifiers. For production use, consider a
    specialized binary classifier trained to detect civic issues.
    """
    if not labels:
        return False
    lowered = [l.lower() for l in labels if isinstance(l, str)]
    for kw in CIVIC_KEYWORDS:
        for lab in lowered:
            if kw in lab:
                return True
    return False

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("LOCAL_VISION_MODEL", os.getenv("CIVIC_VISION_MODEL", "google/vit-base-patch16-224")))
TOP_K = int(os.getenv("TOP_K", "5"))
USE_TEXT_MODEL = os.getenv("USE_TEXT_MODEL", "0") in ("1", "true", "True")
# Text model provider: 'hf' for Hugging Face zero-shot (default), or 'gemini'
# to use Google Gemini/Vertex AI (requires external credentials and config).
TEXT_MODEL_PROVIDER = os.getenv("TEXT_MODEL_PROVIDER", "hf")
# Model to use for text-based zero-shot checking when provider is 'hf'. This
# model needs transformers + a backend (PyTorch/TF/Flax) available in the env.
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "facebook/bart-large-mnli")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-mini")
# Threshold (percent) above which we accept Gemini / text-model confidence as
# indicating a civic issue. Can be tuned via environment.
CIVIC_CONFIDENCE_THRESHOLD = float(os.getenv("CIVIC_CONFIDENCE_THRESHOLD", "30"))


def call_gemini_confidence(labels: List[str], required_labels: List[str]) -> float:
    """Call Gemini with the provided labels and required civic labels.

    The function builds a prompt asking Gemini to return a JSON object with
    a single numeric field `civic_confidence` (0-100). Returns the parsed
    numeric value, or 0.0 on failure.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    labels_text = "; ".join(labels)
    required_text = ", ".join(required_labels[:20])  # keep prompt short
    prompt_text = (
        "You are a classifier. Given the following detected image labels: '" + labels_text + "'.\n"
        "Also consider these civic-related labels (for reference): '" + required_text + "'.\n"
        "Return ONLY a JSON object with a single numeric field 'civic_confidence' whose value is a number between 0 and 100 representing the percent likelihood that the detected labels indicate a civic/municipal issue (pothole, flooding, graffiti, illegal dumping, broken streetlight, etc.).\n"
    )

    endpoint = f"https://generativelanguage.googleapis.com/v1beta2/{GEMINI_MODEL}:generate?key={GEMINI_API_KEY}"
    body = {"prompt": {"text": prompt_text}, "temperature": 0.0, "max_output_tokens": 256}
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_text = resp.read().decode("utf-8")
    except Exception as ex:
        logger.exception("Gemini request failed: %s", ex)
        return 0.0

    # Try to parse JSON from the response; fall back to regex
    try:
        parsed = json.loads(resp_text)
        # Try common shapes
        # 'candidates' -> list with 'content' or text
        cand_text = None
        if isinstance(parsed, dict):
            if "candidates" in parsed and parsed["candidates"]:
                cand = parsed["candidates"][0]
                if isinstance(cand, dict):
                    cand_text = cand.get("content") or cand.get("output") or json.dumps(cand)
                else:
                    cand_text = str(cand)
            elif "output" in parsed and parsed["output"]:
                cand = parsed["output"][0]
                cand_text = cand.get("content") if isinstance(cand, dict) else str(cand)
            else:
                cand_text = json.dumps(parsed)
        else:
            cand_text = str(parsed)
    except Exception:
        cand_text = resp_text

    m = re.search(r"(\d{1,3}(?:\.\d+)?)", cand_text)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(100.0, val))
        except Exception:
            return 0.0
    return 0.0

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

    # If the user has provided a Hugging Face token (for private/gated repos), use it
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    def _try_from_pretrained(name: str):
        # Centralize call so we can pass token when present.
        kwargs = {}
        if hf_token:
            # Use the modern `token` kwarg. Do NOT pass `use_auth_token` as some
            # transformer versions will raise if both are provided.
            kwargs["token"] = hf_token
        logger.info("Attempting to load model '%s' (use_auth_token=%s)", name, bool(hf_token))
        proc = AutoImageProcessor.from_pretrained(name, **kwargs)
        mod = AutoModelForImageClassification.from_pretrained(name, **kwargs)
        return proc, mod

    # First attempt: try the provided MODEL_NAME directly.
    e_first = None
    e_alt = None
    try:
        logger.info("Loading model %s ...", MODEL_NAME)
        processor, model = _try_from_pretrained(MODEL_NAME)
        model.eval()
        logger.info("Model %s loaded successfully", MODEL_NAME)
        return
    except Exception as exc1:
        e_first = exc1
        logger.debug("Initial load of model %s failed: %s", MODEL_NAME, e_first)

    # If the repo id looks like a common typo (owner-name instead of owner/name),
    # try a simple auto-fix: replace the first '-' with '/'. This will turn
    # microsoft-resnet-50 -> microsoft/resnet-50 which is a common mistake.
    alt_name = None
    if "/" not in MODEL_NAME and "-" in MODEL_NAME:
        alt_name = MODEL_NAME.replace("-", "/", 1)
        try:
            logger.warning("Model id '%s' not found; trying alternative id '%s'", MODEL_NAME, alt_name)
            processor, model = _try_from_pretrained(alt_name)
            model.eval()
            logger.info("Model %s loaded successfully (as %s)", MODEL_NAME, alt_name)
            return
        except Exception as exc2:
            e_alt = exc2
            logger.debug("Alternative load %s also failed: %s", alt_name, e_alt)

    # If we get here, both attempts failed. Provide clearer guidance in the raised error.
    guidance = (
        "Model identifier not found or access denied. "
        "If this is a private model, set HUGGINGFACE_HUB_TOKEN in the environment (Render secret) "
        "or log in with `huggingface-cli login`. Also verify MODEL_NAME uses owner/repo format, "
        "for example 'microsoft/resnet-50'."
    )
    logger.exception("Failed to load model %s. %s", MODEL_NAME, guidance)
    # Build a compact diagnostics string from captured exceptions.
    details = []
    if e_first is not None:
        details.append(f"initial error: {e_first}")
    if e_alt is not None:
        details.append(f"alternative error: {e_alt}")
    detail_msg = " | ".join(details) if details else "no exception captured"

    # Raise a RuntimeError with combined info to make logs and HTTP 503 responses helpful.
    # Attach the original exception as the __cause__ when available.
    cause = e_first or e_alt
    if cause is not None:
        raise RuntimeError(f"Failed to load model {MODEL_NAME}: {detail_msg}. {guidance}") from cause
    else:
        raise RuntimeError(f"Failed to load model {MODEL_NAME}: {detail_msg}. {guidance}")


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
        # Check for civic-issue signals in the predicted labels. If none of the
        # top labels looks like a civic issue, reject the image to avoid
        # processing irrelevant/non-civic content.
        top_labels = [r["label"] for r in results]
        # Compute a civic-confidence value. Prefer a text-based zero-shot model
        # (controlled by USE_TEXT_MODEL). If unavailable, fall back to the max
        # label probability produced by the image model.
        civic_confidence_pct = None

        def _fallback_confidence(results_list: List[dict]) -> float:
            # Use the highest softmax probability as a conservative proxy.
            if not results_list:
                return 0.0
            return max(r.get("score", 0.0) for r in results_list) * 100.0

        if USE_TEXT_MODEL:
            try:
                # Lazy import so environments without transformers/torch don't fail.
                if TEXT_MODEL_PROVIDER == "hf":
                    from transformers import pipeline

                    seq = "; ".join(top_labels) or ""
                    clf = pipeline(
                        "zero-shot-classification",
                        model=TEXT_MODEL_NAME,
                    )
                    # Candidate labels: civic vs not civic
                    candidate_labels = ["civic_issue", "not_civic_issue"]
                    z = clf(seq, candidate_labels)
                elif TEXT_MODEL_PROVIDER == "gemini":
                    # Call Gemini (Generative Language API) via REST using an
                    # API key. The caller must set GEMINI_API_KEY and optionally
                    # GEMINI_MODEL (defaults to models/gemini-1.5-mini).
                    if not GEMINI_API_KEY:
                        raise RuntimeError("TEXT_MODEL_PROVIDER=gemini selected but GEMINI_API_KEY is not set in the environment.")

                    def _call_gemini_for_confidence(text: str) -> float:
                        """Call the Generative Language `generate` endpoint and
                        parse a percentage confidence that the `text` indicates a
                        civic issue. Returns a float 0.0-100.0.
                        """
                        endpoint = f"https://generativelanguage.googleapis.com/v1beta2/{GEMINI_MODEL}:generate?key={GEMINI_API_KEY}"
                        # Prompt the model to return a strict JSON with a
                        # numeric 'civic_confidence' field to simplify parsing.
                        prompt_text = (
                            "You are a classifier. Given the following labels: '" + text + "' \n"
                            "Respond ONLY with a JSON object containing a single key 'civic_confidence' whose value is a number between 0 and 100 representing the percent likelihood that the labels indicate a civic/municipal issue (pothole, flooding, graffiti, illegal dumping, broken streetlight, etc.).\n"
                        )
                        body = {
                            "prompt": {"text": prompt_text},
                            "temperature": 0.0,
                            "max_output_tokens": 256,
                        }
                        data = json.dumps(body).encode("utf-8")
                        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
                        try:
                            with urllib.request.urlopen(req, timeout=15) as resp:
                                resp_text = resp.read().decode("utf-8")
                        except urllib.error.HTTPError as he:
                            logger.exception("Gemini HTTP error: %s", he)
                            raise
                        except Exception as ex:
                            logger.exception("Gemini request failed: %s", ex)
                            raise

                        # The response is JSON; try extracting candidate text.
                        try:
                            parsed = json.loads(resp_text)
                            # Expect 'candidates' or 'outputs' holding text.
                            cand_text = None
                            if isinstance(parsed, dict):
                                # new API surface: 'candidates' or 'output'
                                if "candidates" in parsed and parsed["candidates"]:
                                    cand_text = parsed["candidates"][0].get("content", "")
                                elif "output" in parsed and parsed["output"]:
                                    # fallback naming
                                    cand = parsed["output"][0]
                                    cand_text = cand.get("content", "") if isinstance(cand, dict) else str(cand)
                                elif "candidates" in parsed and isinstance(parsed.get("candidates"), list):
                                    cand_text = str(parsed.get("candidates"))
                                else:
                                    # Some responses put text under 'candidates'[0]['output']
                                    cand_text = json.dumps(parsed)
                            else:
                                cand_text = str(parsed)
                        except Exception:
                            # If parsing as JSON fails, treat entire response as text
                            cand_text = resp_text

                        # Find a number in the candidate text (0-100). If none,
                        # fallback to 0.
                        m = re.search(r"(\d{1,3}(?:\.\d+)?)", cand_text)
                        if m:
                            try:
                                val = float(m.group(1))
                                # clamp
                                val = max(0.0, min(100.0, val))
                                return val
                            except Exception:
                                return 0.0
                        # final fallback
                        return 0.0

                    # Use structured Gemini helper passing both detected labels
                    # and the canonical civic keyword list so Gemini can compare.
                    civic_confidence_pct = call_gemini_confidence(top_labels, CIVIC_KEYWORDS)
                # z['labels'] lists labels in order and 'scores' correspond.
                # Find the score for 'civic_issue' if present.
                if "civic_issue" in z.get("labels", []):
                    idx = z["labels"].index("civic_issue")
                    civic_confidence_pct = float(z.get("scores", [0.0])[idx]) * 100.0
                else:
                    civic_confidence_pct = float(z.get("scores", [0.0])[0]) * 100.0
            except Exception:
                # If any error occurs (missing backends, network, etc.), fall back.
                civic_confidence_pct = _fallback_confidence(results)
        else:
            civic_confidence_pct = _fallback_confidence(results)
        # Decide acceptance using the text-model confidence when enabled;
        # otherwise fall back to the heuristic substring matcher.
        accepted = False
        if USE_TEXT_MODEL:
            try:
                accepted = (civic_confidence_pct is not None and civic_confidence_pct >= CIVIC_CONFIDENCE_THRESHOLD)
            except Exception:
                accepted = False
        else:
            accepted = is_civic_issue(top_labels)

        if not accepted:
            logger.info("Image rejected: civic_confidence_pct=%s, top_labels=%s", civic_confidence_pct, top_labels)
            raise HTTPException(status_code=422, detail={
                "error": "No civic issue detected",
                "top_labels": top_labels,
                "civic_confidence_pct": civic_confidence_pct,
                "threshold": CIVIC_CONFIDENCE_THRESHOLD,
                "suggestion": "Take a photo that clearly shows a civic issue (pothole, flooding, graffiti, illegal dumping, damaged streetlight, etc.)"
            })

    # Attach civic confidence to the response for client-side UX.
        resp = {"predictions": results, "civic_confidence_pct": civic_confidence_pct}
        return JSONResponse(resp)
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


@app.get("/civic-keywords")
def civic_keywords():
    """Return the list of civic keywords used to filter accepted images.

    Clients can call this endpoint to show guidance or a help UI explaining
    what types of photos the service accepts.
    """
    return {"keywords": CIVIC_KEYWORDS}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("LOCAL_VISION_HOST", "127.0.0.1")
    port = int(os.getenv("LOCAL_VISION_PORT", "8001"))
    # Run directly without module string to avoid import path issues
    uvicorn.run(app, host=host, port=port)
