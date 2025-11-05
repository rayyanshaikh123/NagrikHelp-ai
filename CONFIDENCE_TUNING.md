# 🎚️ Adjusting AI Confidence Threshold

## Current Issue: "Low confidence rejection"

Your AI correctly identified the pothole but rejected it due to low confidence (49.5% < 60% default threshold).

## ✅ Solution Applied

**New Threshold: 0.45 (45%)**

This means:
- ✅ Your pothole image (49.5%) will now be **ACCEPTED**
- ✅ More legitimate issues will pass validation
- ⚠️ Slightly higher chance of false positives (but still reasonable)

---

## 🎯 How Confidence Works

The AI combines 3 models to produce a final confidence score:
- **YOLOv8** (40% weight) - Object detection
- **ResNet-50** (30% weight) - Image classification  
- **CLIP** (50% weight) - Semantic understanding

**Final Score = Weighted Average of All Models**

---

## ⚙️ Threshold Recommendations

### **Lenient (0.35 - 0.45)** ← **Current: 0.45**
- ✅ Accepts most valid issues
- ✅ Good for early testing
- ⚠️ May accept some borderline/unclear images
- **Best for:** Development, inclusivity, fewer rejections

### **Balanced (0.50 - 0.60)** ← Default was 0.60
- ✅ Good balance of accuracy
- ⚠️ May reject some valid but unclear photos
- **Best for:** Production, general use

### **Strict (0.65 - 0.80)**
- ✅ Only high-quality, clear images accepted
- ❌ Will reject many borderline cases
- **Best for:** Quality control, verified submissions

---

## 🔧 How to Change Threshold

### **Method 1: Edit .env.local (Permanent)**
```bash
# Edit ai/.env.local
CONFIDENCE_THRESHOLD=0.45  # Change this value
```

Then restart server:
```bash
cd ai
./start-server.sh
```

### **Method 2: Environment Variable (Temporary)**
```bash
export CONFIDENCE_THRESHOLD=0.45
python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8001
```

### **Method 3: Use Startup Script**
```bash
cd "/Applications/rayyan dev/NagrikHelp/ai"
./start-server.sh
# Automatically loads settings from .env.local
```

---

## 📊 Testing Different Thresholds

Try uploading the same pothole image with different thresholds:

| Threshold | Pothole Result (49.5%) | Behavior |
|-----------|----------------------|----------|
| 0.35 | ✅ Accepted | Very lenient |
| 0.40 | ✅ Accepted | Lenient |
| **0.45** | ✅ **Accepted** | **Current** |
| 0.50 | ❌ Rejected | Balanced |
| 0.55 | ❌ Rejected | Strict |
| 0.60 | ❌ Rejected | Default |

---

## 🎓 Understanding Your Pothole Result

```
Confidence: 49.5%
Category: POTHOLE
Status: LOW CONFIDENCE (with 0.60 threshold)
Status: ACCEPTED (with 0.45 threshold) ✅
```

**Why 49.5%?**
- The pothole is visible but may have:
  - ✅ Correct category identified
  - ⚠️ Image quality factors (angle, lighting, clarity)
  - ⚠️ No bounding box detected by YOLO
  - ⚠️ ResNet/CLIP contributing moderate scores

**This is a legitimate issue** that should be accepted! The 0.45 threshold is perfect for this.

---

## 🚀 Recommended Settings

### For Your Use Case (Civic Issues):
```bash
CONFIDENCE_THRESHOLD=0.45  # ← Current (good!)
ENABLE_YOLO=true
ENABLE_CLIP=true
```

**Why 0.45 is good:**
- Real civic issues often aren't "perfect" photos
- Citizens report in various lighting/angles
- Better to accept borderline cases than reject legitimate issues
- You can always manually verify later

---

## 📝 Quick Start Commands

### Check Current Threshold:
```bash
curl http://127.0.0.1:8001/ | grep confidence_threshold
```

### Start Server with Custom Threshold:
```bash
cd "/Applications/rayyan dev/NagrikHelp/ai"
export CONFIDENCE_THRESHOLD=0.45
source .venv/bin/activate
python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8001
```

### Or Use the Startup Script:
```bash
cd "/Applications/rayyan dev/NagrikHelp/ai"
./start-server.sh  # Loads .env.local automatically
```

---

## ✅ Current Status

- ✅ Server running with **0.45 threshold**
- ✅ Your pothole image will now be **accepted**
- ✅ Configuration saved in `.env.local`
- ✅ Startup script created: `start-server.sh`

**Try uploading your pothole image again - it should work now!** 🎉

---

## 🐛 Troubleshooting

**Still getting rejected?**
1. Check server is using new threshold: `curl http://127.0.0.1:8001/`
2. Clear browser cache / reload page
3. Re-upload the image
4. Check console for AI response

**Want even more lenient?**
- Set `CONFIDENCE_THRESHOLD=0.40` in `.env.local`
- Restart server with `./start-server.sh`

---

**Updated:** Server now running with 0.45 threshold  
**Status:** Ready to accept your pothole images! ✅
