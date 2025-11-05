# 🚀 Quick Start: AI Image Validation Setup

## Installation & Running

### 1. Install Dependencies

```bash
cd "/Applications/rayyan dev/NagrikHelp/ai"

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# or on Windows: .venv\Scripts\activate

# Install packages
pip install -r local_vision_requirements.txt
```

### 2. Start AI Server

```bash
# Make sure you're in the ai directory with venv activated
python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8001
```

**Expected Output:**
```
INFO: Loading YOLOv8 detection model...
INFO: ✓ YOLOv8 pothole detection model loaded
INFO: Loading ResNet-50 classification model...
INFO: ✓ ResNet-50 loaded
INFO: Loading CLIP zero-shot model...
INFO: ✓ CLIP loaded
INFO: Started server on 0.0.0.0:8001
```

### 3. Test the Server

Open a new terminal:

```bash
# Test health check
curl http://127.0.0.1:8001/

# Should return:
# {"ok": true, "service": "NagrikHelp AI Validation", ...}
```

### 4. Configure Frontend

Update your `frontend/.env.local`:

```env
LOCAL_VISION_URL=http://127.0.0.1:8001
LOCAL_VISION_TIMEOUT_MS=30000
```

### 5. Use in Your App

Navigate to `/citizen/create` in your app and upload an image. The AI will automatically:
- ✅ Detect if it's a valid civic issue
- ✅ Suggest the issue category
- ✅ Show confidence score
- ✅ Display bounding boxes (if detected)

---

## ⚡ Running the Full Stack

**Terminal 1 (AI Server):**
```bash
cd "/Applications/rayyan dev/NagrikHelp/ai"
source .venv/bin/activate
python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8001
```

**Terminal 2 (Backend - if needed):**
```bash
cd "/Applications/rayyan dev/NagrikHelp/backend"
./gradlew bootRun
```

**Terminal 3 (Frontend):**
```bash
cd "/Applications/rayyan dev/NagrikHelp/frontend"
npm run dev
```

---

## 🎯 Testing AI Validation

### Test with Sample Images

1. Navigate to: `http://localhost:3000/citizen/create`
2. Upload a test image:
   - **Pothole**: Photo of road with cracks/holes
   - **Garbage**: Photo of trash/litter
   - **Water**: Photo of waterlogging/leaks
3. Watch the AI analyze in real-time
4. Check the confidence score and suggested category

### Expected Behavior

**Valid Issue (High Confidence > 0.6):**
- ✅ Green border
- ✅ Category auto-selected
- ✅ Submit button enabled
- ✅ Shows "Detected [category] issue with XX% confidence"

**Invalid/Unclear (Low Confidence < 0.6):**
- ⚠️ Yellow border
- ⚠️ Submit button disabled
- ⚠️ Shows "Low confidence. Upload clearer photo"

---

## 🔧 Troubleshooting

### Server Won't Start

**Error: "Module not found"**
```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r local_vision_requirements.txt
```

**Error: "Port already in use"**
```bash
# Kill existing process
lsof -ti:8001 | xargs kill -9

# Or use different port
python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8002
```

### Models Not Loading

**Error: "Failed to load model"**
- Check internet connection (models download on first run)
- Wait 2-3 minutes for initial download
- Models are cached after first download

**Storage**: First run needs ~4GB for all models

### Frontend Can't Connect

**Error: "Local AI server not available"**

1. Verify server is running: `curl http://127.0.0.1:8001/`
2. Check `LOCAL_VISION_URL` in `.env.local`
3. Restart frontend dev server

---

## 📊 Performance Tips

### Speed Up Inference

1. **Use GPU** (if available):
   ```bash
   # Install CUDA-enabled PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Lower Confidence Threshold** (more lenient):
   ```bash
   export CONFIDENCE_THRESHOLD=0.5
   ```

3. **Disable Heavy Models**:
   ```bash
   export ENABLE_CLIP=false  # Faster but less accurate
   ```

### Reduce Memory Usage

```bash
# Disable YOLO if not needed
export ENABLE_YOLO=false

# This reduces memory from ~4GB to ~2GB
```

---

## 📈 Monitoring

### Check Model Status

```bash
curl http://127.0.0.1:8001/model-status
```

Returns:
```json
{
  "loaded": {
    "yolo": true,
    "resnet": true,
    "clip": true
  },
  "all_loaded": true,
  "confidence_threshold": 0.6
}
```

### View Server Logs

The server logs show detailed information:
- Model loading progress
- Request processing time
- Confidence scores from each model
- Debug information

---

## 🎓 Next Steps

1. **Test with various images** - potholes, garbage, water issues
2. **Adjust confidence threshold** - find what works best
3. **Review debug info** - see which models contribute most
4. **Integrate with backend** - save AI metadata with issues

---

## 📚 Full Documentation

See `AI_MODELS_DOCUMENTATION.md` for:
- Detailed model information
- Pipeline architecture
- API reference
- Fine-tuning guides
- Dataset information

---

**Happy Building! 🎉**
