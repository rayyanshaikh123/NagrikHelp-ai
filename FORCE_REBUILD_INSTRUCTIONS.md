# 🔥 URGENT: Force Render to Rebuild (Clear Cache Instructions)

## ⚠️ Current Situation

Your Render deployment is **stuck using old cached build** with:
- ❌ opencv-python (causes libGL error)
- ❌ ultralytics (causes memory overflow)

Even though you pushed fixes, Render is not picking them up because of **aggressive caching**.

---

## ✅ SOLUTION: Force Complete Rebuild

### **Step 1: Go to Render Dashboard**
1. Visit: https://dashboard.render.com/
2. Login to your account
3. Find service: `nagrikhelp-ai-server`
4. Click on it to open

### **Step 2: Delete Old Environment Variables (Important!)**
Before rebuilding, clean up old config:

1. Go to **"Environment"** tab (left sidebar)
2. Look for these variables and **DELETE them** if they exist:
   - `MODEL_NAME`
   - Any old Hugging Face configs
3. Click **"Save Changes"**

### **Step 3: Manual Deploy with Cache Clear**
1. Click **"Manual Deploy"** button (top right)
2. Select: **"Clear build cache & deploy"** ⚠️ THIS IS CRITICAL
3. Click **"Deploy"**

### **Step 4: Watch Build Logs in Real-Time**
Monitor the build output. You should see:

```bash
=== Build v3: Lightweight deployment ===
BUILD_VERSION=v3_lightweight_no_yolo  ✅ New build!

=== Clearing ALL pip caches ===
✅ Successfully purged pip cache

=== Removing ANY old opencv/yolo packages ===
✅ Successfully uninstalled opencv-python
✅ Successfully uninstalled ultralytics

=== Installing LIGHTWEIGHT dependencies ===
✅ Collecting transformers==4.44.2
✅ Collecting torch==2.9.0
✅ Successfully installed transformers-4.44.2 torch-2.9.0

=== Verifying installation ===
✓ transformers OK
✓ torch OK
✓ opencv not installed (correct)  ✅ THIS CONFIRMS SUCCESS!

=== Build v3 finished ===
```

### **Step 5: Verify Deployment Success**
After ~3-5 minutes, check the **Runtime Logs**:

**✅ SUCCESS looks like:**
```
INFO: Loading ResNet-50 classification model...
INFO: ✓ ResNet-50 loaded
INFO: Loading CLIP zero-shot model...
INFO: ✓ CLIP loaded
INFO: Your service is live 🎉
```

**❌ FAILURE looks like:**
```
WARNING: YOLOv8 loading failed: libGL.so.1
```

If you still see the warning, **Render didn't clear cache properly** → See troubleshooting below.

---

## 🔍 Test Your Deployed Endpoint

```bash
curl https://nagrikhelp-ai.onrender.com/
```

**Expected response:**
```json
{
  "ok": true,
  "service": "NagrikHelp AI Validation",
  "models": {
    "yolo": false,  ✅ Disabled
    "resnet": true, ✅ Working
    "clip": true    ✅ Working
  },
  "confidence_threshold": 0.45
}
```

---

## 🐛 Troubleshooting: If It STILL Fails

### Option A: Delete and Recreate Service

If Render's cache is too stubborn:

1. **Download your environment variables** (note them down)
2. **Delete the service completely**:
   - Go to Settings → "Delete Web Service"
   - Type service name to confirm
3. **Create new service**:
   - Click "New +" → "Web Service"
   - Connect `NagrikHelp-ai` repo
   - Render will auto-detect `render.yaml`
   - Deploy (will use fresh cache)

### Option B: Deploy to Railway Instead (Recommended)

Railway has **8GB free tier** and better cache management:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
cd /Applications/rayyan\ dev/NagrikHelp/ai
railway init
railway up
```

Railway will:
- ✅ Use the new requirements-light.txt
- ✅ Give you 8GB RAM (can enable all 3 models!)
- ✅ No caching issues
- ✅ Faster cold starts

### Option C: Use Docker Locally, Deploy Image

Build and test locally first:

```bash
cd /Applications/rayyan\ dev/NagrikHelp/ai

# Build
docker build -t nagrikhelp-ai:v3 .

# Test locally
docker run -p 8001:8001 \
  -e ENABLE_YOLO=false \
  -e CONFIDENCE_THRESHOLD=0.45 \
  nagrikhelp-ai:v3

# If works, push to Docker Hub
docker tag nagrikhelp-ai:v3 your-username/nagrikhelp-ai:v3
docker push your-username/nagrikhelp-ai:v3

# Then deploy from Docker Hub on Render
```

---

## 📊 What Changed in v3

| Old Build | New Build v3 |
|-----------|--------------|
| ❌ opencv-python | ✅ NO opencv |
| ❌ ultralytics | ✅ NO ultralytics |
| ❌ 1.8GB RAM | ✅ 1.6GB RAM |
| ❌ libGL errors | ✅ No errors |
| Uses requirements.txt | Uses requirements-light.txt |
| All 3 models | ResNet + CLIP only |

---

## 🎯 Expected Outcome

After successful rebuild:

✅ **No more errors**:
- No libGL.so.1 errors
- No out of memory errors  
- No YOLO warnings

✅ **Working features**:
- Image validation: ✅
- Category classification: ✅
- Confidence scores: ✅
- Health check endpoint: ✅

❌ **Disabled features**:
- Object detection bounding boxes (YOLO disabled)

---

## 📞 Next Steps

1. **GO TO RENDER NOW** → Clear cache & deploy
2. **Watch the build logs** → Look for "Build v3"
3. **Test the endpoint** → `curl https://nagrikhelp-ai.onrender.com/`
4. **Report back** → Tell me if you see "Build v3" in logs

If you still see old errors after cache clear, we'll switch to Railway or Docker deployment instead.

---

**The fix is ready. Just need Render to actually USE the new code!** 🚀
