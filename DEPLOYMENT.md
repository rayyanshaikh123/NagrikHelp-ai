# 🚀 Deployment Guide for NagrikHelp AI Server

## The Problem You Encountered

```
WARNING: YOLOv8 loading failed: libGL.so.1: cannot open shared object file
```

**Root Cause**: The package `opencv-python` requires GUI libraries (OpenGL, GTK) that aren't available on headless Linux servers.

**Solution**: Use `opencv-python-headless` which is designed for servers without displays.

---

## ✅ Fixed Files

1. **`requirements.txt`** (NEW) - Production dependencies with `opencv-python-headless`
2. **`Dockerfile`** - Updated to use `requirements.txt` and install minimal system dependencies
3. **`render.yaml`** - Simplified build process using `requirements.txt`

---

## 🐳 Option 1: Deploy with Docker

### Build the Docker image:
```bash
cd ai
docker build -t nagrikhelp-ai .
```

### Run locally:
```bash
docker run -p 8001:8001 \
  -e CONFIDENCE_THRESHOLD=0.45 \
  -e ENABLE_YOLO=true \
  -e ENABLE_CLIP=true \
  nagrikhelp-ai
```

### Deploy to Docker Hub:
```bash
# Tag and push
docker tag nagrikhelp-ai your-dockerhub-username/nagrikhelp-ai:latest
docker push your-dockerhub-username/nagrikhelp-ai:latest
```

---

## 🌐 Option 2: Deploy to Render

### Steps:

1. **Push your changes to GitHub**:
```bash
cd ai
git add .
git commit -m "fix: use opencv-python-headless for deployment"
git push origin main
```

2. **Create a new Web Service on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your `NagrikHelp-ai` repository
   - Render will auto-detect `render.yaml`

3. **Environment Variables** (already configured in `render.yaml`):
   - `CONFIDENCE_THRESHOLD=0.45`
   - `ENABLE_YOLO=true`
   - `ENABLE_CLIP=true`

4. **Deploy**:
   - Click "Create Web Service"
   - Wait 5-10 minutes for first deployment
   - Models (~4GB) will be downloaded during first request

---

## ☁️ Option 3: Deploy to Railway

### Steps:

1. **Install Railway CLI**:
```bash
npm install -g @railway/cli
railway login
```

2. **Initialize project**:
```bash
cd ai
railway init
```

3. **Add environment variables**:
```bash
railway variables set CONFIDENCE_THRESHOLD=0.45
railway variables set ENABLE_YOLO=true
railway variables set ENABLE_CLIP=true
```

4. **Deploy**:
```bash
railway up
```

---

## 🧪 Option 4: Deploy to Vercel (Serverless)

**⚠️ NOT RECOMMENDED** - AI models are too large (4GB+) for serverless cold starts. Use Docker/Render/Railway instead.

---

## 📊 Resource Requirements

| Service | RAM | Storage | Notes |
|---------|-----|---------|-------|
| **Render Free** | 512 MB | ❌ **Too small** | Will OOM during model loading |
| **Render Starter** | 2 GB | ✅ Works | Recommended minimum |
| **Railway** | 8 GB (free) | ✅ Works | Best free option |
| **Docker (VPS)** | 4 GB+ | ✅ Works | Full control |

---

## 🔍 Verify Deployment

After deployment, test your endpoint:

```bash
curl https://your-app-url.onrender.com/
```

Expected response:
```json
{
  "ok": true,
  "service": "NagrikHelp AI Validation",
  "models": {
    "yolo": true,
    "resnet": true,
    "clip": true
  },
  "confidence_threshold": 0.45
}
```

---

## 🐛 Troubleshooting

### Issue: Models fail to load (OOM)
**Solution**: Upgrade to a plan with more RAM (2GB minimum)

### Issue: Build times out
**Solution**: 
- Use `pip install --prefer-binary` (already in `render.yaml`)
- Increase build timeout in platform settings

### Issue: YOLO still fails with libGL error
**Solution**: Make sure you're using `requirements.txt` NOT `local_vision_requirements.txt`

---

## 📁 File Reference

- **`requirements.txt`** - Use this for deployment (has `opencv-python-headless`)
- **`local_vision_requirements.txt`** - Use this for local development (has `opencv-python`)
- **`Dockerfile`** - For Docker deployments
- **`render.yaml`** - For Render deployments

---

## 🎯 Next Steps

1. Push the fixed code:
```bash
git add requirements.txt Dockerfile render.yaml
git commit -m "fix: opencv-python-headless for headless deployment"
git push origin main
```

2. Choose your deployment platform (Render recommended for simplicity)

3. Update your frontend API endpoint to use the deployed URL

4. Test with a real image upload!

---

## 💡 Pro Tips

- **Free tier limitations**: Models load on first request (5-15 sec delay)
- **Production**: Use a paid plan with persistent storage to cache models
- **Monitoring**: Add health checks at `/` endpoint
- **Scaling**: Use load balancing for multiple instances

---

Need help? The AI server is now ready to deploy! 🚀
