# AI Server Deployment Guide# 🚀 Deployment Guide for NagrikHelp AI Server



## 🚀 Quick Start## The Problem You Encountered



### Run Local Development Server```

WARNING: YOLOv8 loading failed: libGL.so.1: cannot open shared object file

```bash```

# Navigate to ai directory

cd ai**Root Cause**: The package `opencv-python` requires GUI libraries (OpenGL, GTK) that aren't available on headless Linux servers.



# Activate virtual environment (if not already activated)**Solution**: Use `opencv-python-headless` which is designed for servers without displays.

source .venv/bin/activate

---

# Start the development server with hot reload

python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8001 --reload## ✅ Fixed Files

```

1. **`requirements.txt`** (NEW) - Production dependencies with `opencv-python-headless`

**Server will be available at:** `http://localhost:8001`2. **`Dockerfile`** - Updated to use `requirements.txt` and install minimal system dependencies

3. **`render.yaml`** - Simplified build process using `requirements.txt`

**Features:**

- ✓ YOLOv8 object detection---

- ✓ ResNet-50 classification

- ✓ CLIP zero-shot classification## 🐳 Option 1: Deploy with Docker

- ✓ Hot reload enabled (auto-restart on code changes)

### Build the Docker image:

---```bash

cd ai

## 📦 Deploy to Production (Render)docker build -t nagrikhelp-ai .

```

### Push Changes to Deploy

### Run locally:

```bash```bash

# Make sure you're in the root directorydocker run -p 8001:8001 \

cd /Applications/rayyan\ dev/NagrikHelp  -e CONFIDENCE_THRESHOLD=0.45 \

  -e ENABLE_YOLO=true \

# Check status of changes  -e ENABLE_CLIP=true \

git status  nagrikhelp-ai

```

# Add all changes

git add .### Deploy to Docker Hub:

```bash

# Commit with descriptive message# Tag and push

git commit -m "deploy: update AI server for production"docker tag nagrikhelp-ai your-dockerhub-username/nagrikhelp-ai:latest

docker push your-dockerhub-username/nagrikhelp-ai:latest

# Push to main branch (triggers Render auto-deploy)```

git push origin main

```---



### What Happens After Push:## 🌐 Option 2: Deploy to Render

1. **Render detects changes** on `main` branch

2. **Builds Docker image** using `ai/Dockerfile`### Steps:

3. **Installs dependencies** from `ai/requirements-render.txt`

4. **Starts** `render_server.py` (lightweight, HF API-based)1. **Push your changes to GitHub**:

5. **Available at:** `https://nagrikhelp-ai.onrender.com````bash

cd ai

---git add .

git commit -m "fix: use opencv-python-headless for deployment"

## ⚙️ Configurationgit push origin main

```

### Local Development Environment Variables

2. **Create a new Web Service on Render**:

```bash   - Go to [Render Dashboard](https://dashboard.render.com/)

# Optional: Configure model behavior   - Click "New +" → "Web Service"

export ENABLE_YOLO=true          # Enable/disable YOLO detection   - Connect your `NagrikHelp-ai` repository

export ENABLE_CLIP=true          # Enable/disable CLIP classification   - Render will auto-detect `render.yaml`

export CONFIDENCE_THRESHOLD=0.5  # Adjust confidence threshold

```3. **Environment Variables** (already configured in `render.yaml`):

   - `CONFIDENCE_THRESHOLD=0.45`

### Production Environment Variables (Render Dashboard)   - `ENABLE_YOLO=true`

   - `ENABLE_CLIP=true`

1. Go to [Render Dashboard](https://dashboard.render.com)

2. Select your service: `nagrikhelp-ai`4. **Deploy**:

3. Go to **Environment** tab   - Click "Create Web Service"

4. Add environment variable:   - Wait 5-10 minutes for first deployment

   - **Key:** `HUGGINGFACE_API_TOKEN`   - Models (~4GB) will be downloaded during first request

   - **Value:** `your_hf_token_here`

   - *Optional but recommended for higher rate limits*---



---## ☁️ Option 3: Deploy to Railway



## 🧪 Testing### Steps:



### Test Local Server1. **Install Railway CLI**:

```bash

```bashnpm install -g @railway/cli

# Health checkrailway login

curl http://localhost:8001/```



# Test validation endpoint (with sample image)2. **Initialize project**:

curl -X POST http://localhost:8001/validate \```bash

  -H "Content-Type: application/json" \cd ai

  -d '{railway init

    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="```

  }'

```3. **Add environment variables**:

```bash

### Test Production Serverrailway variables set CONFIDENCE_THRESHOLD=0.45

railway variables set ENABLE_YOLO=true

```bashrailway variables set ENABLE_CLIP=true

# Health check```

curl https://nagrikhelp-ai.onrender.com/

4. **Deploy**:

# Test validation endpoint```bash

curl -X POST https://nagrikhelp-ai.onrender.com/validate \railway up

  -H "Content-Type: application/json" \```

  -d '{

    "image_base64": "YOUR_BASE64_IMAGE_HERE"---

  }'

```## 🧪 Option 4: Deploy to Vercel (Serverless)



---**⚠️ NOT RECOMMENDED** - AI models are too large (4GB+) for serverless cold starts. Use Docker/Render/Railway instead.



## 📊 Server Comparison---



| Feature | Local Dev Server | Production Server |## 📊 Resource Requirements

|---------|-----------------|-------------------|

| File | `local_vision_server.py` | `render_server.py` || Service | RAM | Storage | Notes |

| Models | YOLO + ResNet + CLIP | microsoft/resnet-50 ||---------|-----|---------|-------|

| Memory | ~2-4GB | ~100MB || **Render Free** | 512 MB | ❌ **Too small** | Will OOM during model loading |

| Speed | Fast (local inference) | Medium (API calls) || **Render Starter** | 2 GB | ✅ Works | Recommended minimum |

| Accuracy | Highest (multi-model) | Good (single model) || **Railway** | 8 GB (free) | ✅ Works | Best free option |

| Cost | Free (local) | Free (Render tier) || **Docker (VPS)** | 4 GB+ | ✅ Works | Full control |



------



## 🔧 Troubleshooting## 🔍 Verify Deployment



### Port Already in UseAfter deployment, test your endpoint:



```bash```bash

# Kill process on port 8001curl https://your-app-url.onrender.com/

lsof -ti:8001 | xargs kill -9```



# Then restart serverExpected response:

python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8001 --reload```json

```{

  "ok": true,

### Models Not Loading  "service": "NagrikHelp AI Validation",

  "models": {

```bash    "yolo": true,

# Reinstall dependencies    "resnet": true,

pip install -r requirements-local.txt    "clip": true

  },

# Clear model cache  "confidence_threshold": 0.45

rm -rf ~/.cache/huggingface/hub/}

rm -f yolov8n.pt```

```

---

### Render Deployment Failed

## 🐛 Troubleshooting

1. Check build logs in Render dashboard

2. Verify `requirements-render.txt` has no errors### Issue: Models fail to load (OOM)

3. Ensure `Dockerfile` points to correct server file**Solution**: Upgrade to a plan with more RAM (2GB minimum)

4. Check if HUGGINGFACE_API_TOKEN is valid (if set)

### Issue: Build times out

---**Solution**: 

- Use `pip install --prefer-binary` (already in `render.yaml`)

## 📝 Notes- Increase build timeout in platform settings



- **Local server** uses full ML models for maximum accuracy### Issue: YOLO still fails with libGL error

- **Production server** uses Hugging Face Inference API (no local models)**Solution**: Make sure you're using `requirements.txt` NOT `local_vision_requirements.txt`

- Both servers expose the same `/validate` endpoint

- Frontend works with both servers without code changes---

- Git push to `main` automatically deploys to Render

## 📁 File Reference

---

- **`requirements.txt`** - Use this for deployment (has `opencv-python-headless`)

## 🎯 Development Workflow- **`local_vision_requirements.txt`** - Use this for local development (has `opencv-python`)

- **`Dockerfile`** - For Docker deployments

```bash- **`render.yaml`** - For Render deployments

# 1. Make changes to local_vision_server.py

vim local_vision_server.py---



# 2. Server auto-reloads (hot reload enabled)## 🎯 Next Steps

# Test your changes at http://localhost:8001

1. Push the fixed code:

# 3. When ready to deploy to production:```bash

git add .git add requirements.txt Dockerfile render.yaml

git commit -m "feat: improved classification accuracy"git commit -m "fix: opencv-python-headless for headless deployment"

git push origin maingit push origin main

```

# 4. Monitor deployment at https://dashboard.render.com

```2. Choose your deployment platform (Render recommended for simplicity)



---3. Update your frontend API endpoint to use the deployed URL



**Last Updated:** November 5, 20254. Test with a real image upload!


---

## 💡 Pro Tips

- **Free tier limitations**: Models load on first request (5-15 sec delay)
- **Production**: Use a paid plan with persistent storage to cache models
- **Monitoring**: Add health checks at `/` endpoint
- **Scaling**: Use load balancing for multiple instances

---

Need help? The AI server is now ready to deploy! 🚀
