# Deploying to Render (fast, no-Docker flow)

This repository is configured to run on Render using `render.yaml` (already present). This document shows exact commands to build and test locally, and the steps to deploy on Render's dashboard.

Prerequisites
- Git remote (push access) for this repo
- A Render account (https://render.com)
- Optional for classification: Python 3.11/3.12 and a CPU `torch` wheel (recommended)

Quick local build & smoke test (fast)

1. Create and activate a virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the lightweight dependencies (used for serving the GET / endpoint and CI smoke test):

```bash
pip install -r requirements-light.txt
```

3. Load local env vars and run the server:

```bash
export $(grep -v '^[[:space:]]*#' .env.local | xargs)
python -m uvicorn local_vision_server:app --host $LOCAL_VISION_HOST --port $LOCAL_VISION_PORT
```

4. Verify the root endpoint:

```bash
curl http://$LOCAL_VISION_HOST:$LOCAL_VISION_PORT/
# Expect: {"ok": true, "model": "microsoft-resnet-500"}
```

Enable full classification (optional)
- To POST images to `/classify` you will need `transformers` and `torch` installed. We recommend using Python 3.11 or 3.12 to avoid building `tokenizers` from source.
- Install full requirements (after creating a venv with python3.11):

```bash
# Example (make sure python3.11 exists on your machine)
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r local_vision_requirements.txt
# Install CPU torch wheel (adjust if you need CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

When you call `POST /classify` the model weights will be downloaded on first use (unless you pre-cache them).

Deploying on Render (Dashboard)

1. Push your branch to GitHub/GitLab/Bitbucket:

```bash
git add .
git commit -m "Prepare Render deployment"
git push origin main
```

2. In the Render dashboard create a new Web Service and connect your repository. When Render reads the repo it will detect `render.yaml` and configure the service accordingly.

3. Review the service settings that Render displays. Confirm:
   - The build command is: `pip install -r local_vision_requirements.txt`
   - The start command is: `uvicorn local_vision_server:app --host 0.0.0.0 --port $PORT`

4. Environment variables
   - `render.yaml` already contains `MODEL_NAME` and `TOP_K`. You can override them in the Render dashboard under Environment > Environment Variables.
   - For secrets (API keys, credentials), add them via the Render dashboard under Environment > Secrets. Do not commit secret values to the repo.

5. Deploy
   - After you create the service Render will build and deploy automatically. Subsequent pushes to the branch will trigger new builds.

Notes & troubleshooting
- The `tokenizers` package is a Rust extension used by `transformers` and sometimes needs to be compiled if a wheel isn't available for your Python version/architecture — using Python 3.11 or 3.12 on Render avoids that for most platforms.
- If builds fail on Render due to `torch`/`tokenizers`, prefer adding the appropriate prebuilt wheel to `local_vision_requirements.txt` (for example, a `torch` CPU wheel via the PyTorch index) or change the service instance to one with more memory if necessary.
- Render provides the `$PORT` env var at runtime; do not hardcode a port in production.

If you want, I can:
- Add an example Render secret mapping to `render.yaml` (placeholder keys only) so you have a template.
- Add a `Makefile` with `make dev`, `make install-full`, and `make start` targets to simplify local workflows.
