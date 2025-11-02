# Docker build & run for the local vision server

This file explains how to build and run a Docker image for the FastAPI-based local vision server in this repo.

Build the image (run from repo root):

```bash
docker build -t local-vision-server:latest .
```

Run the container (expose port 8001):

```bash
docker run --rm -p 8001:8001 \
  -e LOCAL_VISION_HOST=0.0.0.0 \
  -e LOCAL_VISION_PORT=8001 \
  local-vision-server:latest
```

Notes and tips
- The Dockerfile installs Python packages from `local_vision_requirements.txt`. If your server requires `torch` (very likely for CPU inference with Hugging Face Transformers), either add an appropriate `torch` wheel to the requirements file or uncomment the optional `pip install torch` line in the `Dockerfile` and rebuild.
- The `.dockerignore` file excludes `.venv` and other local artifacts so that local virtualenv packages are not copied into the image.
- To supply a different model, set the `MODEL_NAME` env var when running the container.

Example with a model override:

```bash
docker run --rm -p 8001:8001 -e MODEL_NAME=google/vit-base-patch16-224 local-vision-server:latest
```
