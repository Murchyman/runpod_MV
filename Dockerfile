# Latest ComfyUI from source, GPU-ready
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# --- system deps ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl jq ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- where we'll install ---
ENV COMFY_ROOT=/opt/ComfyUI
WORKDIR $COMFY_ROOT

# Build arg:
#   COMFYUI_REF=auto  -> detect latest release via GitHub API at build time
#   COMFYUI_REF=<tag> -> pin to a specific tag, e.g., v0.3.48
ARG COMFYUI_REF=auto

# --- fetch ComfyUI (latest release by default) ---
RUN set -eux; \
    if [ "$COMFYUI_REF" = "auto" ]; then \
      COMFYUI_REF="$(curl -s https://api.github.com/repos/comfyanonymous/ComfyUI/releases/latest | jq -r .tag_name)"; \
    fi; \
    echo "Using ComfyUI ref: $COMFYUI_REF"; \
    git clone --depth 1 --branch "$COMFYUI_REF" https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# --- Python venv + PyTorch (CUDA 12.1) + ComfyUI deps ---
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt

# Optional: a place to mount persistent models into the container
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models

EXPOSE 8188
ENV PYTHONUNBUFFERED=1

# Run ComfyUI on all interfaces (port 8188)
CMD ["/opt/ComfyUI/venv/bin/python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
