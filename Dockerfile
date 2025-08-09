# ---------- Stage 1: fetch Qwen-Image files ----------
FROM python:3.11-slim AS weights-builder

ENV PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/artifacts/.hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "huggingface_hub>=0.23"

# Download only the needed 3 model files; symlinks avoid duplicate copies
RUN mkdir -p /artifacts && python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Comfy-Org/Qwen-Image_ComfyUI",
    repo_type="model",
    local_dir="/artifacts",
    local_dir_use_symlinks=True,
    allow_patterns=[
        "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "split_files/vae/qwen_image_vae.safetensors",
    ],
)
PY

# ---------- Stage 2: runtime (CUDA + ComfyUI) ----------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl jq wget ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Layout
ENV COMFY_ROOT=/opt/ComfyUI \
    PYTHONUNBUFFERED=1
WORKDIR $COMFY_ROOT

# ComfyUI (nightly / master)
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# Python venv + CUDA 12.1 PyTorch + ComfyUI deps
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt && \
    rm -rf /root/.cache/pip

# Optional external mount and in-image tree
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models || true
RUN mkdir -p $COMFY_ROOT/models/diffusion_models \
             $COMFY_ROOT/models/text_encoders \
             $COMFY_ROOT/models/vae

# Copy the 3 weights from Stage 1 into final image
COPY --from=weights-builder /artifacts/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors $COMFY_ROOT/models/diffusion_models/
COPY --from=weights-builder /artifacts/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors $COMFY_ROOT/models/text_encoders/
COPY --from=weights-builder /artifacts/split_files/vae/qwen_image_vae.safetensors $COMFY_ROOT/models/vae/

# Entrypoint — just runs ComfyUI
RUN mkdir -p /usr/local/bin && cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port 8188
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8188
CMD ["/usr/local/bin/entrypoint.sh"]
