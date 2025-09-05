# ComfyUI (nightly) + Qwen-Image prerequisites
# New GPUs (e.g., RTX 50xx/Blackwell) need newer CUDA runtime for PyTorch nightlies
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# --- system deps ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl jq wget ca-certificates aria2 build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- env/layout ---
ENV COMFY_ROOT=/opt/ComfyUI
ENV PYTHONUNBUFFERED=1
WORKDIR $COMFY_ROOT

# --- clone ComfyUI (nightly / master) ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# --- Python venv + CUDA 12.1 PyTorch + ComfyUI deps + HF hub ---
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    # Install PyTorch nightly with CUDA 12.4 to support latest GPUs (e.g., sm_120)
    pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu124 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt && \
    pip install huggingface_hub

# --- custom nodes ---
# rgthree-comfy: quality-of-life nodes and UI improvements
RUN set -eux; \
    mkdir -p "$COMFY_ROOT/custom_nodes"; \
    if [ ! -d "$COMFY_ROOT/custom_nodes/rgthree-comfy/.git" ]; then \
      git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git "$COMFY_ROOT/custom_nodes/rgthree-comfy"; \
    else \
      git -C "$COMFY_ROOT/custom_nodes/rgthree-comfy" pull --ff-only; \
    fi; \
    . "$COMFY_ROOT/venv/bin/activate"; \
    if [ -f "$COMFY_ROOT/custom_nodes/rgthree-comfy/requirements.txt" ]; then \
      pip install --no-cache-dir -r "$COMFY_ROOT/custom_nodes/rgthree-comfy/requirements.txt"; \
    fi

# Optional: external model mount point
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models

# --- knobs ---
ENV QWEN_AUTO_DOWNLOAD=1 \
    HF_TOKEN=""

# --- write entrypoint ---
RUN mkdir -p /usr/local/bin && cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
MODELS_DIR="$COMFY_ROOT/models"
mkdir -p \
  "$MODELS_DIR/diffusion_models" \
  "$MODELS_DIR/text_encoders" \
  "$MODELS_DIR/vae" \
  "$MODELS_DIR/loras"

# fp8 artifacts (UNet/text-encoder); VAE unchanged
DM_PATH="split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
TE_PATH="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_PATH="split_files/vae/qwen_image_vae.safetensors"

if [ "${QWEN_AUTO_DOWNLOAD:-1}" = "1" ]; then
  export HF_TOKEN=${HF_TOKEN:-}
  "$COMFY_ROOT/venv/bin/python" - <<PY
import os
from huggingface_hub import hf_hub_download

MODELS_DIR = os.path.join(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"), "models")
token = os.environ.get("HF_TOKEN") or None

targets = [
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", os.path.join(MODELS_DIR, "diffusion_models")),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", os.path.join(MODELS_DIR, "text_encoders")),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", os.path.join(MODELS_DIR, "vae")),
    # Helen / ppet LoRAs (optional)
    ("momo1231231/helen", "helen.safetensors", os.path.join(MODELS_DIR, "loras")),
    ("momo1231231/ppet", "ppet.safetensors", os.path.join(MODELS_DIR, "loras")),
]

for repo_id, filename, outdir in targets:
    os.makedirs(outdir, exist_ok=True)
    print(f"Downloading {filename} from {repo_id} -> {outdir}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        local_dir=outdir,
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
PY
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port 8188
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8188
CMD ["/usr/local/bin/entrypoint.sh"]
