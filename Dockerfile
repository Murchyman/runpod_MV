# ComfyUI (nightly) + Qwen-Image + JupyterLab
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

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

# --- Python venv + CUDA 12.1 PyTorch + ComfyUI deps + HF hub + JupyterLab ---
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt && \
    pip install huggingface_hub jupyterlab

# Optional: external model mount point
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models

# --- Qwen-Image knobs (change at run-time if you like) ---
ENV QWEN_AUTO_DOWNLOAD=1 \
    QWEN_VARIANT=fp8 \
    HF_TOKEN=""

# --- JupyterLab knobs ---
# Change at runtime with -e JUPYTER_TOKEN=... and/or -e JUPYTER_PORT=...
ENV JUPYTER_PORT=8888 \
    JUPYTER_TOKEN=runpod

# --- write entrypoint ---
RUN mkdir -p /usr/local/bin && cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
MODELS_DIR="$COMFY_ROOT/models"
# Ensure common model dirs exist, including loras
mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" "$MODELS_DIR/loras"

variant="${QWEN_VARIANT:-fp8}"
case "$variant" in
  fp8)
    DM_PATH="split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
    TE_PATH="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    ;;
  bf16)
    DM_PATH="split_files/diffusion_models/qwen_image_bf16.safetensors"
    TE_PATH="split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
    ;;
  distill_fp8)
    DM_PATH="split_files/diffusion_models/qwen_image_distill_full_fp8.safetensors"
    TE_PATH="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    ;;
  distill_bf16)
    DM_PATH="split_files/diffusion_models/qwen_image_distill_full_bf16.safetensors"
    TE_PATH="split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
    ;;
  *) echo "Unknown QWEN_VARIANT: $variant" >&2; exit 1 ;;
esac
VAE_PATH="split_files/vae/qwen_image_vae.safetensors"

if [ "${QWEN_AUTO_DOWNLOAD:-1}" = "1" ]; then
  export HF_TOKEN=${HF_TOKEN:-}
  "$COMFY_ROOT/venv/bin/python" - <<PY
import os
from huggingface_hub import hf_hub_download

repo_id = "Comfy-Org/Qwen-Image_ComfyUI"
targets = [
    ("$DM_PATH", os.path.join("$MODELS_DIR", "diffusion_models")),
    ("$TE_PATH", os.path.join("$MODELS_DIR", "text_encoders")),
    ("$VAE_PATH", os.path.join("$MODELS_DIR", "vae")),
]
for path, outdir in targets:
    os.makedirs(outdir, exist_ok=True)
    print(f"Downloading {path} -> {outdir}")
    hf_hub_download(
        repo_id=repo_id,
        filename=path,
        repo_type="model",
        local_dir=outdir,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN") or None,
        resume_download=True,
    )
PY
fi

# --- start JupyterLab in the background ---
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
echo "Starting JupyterLab on 0.0.0.0:${JUPYTER_PORT}"
"$COMFY_ROOT/venv/bin/jupyter" lab \
  --ip=0.0.0.0 --port="${JUPYTER_PORT}" \
  --ServerApp.token="${JUPYTER_TOKEN}" --ServerApp.password='' \
  --ServerApp.allow_remote_access=True \
  --no-browser --allow-root &

# --- start ComfyUI ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port 8188
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8188 8888
CMD ["/usr/local/bin/entrypoint.sh"]
