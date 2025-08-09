# ComfyUI (nightly) + Qwen-Image + auto-download LoRA
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# --- system deps ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git wget ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- env/layout ---
ENV COMFY_ROOT=/opt/ComfyUI
WORKDIR $COMFY_ROOT

# --- clone ComfyUI ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# --- Python setup ---
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt && \
    pip install huggingface_hub jupyterlab

# Optional external mounting
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models

# --- Qwen-Image env vars ---
ENV QWEN_AUTO_DOWNLOAD=1 \
    QWEN_VARIANT=fp8 \
    HF_TOKEN=""

# --- JupyterLab token/env ---
ENV JUPYTER_PORT=8888 \
    JUPYTER_TOKEN=runpod

# --- Entrypoint ---
RUN mkdir -p /usr/local/bin && cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
MODELS_DIR="$COMFY_ROOT/models"
mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" "$MODELS_DIR/loras"

# Download Qwen-Image core assets if configured
if [ "${QWEN_AUTO_DOWNLOAD:-1}" = "1" ]; then
  "$COMFY_ROOT/venv/bin/python" <<'PYQ'
import os
from huggingface_hub import hf_hub_download
repo = "Comfy-Org/Qwen-Image_ComfyUI"
variants = {
  "fp8": ("split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
  # add others if needed
}
mode = os.environ.get("QWEN_VARIANT","fp8")
dm_path, te_path = variants.get(mode, variants["fp8"])
base = os.path.join(os.environ["COMFY_ROOT"], "models")
for src, subdir in [(dm_path, "diffusion_models"), (te_path, "text_encoders"), ("split_files/vae/qwen_image_vae.safetensors", "vae")]:
    os.makedirs(os.path.join(base, subdir), exist_ok=True)
    print(f"Downloading {src} …")
    hf_hub_download(repo_id=repo, filename=src, local_dir=os.path.join(base, subdir), local_dir_use_symlinks=False, token=os.environ.get("HF_TOKEN"))
PYQ
fi

# Download Helen LoRA from Hugging Face
"$COMFY_ROOT/venv/bin/python" <<'PYL'
import os
from huggingface_hub import hf_hub_download
repo_id = "momo1231231/HelenLora"
filename = "helendog.safetensors"
outdir = os.path.join(os.environ["COMFY_ROOT"], "models", "loras")
os.makedirs(outdir, exist_ok=True)
print(f"Downloading LoRA: {filename}")
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=outdir,
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN"),
)
print("LoRA downloaded.")
PYL

# Start JupyterLab
"$COMFY_ROOT/venv/bin/python" -m jupyterlab \
  --ip=0.0.0.0 --port="${JUPYTER_PORT:-8888}" --ServerApp.token="${JUPYTER_TOKEN:-}" \
  --ServerApp.password='' --no-browser --allow-root &

# Start ComfyUI
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port 8188
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8188 8888
CMD ["/usr/local/bin/entrypoint.sh"]
