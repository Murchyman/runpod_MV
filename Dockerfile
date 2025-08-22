# ComfyUI (nightly) + Qwen-Image-Edit
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ---- system deps ----
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git git-lfs curl jq wget ca-certificates aria2 build-essential \
 && rm -rf /var/lib/apt/lists/*

ENV COMFY_ROOT=/opt/ComfyUI
ENV PYTHONUNBUFFERED=1
WORKDIR $COMFY_ROOT

# ---- clone ComfyUI (nightly / master) ----
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# ---- Python venv + CUDA 12.1 wheels ----
RUN python3 -m venv "$COMFY_ROOT/venv" && \
    "$COMFY_ROOT/venv/bin/pip" install --upgrade pip setuptools wheel && \
    "$COMFY_ROOT/venv/bin/pip" install \
      torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
      --extra-index-url https://download.pytorch.org/whl/cu121 && \
    "$COMFY_ROOT/venv/bin/pip" install -r "$COMFY_ROOT/requirements.txt" && \
    "$COMFY_ROOT/venv/bin/pip" install huggingface_hub>=0.25.0 safetensors

# ---- model folders ----
RUN mkdir -p \
  "$COMFY_ROOT/models/diffusion_models" \
  "$COMFY_ROOT/models/text_encoders" \
  "$COMFY_ROOT/models/vae" \
  "$COMFY_ROOT/models/loras"

# Optional: pass a token at runtime:  docker run -e HF_TOKEN=...
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# ---- Entrypoint: fetch Qwen-Image-Edit + deps if missing, then launch ----
RUN set -eux; mkdir -p /usr/local/bin && cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$COMFY_ROOT/venv/bin/activate"

DIFF="$COMFY_ROOT/models/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"
CLIP="$COMFY_ROOT/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE="$COMFY_ROOT/models/vae/qwen_image_vae.safetensors"
LORA="$COMFY_ROOT/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors"

if [ ! -f "$DIFF" ] || [ ! -f "$CLIP" ] || [ ! -f "$VAE" ] || [ ! -f "$LORA" ]; then
  python - <<PY
import os, shutil
from huggingface_hub import hf_hub_download, login

token = os.getenv("HF_TOKEN") or None
if token: 
    try: login(token=token); 
    except Exception: pass

COMFY_ROOT = os.environ["COMFY_ROOT"]

def fetch(repo_id, filename, out_dir, out_name=None):
    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="/root/.cache/hf",
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, out_name or os.path.basename(filename))
    if os.path.abspath(p) != os.path.abspath(dst):
        shutil.copy2(p, dst)

# Qwen-Image-Edit diffusion (FP8)
fetch("Comfy-Org/Qwen-Image-Edit_ComfyUI",
      "split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
      os.path.join(COMFY_ROOT, "models/diffusion_models"))

# Qwen2.5-VL text encoder (FP8)
fetch("Comfy-Org/Qwen-Image_ComfyUI",
      "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
      os.path.join(COMFY_ROOT, "models/text_encoders"))

# Qwen VAE
fetch("Comfy-Org/Qwen-Image_ComfyUI",
      "split_files/vae/qwen_image_vae.safetensors",
      os.path.join(COMFY_ROOT, "models/vae"))

# Optional: Lightning LoRA (4 steps) for speed
fetch("lightx2v/Qwen-Image-Lightning",
      "Qwen-Image-Lightning-4steps-V1.0.safetensors",
      os.path.join(COMFY_ROOT, "models/loras"))
PY
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port 8188
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8188
CMD ["/usr/local/bin/entrypoint.sh"]
