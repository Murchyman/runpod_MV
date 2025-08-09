# ---------- Stage 1: fetch Qwen-Image files (no CUDA needed) ----------
FROM python:3.11-slim AS weights-builder

RUN pip install --no-cache-dir huggingface_hub

# Download the three weights and FLATTEN them under /artifacts/<subdir>/<file>
RUN mkdir -p /artifacts && \
    python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

repo_id = "Comfy-Org/Qwen-Image_ComfyUI"
targets = [
    ("split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
     "diffusion_models/qwen_image_fp8_e4m3fn.safetensors"),
    ("split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("split_files/vae/qwen_image_vae.safetensors",
     "vae/qwen_image_vae.safetensors"),
]

root = Path("/artifacts")
for src_rel, dest_rel in targets:
    dest = root / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_path = hf_hub_download(repo_id=repo_id, filename=src_rel, repo_type="model")
    shutil.copy2(src_path, dest)
    print(f"Copied {src_path} -> {dest}")
PY

# ---------- Stage 2: runtime (CUDA + ComfyUI + Jupyter) ----------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl jq wget ca-certificates aria2 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Layout
ENV COMFY_ROOT=/opt/ComfyUI
ENV PYTHONUNBUFFERED=1
WORKDIR $COMFY_ROOT

# ComfyUI (nightly / master)
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# Python venv + CUDA 12.1 PyTorch + ComfyUI deps + JupyterLab
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt && \
    pip install jupyterlab && \
    rm -rf /root/.cache/pip

# (Optional) external models mount point; files are also baked in below
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models || true

# Ensure models tree exists inside the image
RUN mkdir -p $COMFY_ROOT/models/diffusion_models \
             $COMFY_ROOT/models/text_encoders \
             $COMFY_ROOT/models/vae

# Copy ONLY the three weights from builder stage to final locations
COPY --from=weights-builder /artifacts/diffusion_models/qwen_image_fp8_e4m3fn.safetensors $COMFY_ROOT/models/diffusion_models/
COPY --from=weights-builder /artifacts/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors $COMFY_ROOT/models/text_encoders/
COPY --from=weights-builder /artifacts/vae/qwen_image_vae.safetensors $COMFY_ROOT/models/vae/

# JupyterLab settings (you can change at runtime with -e)
ENV JUPYTER_PORT=8888 \
    JUPYTER_TOKEN=runpod

# Entrypoint
RUN mkdir -p /usr/local/bin && cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Start JupyterLab in background
echo "Starting JupyterLab on 0.0.0.0:${JUPYTER_PORT}"
"$COMFY_ROOT/venv/bin/jupyter" lab \
  --ip=0.0.0.0 --port="${JUPYTER_PORT}" \
  --ServerApp.token="${JUPYTER_TOKEN}" --ServerApp.password='' \
  --ServerApp.allow_remote_access=True \
  --no-browser --allow-root &

# Start ComfyUI
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port 8188
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8188 8888
CMD ["/usr/local/bin/entrypoint.sh"]
