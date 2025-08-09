# ComfyUI (nightly) + Qwen-Image prerequisites
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

# --- Python venv + CUDA 12.1 PyTorch + ComfyUI deps ---
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt

# Optional: external model mount point
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models

# --- Qwen-Image knobs (change at run-time if you like) ---
ENV QWEN_AUTO_DOWNLOAD=1 \
    QWEN_VARIANT=fp8

# --- write entrypoint (note: EOF must be at column 1, no spaces) ---
RUN mkdir -p /usr/local/bin && cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
MODELS_DIR="$COMFY_ROOT/models"
mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae"

variant="${QWEN_VARIANT:-fp8}"
case "$variant" in
  fp8)
    DM_FILE="qwen_image_fp8_e4m3fn.safetensors"
    DM_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
    TE_FILE="qwen_2.5_vl_7b_fp8_scaled.safetensors"
    TE_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    ;;
  bf16)
    DM_FILE="qwen_image_bf16.safetensors"
    DM_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors"
    TE_FILE="qwen_2.5_vl_7b.safetensors"
    TE_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
    ;;
  distill_fp8)
    DM_FILE="qwen_image_distill_full_fp8.safetensors"
    DM_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_distill_full_fp8.safetensors"
    TE_FILE="qwen_2.5_vl_7b_fp8_scaled.safetensors"
    TE_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
    ;;
  distill_bf16)
    DM_FILE="qwen_image_distill_full_bf16.safetensors"
    DM_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_distill_full_bf16.safetensors"
    TE_FILE="qwen_2.5_vl_7b.safetensors"
    TE_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
    ;;
  *) echo "Unknown QWEN_VARIANT: $variant" >&2; exit 1 ;;
esac

VAE_FILE="qwen_image_vae.safetensors"
VAE_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

fetch() {
  local url="$1" out="$2"
  if [ ! -f "$out" ]; then
    echo "Downloading $(basename "$out") ..."
    aria2c -x 16 -s 16 -k 10M --console-log-level=warn -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  else
    echo "Found $(basename "$out")"
  fi
}

if [ "${QWEN_AUTO_DOWNLOAD:-1}" = "1" ]; then
  fetch "$DM_URL" "$MODELS_DIR/diffusion_models/$DM_FILE"
  fetch "$TE_URL" "$MODELS_DIR/text_encoders/$TE_FILE"
  fetch "$VAE_URL" "$MODELS_DIR/vae/$VAE_FILE"
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port 8188
EOF

# make it executable (separate RUN so the parser can't confuse it)
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8188
CMD ["/usr/local/bin/entrypoint.sh"]
