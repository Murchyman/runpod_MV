# ComfyUI (nightly) + Qwen-Image prerequisites
# CUDA 12.1 runtime; adjust if your host drivers/toolkit differ.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# --- system deps ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl jq wget ca-certificates aria2 build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- env/layout ---
ENV COMFY_ROOT=/opt/ComfyUI
ENV PYTHONUNBUFFERED=1
WORKDIR $COMFY_ROOT

# --- clone ComfyUI NIGHTLY (master) ---
# The Qwen tutorial requires the latest dev build, not a release tag. :contentReference[oaicite:1]{index=1}
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# --- Python venv + CUDA 12.1 PyTorch + ComfyUI deps ---
RUN python3 -m venv $COMFY_ROOT/venv && \
    . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r $COMFY_ROOT/requirements.txt

# Optional: place to mount your big model cache from the host if you prefer
RUN mkdir -p /models && ln -s /models $COMFY_ROOT/models

# --- Qwen-Image auto-download knobs ---
# QWEN_VARIANT: fp8 | bf16 | distill_fp8 | distill_bf16
# Set QWEN_AUTO_DOWNLOAD=0 to skip downloads at container start.
ENV QWEN_AUTO_DOWNLOAD=1 \
    QWEN_VARIANT=fp8

# --- tiny entrypoint that fetches Qwen-Image files if missing, then runs ComfyUI ---
RUN bash -lc 'cat > /usr/local/bin/entrypoint.sh << "EOF"\n\
#!/usr/bin/env bash\n\
set -euo pipefail\n\
COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"\n\
MODELS_DIR="$COMFY_ROOT/models"\n\
mkdir -p \"$MODELS_DIR/diffusion_models\" \"$MODELS_DIR/text_encoders\" \"$MODELS_DIR/vae\"\n\
\n\
# Choose filenames/URLs based on QWEN_VARIANT (default fp8)\n\
variant="${QWEN_VARIANT:-fp8}"\n\
case "$variant" in\n\
  fp8)\n\
    DM_FILE=\"qwen_image_fp8_e4m3fn.safetensors\"\n\
    DM_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors\"\n\
    TE_FILE=\"qwen_2.5_vl_7b_fp8_scaled.safetensors\"\n\
    TE_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors\"\n\
    ;;\n\
  bf16)\n\
    DM_FILE=\"qwen_image_bf16.safetensors\"\n\
    DM_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors\"\n\
    TE_FILE=\"qwen_2.5_vl_7b.safetensors\"\n\
    TE_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors\"\n\
    ;;\n\
  distill_fp8)\n\
    DM_FILE=\"qwen_image_distill_full_fp8.safetensors\"\n\
    DM_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_distill_full_fp8.safetensors\"\n\
    TE_FILE=\"qwen_2.5_vl_7b_fp8_scaled.safetensors\"\n\
    TE_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors\"\n\
    ;;\n\
  distill_bf16)\n\
    DM_FILE=\"qwen_image_distill_full_bf16.safetensors\"\n\
    DM_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_distill_full_bf16.safetensors\"\n\
    TE_FILE=\"qwen_2.5_vl_7b.safetensors\"\n\
    TE_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors\"\n\
    ;;\n\
  *) echo \"Unknown QWEN_VARIANT: $variant\" >&2; exit 1 ;;\n\
esac\n\
VAE_FILE=\"qwen_image_vae.safetensors\"\n\
VAE_URL=\"https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors\"\n\
\n\
fetch() {\n\
  local url=\"$1\" out=\"$2\";\n\
  if [ ! -f \"$out\" ]; then\n\
    echo \"Downloading $(basename \"$out\") ...\";\n\
    # aria2c handles the huge files more reliably than wget/curl\n\
    aria2c -x 16 -s 16 -k 10M --console-log-level=warn -o \"$(basename \"$out\")\" -d \"$(dirname \"$out\")\" \"$url\";\n\
  else\n\
    echo \"Found $(basename \"$out\")\";\n\
  fi\n\
}\n\
\n\
if [ \"${QWEN_AUTO_DOWNLOAD:-1}\" = \"1\" ]; then\n\
  fetch \"$DM_URL\" \"$MODELS_DIR/diffusion_models/$DM_FILE\"\n\
  fetch \"$TE_URL\" \"$MODELS_DIR/text_encoders/$TE_FILE\"\n\
  fetch \"$VAE_URL\" \"$MODELS_DIR/vae/$VAE_FILE\"\n\
fi\n\
\n\
# Tip: avoid CUDA allocator fragmentation with giant models\n\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128\n\
\n\
exec \"$COMFY_ROOT/venv/bin/python\" \"$COMFY_ROOT/main.py\" --listen 0.0.0.0 --port 8188\n\
EOF\n\
chmod +x /usr/local/bin/entrypoint.sh'

EXPOSE 8188
CMD ["/usr/local/bin/entrypoint.sh"]
