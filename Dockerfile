# ComfyUI + Qwen-Image-Edit (GPU) Dockerfile
# - Installs ComfyUI (latest), Torch CUDA 12.1, and downloads the Qwen Image Edit models
# - Ships the default Qwen-Image-Edit example workflow PNG for drag-and-drop import
#
# Build:
#   docker build -t comfy-qwen-edit .
# Run (Linux, NVIDIA GPU):
#   docker run --gpus all -it --rm -p 8188:8188 \
#     -e HF_TOKEN=${HF_TOKEN:-} \
#     -v $(pwd)/ComfyUI_user:/opt/ComfyUI/user \
#     comfy-qwen-edit
#
# After it starts, open http://localhost:8188 and (optionally) load the example PNG at:
#   /opt/ComfyUI/user/workflows/qwen_image_edit_basic_example.png
#
# Notes:
# - The model files are downloaded into the standard ComfyUI model folders:
#     models/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors
#     models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
#     models/vae/qwen_image_vae.safetensors
#     models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors
# - You can delete/replace these with newer versions later if desired.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# --- system deps ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl jq wget ca-certificates aria2 ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# --- env/layout ---
ENV COMFY_ROOT=/opt/ComfyUI \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf_cache \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR $COMFY_ROOT

# --- clone ComfyUI (latest) ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# --- Python venv + PyTorch CUDA 12.1 ---
RUN python3 -m venv "$COMFY_ROOT/venv" && \
    . "$COMFY_ROOT/venv/bin/activate" && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1+cu121 torchvision==0.18.1+cu121 xformers==0.0.26.post1 && \
    pip install -r "$COMFY_ROOT/requirements.txt" \
                huggingface_hub>=0.23.0 \
                safetensors>=0.4.3

# --- (optional) ComfyUI-Manager ---
RUN mkdir -p "$COMFY_ROOT/custom_nodes" && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git "$COMFY_ROOT/custom_nodes/ComfyUI-Manager" || true

# --- model dirs ---
RUN mkdir -p "$COMFY_ROOT/models/diffusion_models" \
             "$COMFY_ROOT/models/text_encoders" \
             "$COMFY_ROOT/models/vae" \
             "$COMFY_ROOT/models/loras" \
             "$COMFY_ROOT/user/workflows"

# --- download models (FP8 edit model, text encoder, VAE, optional Lightning LoRA) ---
# These are the official Comfy-Org mirrors that match the ComfyUI folder layout.
# You may provide HF_TOKEN at runtime for higher rate limits.
RUN set -eux; \
    cd "$COMFY_ROOT/models/diffusion_models" && \
    aria2c -x 16 -s 16 -k 1M -o qwen_image_edit_fp8_e4m3fn.safetensors \
      "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"; \
    cd "$COMFY_ROOT/models/text_encoders" && \
    aria2c -x 16 -s 16 -k 1M -o qwen_2.5_vl_7b_fp8_scaled.safetensors \
      "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"; \
    cd "$COMFY_ROOT/models/vae" && \
    aria2c -x 16 -s 16 -k 1M -o qwen_image_vae.safetensors \
      "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"; \
    cd "$COMFY_ROOT/models/loras" && \
    aria2c -x 16 -s 16 -k 1M -o Qwen-Image-Lightning-4steps-V1.0.safetensors \
      "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors" || true


# --- runtime ---
EXPOSE 8188

# Start ComfyUI
CMD ["/bin/bash", "-lc", ". /opt/ComfyUI/venv/bin/activate && python /opt/ComfyUI/main.py --listen 0.0.0.0 --port 8188"]
