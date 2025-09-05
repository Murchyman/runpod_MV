# ComfyUI (nightly) + Qwen-Image prerequisites
# New GPUs (e.g., RTX 50xx/Blackwell) need newer CUDA runtime for PyTorch nightlies
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# Metadata labels
LABEL maintainer="ComfyUI Team"
LABEL version="1.1"
LABEL description="ComfyUI with Qwen-Image support and CUDA 12.6 (RTX 5090 compatible)"

# --- system deps ---
# Some CUDA images ship without Ubuntu repos enabled; seed sources.list explicitly
RUN set -eux; \
    echo 'deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse' > /etc/apt/sources.list; \
    echo 'deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list; \
    echo 'deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse' >> /etc/apt/sources.list; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3=3.10.* \
      python3-venv=3.10.* \
      python3-pip=22.0.* \
      git=1:2.34.* \
      curl=7.81.* \
      jq=1.6-* \
      wget=1.21.* \
      ca-certificates=20230311* \
      aria2=1.36.* \
      build-essential=12.9* \
    ; \
    rm -rf /var/lib/apt/lists/*

# --- env/layout ---
ENV COMFY_ROOT=/opt/ComfyUI \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8 \
    QWEN_AUTO_DOWNLOAD=1 \
    HF_TOKEN="" \
    COMFY_PORT=8188 \
    COMFY_HOST=0.0.0.0

WORKDIR $COMFY_ROOT

# --- PyTorch channel knobs ---
# Change at build time: --build-arg TORCH_CHANNEL=nightly|stable, --build-arg TORCH_CUDA_TAG=cu126
ARG TORCH_CHANNEL=nightly
ARG TORCH_CUDA_TAG=cu126
ENV TORCH_INDEX_STABLE="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
    TORCH_INDEX_NIGHTLY="https://download.pytorch.org/whl/nightly/${TORCH_CUDA_TAG}"

# --- clone ComfyUI (nightly / master) ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"

# --- RTX 5090/Blackwell compatibility checks ---
RUN set -eux; \
    echo "Checking NVIDIA driver compatibility for RTX 5090/Blackwell..."; \
    if command -v nvidia-smi >/dev/null 2>&1; then \
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0"); \
        if [ "${DRIVER_VERSION%%.*}" -lt 560 ] 2>/dev/null; then \
            echo "WARNING: RTX 5090 requires NVIDIA driver 560+, detected: $DRIVER_VERSION"; \
            echo "Container may not work properly with RTX 5090"; \
        else \
            echo "Driver version $DRIVER_VERSION is compatible with RTX 5090"; \
        fi; \
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "0.0"); \
        if [ "${COMPUTE_CAP%%.*}" -ge 9 ] 2>/dev/null; then \
            echo "Detected Blackwell architecture (compute $COMPUTE_CAP) - optimizations enabled"; \
        fi; \
    else \
        echo "nvidia-smi not available during build - runtime checks will be performed"; \
    fi

# --- Python venv setup (separate layer for better caching) ---
RUN python3 -m venv $COMFY_ROOT/venv

# --- Install PyTorch (separate layer for better caching) ---
RUN . $COMFY_ROOT/venv/bin/activate && \
    pip install --upgrade pip==23.3.* wheel==0.42.*

RUN . $COMFY_ROOT/venv/bin/activate && \
    if [ "$TORCH_CHANNEL" = "nightly" ]; then \
      pip install --pre --index-url "$TORCH_INDEX_NIGHTLY" torch torchvision; \
    else \
      pip install --index-url "$TORCH_INDEX_STABLE" torch torchvision; \
    fi

# --- Install ComfyUI dependencies (separate layer) ---
RUN . $COMFY_ROOT/venv/bin/activate && \
    pip install -r $COMFY_ROOT/requirements.txt && \
    pip install huggingface_hub==0.19.*

# --- custom nodes ---
# rgthree-comfy: quality-of-life nodes and UI improvements
RUN set -eux; \
    mkdir -p "$COMFY_ROOT/custom_nodes"; \
    git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git "$COMFY_ROOT/custom_nodes/rgthree-comfy"; \
    . "$COMFY_ROOT/venv/bin/activate"; \
    if [ -f "$COMFY_ROOT/custom_nodes/rgthree-comfy/requirements.txt" ]; then \
      pip install --no-cache-dir -r "$COMFY_ROOT/custom_nodes/rgthree-comfy/requirements.txt"; \
    fi

# Create model directories with proper structure
RUN mkdir -p /models \
    "$COMFY_ROOT/models/diffusion_models" \
    "$COMFY_ROOT/models/text_encoders" \
    "$COMFY_ROOT/models/vae" \
    "$COMFY_ROOT/models/loras"

# Create separate download script for better maintainability
RUN cat >/usr/local/bin/download-models.py <<'EOF'
#!/usr/bin/env python3
import os
import sys
from huggingface_hub import hf_hub_download

def download_models():
    """Download required models with proper error handling"""
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
        try:
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
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Warning: Failed to download {filename} from {repo_id}: {e}", file=sys.stderr)
            # Continue with other downloads instead of failing completely

if __name__ == "__main__":
    download_models()
EOF

# Create improved entrypoint script with proper signal handling
RUN cat >/usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Signal handling for graceful shutdown
cleanup() {
    echo "Received shutdown signal, cleaning up..."
    if [ -n "${COMFY_PID:-}" ]; then
        kill -TERM "$COMFY_PID" 2>/dev/null || true
        wait "$COMFY_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# Environment setup
COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_HOST="${COMFY_HOST:-0.0.0.0}"

# RTX 5090/Blackwell runtime validation
echo "=== GPU Compatibility Check ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "0.0")
    MEMORY_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    
    echo "GPU: $GPU_NAME"
    echo "Driver: $DRIVER_VERSION"
    echo "Compute Capability: $COMPUTE_CAP"
    echo "VRAM: ${MEMORY_TOTAL}MB"
    
    # RTX 5090 specific checks
    if echo "$GPU_NAME" | grep -qi "RTX.*50[0-9][0-9]"; then
        echo "Blackwell GPU detected!"
        if [ "${DRIVER_VERSION%%.*}" -lt 560 ] 2>/dev/null; then
            echo "ERROR: RTX 5090 requires NVIDIA driver 560+, found: $DRIVER_VERSION"
            echo "Please update your NVIDIA drivers"
            exit 1
        fi
        if [ "${COMPUTE_CAP%%.*}" -ge 9 ] 2>/dev/null; then
            echo "Blackwell compute capability confirmed - optimizations active"
            # Optimize memory allocation for 32GB VRAM
            if [ "$MEMORY_TOTAL" -gt 30000 ]; then
                export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.9"
                echo "Applied RTX 5090 memory optimizations"
            fi
        fi
    fi
else
    echo "WARNING: nvidia-smi not available - GPU validation skipped"
fi
echo "=== End GPU Check ==="

# Ensure model directories exist
mkdir -p \
  "$COMFY_ROOT/models/diffusion_models" \
  "$COMFY_ROOT/models/text_encoders" \
  "$COMFY_ROOT/models/vae" \
  "$COMFY_ROOT/models/loras"

# Download models if enabled
if [ "${QWEN_AUTO_DOWNLOAD:-1}" = "1" ]; then
    echo "Starting model download..."
    "$COMFY_ROOT/venv/bin/python" /usr/local/bin/download-models.py
    echo "Model download completed"
fi

# Start ComfyUI with proper signal handling
echo "Starting ComfyUI on ${COMFY_HOST}:${COMFY_PORT}"
"$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/main.py" \
    --listen "$COMFY_HOST" \
    --port "$COMFY_PORT" &

COMFY_PID=$!
wait "$COMFY_PID"
EOF

# Make scripts executable
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/download-models.py

# Health check to verify service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${COMFY_PORT}/ || exit 1

# Expose port
EXPOSE 8188

# Use exec form for better signal handling
CMD ["/usr/local/bin/entrypoint.sh"]
