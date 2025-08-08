FROM runpod/worker-comfyui:5.3.0-sdxl

# Install system packages needed for git and wget (if not already present).
RUN apt-get update && apt-get install -y git wget

# Set working directory to the root of ComfyUI installation.
WORKDIR /comfyui

# Clone the MVAdapter extension into the ComfyUI custom_nodes directory.
# The README recommends placing the repository under ComfyUI/custom_nodes and
# running pip install on its requirements【960072206842080†L18-L26】.
RUN git clone --depth 1 https://github.com/huanngzh/ComfyUI-MVAdapter.git \
    /comfyui/custom_nodes/ComfyUI-MVAdapter

# Install the MVAdapter python dependencies.
# ComfyUI uses its own virtual environment; installing into this environment keeps packages isolated.
RUN pip install -r /comfyui/custom_nodes/ComfyUI-MVAdapter/requirements.txt

# Create a directory for adapter weights and pre‑download the MVAdapter SDXL I2MV beta weight.
# The documentation explains that the Diffusers Model Makeup node should use the
# file mvadapter_i2mv_sdxl_beta.safetensors【960072206842080†L105-L112】.
RUN mkdir -p /comfyui/models/adapters && \
    wget -q -O /comfyui/models/adapters/mvadapter_i2mv_sdxl_beta.safetensors \
    https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_i2mv_sdxl_beta.safetensors

# Optionally download additional VAE models recommended for limited GPU memory.
# If you have a smaller GPU, the MVAdapter guide suggests using the FP16 SDXL VAE fix【960072206842080†L44-L58】.
# The base worker image already includes the SDXL checkpoint and VAEs, but the fix can be added here.
RUN wget -q -O /comfyui/models/vae/sdxl-vae-fp16-fix.safetensors \
    https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors

# Copy the example workflow that uses the View Selector.
# This workflow demonstrates how to set up an image‑to‑multi‑view pipeline and select front/back/left/right views.
# Users can load this workflow directly in the ComfyUI interface.
RUN mkdir -p /comfyui/workflows && \
    wget -q -O /comfyui/workflows/i2mv_sdxl_ldm_view_selector.json \
    https://raw.githubusercontent.com/huanngzh/ComfyUI-MVAdapter/main/workflows/i2mv_sdxl_ldm_view_selector.json

# Set the default command to start the ComfyUI worker.
# The parent image includes a start script that runs ComfyUI with the appropriate environment.
CMD ["/start.sh"]