# iVHL GPU-Accelerated Simulation Docker Image
# ==============================================

FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (use CUDA 12.1 compatible version, smaller install)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies in batches to avoid timeouts
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    pillow \
    matplotlib

RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    websockets \
    pyyaml \
    h5py

RUN pip3 install --no-cache-dir \
    pyvista \
    vtk \
    plotly

# Install vLLM (may take a while)
RUN pip3 install --no-cache-dir vllm

# Install remaining scientific libraries
RUN pip3 install --no-cache-dir \
    networkx \
    sympy \
    pytest

# Install PySCF (optional, can be slow)
RUN pip3 install --no-cache-dir pyscf || echo "PySCF install failed, skipping"

WORKDIR /app

# Copy application code
COPY ivhl/ /app/ivhl/
COPY web_monitor/ /app/web_monitor/
COPY simulations/ /app/simulations/
COPY docker/gpu_detect_and_scale.py /app/docker/
COPY docker/entrypoint.sh /app/docker/
COPY configs/ /app/configs/

RUN chmod +x /app/docker/entrypoint.sh
RUN mkdir -p /results

EXPOSE 8080 8000

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

ENTRYPOINT ["/app/docker/entrypoint.sh"]
