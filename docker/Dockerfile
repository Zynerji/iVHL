# iVHL GPU-Accelerated Simulation Docker Image
# ==============================================
#
# Includes:
# - NVIDIA CUDA 12.5
# - PyTorch with GPU support
# - vLLM for Qwen2.5-2B LLM
# - PyVista for GPU rendering
# - LaTeX for whitepaper generation

FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-dev \
    git \
    wget \
    curl \
    vim \
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

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM for LLM inference
RUN pip3 install vllm

# Install visualization and web frameworks
RUN pip3 install \
    pyvista \
    vtk \
    fastapi \
    uvicorn[standard] \
    websockets \
    pillow \
    matplotlib \
    plotly

# Install scientific computing libraries
RUN pip3 install \
    numpy \
    scipy \
    pyscf \
    networkx \
    sympy \
    pyyaml

# Install additional requirements
RUN pip3 install \
    pytest \
    black \
    h5py

# Create app directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create results directory
RUN mkdir -p /results

# GPU detection and auto-scaling script
COPY docker/gpu_detect_and_scale.py /app/docker/

# Entrypoint script
COPY docker/entrypoint.sh /app/docker/
RUN chmod +x /app/docker/entrypoint.sh

# Expose ports
EXPOSE 8080 8000

# Environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Set entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]
