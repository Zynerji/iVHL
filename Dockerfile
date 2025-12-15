# iVHL Framework - NVIDIA H100 Docker Container
# Optimized for remote deployment with full GPU acceleration
#
# Build: docker build -t ivhl-h100:latest .
# Run: docker run --gpus all -p 8501:8501 ivhl-h100:latest
#
# Author: iVHL Framework
# Date: 2025-12-15

# Base image: NVIDIA CUDA 12.5 with cuDNN 9 for H100 (compute capability 9.0)
FROM nvidia/cuda:12.5.1-cudnn9-devel-ubuntu22.04

# Metadata
LABEL maintainer="iVHL Framework"
LABEL description="iVHL Holographic Resonance Framework with H100 GPU Support"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="9.0" \
    FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Python
    python3.11 \
    python3.11-dev \
    python3-pip \
    # Scientific libraries
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    # Visualization dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # X11 for remote visualization
    xvfb \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (compatible with H100)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core scientific Python packages
# scipy includes signal processing for GW lattice analysis (FFT, curve fitting)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    matplotlib==3.8.2 \
    plotly==5.18.0 \
    scikit-learn==1.3.2

# Install quantum chemistry and physics packages
RUN pip install --no-cache-dir \
    pyscf==2.5.0 \
    qiskit==0.45.2 \
    qutip==4.7.3

# Install visualization packages
RUN pip install --no-cache-dir \
    pyvista==0.43.1 \
    vtk==9.3.0 \
    trame==3.5.3 \
    trame-vuetify==2.4.2 \
    trame-vtk==2.8.2

# Install Streamlit and web frameworks
RUN pip install --no-cache-dir \
    streamlit==1.29.0 \
    streamlit-aggrid==0.3.4.post3 \
    streamlit-plotly-events==0.0.6

# Install additional ML/RL packages
RUN pip install --no-cache-dir \
    tensorboard==2.15.1 \
    gym==0.26.2 \
    stable-baselines3==2.2.1

# Install LaTeX for automated PDF report generation
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy iVHL framework files
COPY . /app/

# Create necessary directories
RUN mkdir -p \
    /app/checkpoints \
    /app/logs \
    /app/exports \
    /app/data \
    /app/results

# Set permissions
RUN chmod -R 755 /app

# Expose LLM chat port
EXPOSE 7860

# Expose Streamlit port
EXPOSE 8501

# Expose trame visualization port (optional)
EXPOSE 8080

# Download default LLM model during build (so it's ready at runtime)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
    cache_dir='/app/.cache/huggingface', \
    ignore_patterns=['*.msgpack', '*.h5'])" || echo "Model download failed - will download at runtime"

# Health check (check both LLM and Streamlit)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD (curl --fail http://localhost:7860/health || true) && \
        (curl --fail http://localhost:8501/_stcore/health || true)

# Set entrypoint to LLM-first startup script
ENTRYPOINT ["python", "scripts/start_with_llm.py"]

# Default command: start both LLM and simulations
CMD ["--full", "--llm-port", "7860", "--sim-port", "8501"]
