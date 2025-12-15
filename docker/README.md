# Docker Deployment Guide

## Pre-Built Image (Recommended)

**GitHub Container Registry** automatically builds the image on every commit.

### Quick Start

```bash
# Pull pre-built image (~15 GB)
docker pull ghcr.io/zynerji/ivhl:hierarchical

# Run with GPU
docker run --gpus all \
  -p 8080:8080 \
  -p 8000:8000 \
  -v $(pwd)/results:/results \
  ghcr.io/zynerji/ivhl:hierarchical

# Access web interface
open http://localhost:8080/
```

**Image includes**:
- ✅ NVIDIA CUDA 12.5
- ✅ PyTorch with GPU support
- ✅ vLLM + Qwen2.5-2B model (~5GB, pre-downloaded)
- ✅ PyVista for GPU rendering
- ✅ All Python dependencies
- ✅ LaTeX for whitepaper generation

**Total size**: ~15 GB (includes model weights)

---

## Build Locally (Alternative)

If you want to build from source:

```bash
# Clone repository
git clone https://github.com/Zynerji/iVHL.git
cd iVHL

# Build image (20-30 minutes)
docker build -t ivhl-hierarchical -f docker/Dockerfile .

# Run
docker run --gpus all \
  -p 8080:8080 \
  -p 8000:8000 \
  -v $(pwd)/results:/results \
  ivhl-hierarchical
```

**Build time breakdown**:
- CUDA base image download: 5 min (6GB)
- Python packages install: 10 min
- Qwen2.5-2B model download: 10 min (5GB)
- Final image assembly: 5 min

---

## Configuration

### Environment Variables

```bash
docker run --gpus all \
  -e IVHL_ACCESS_TOKEN=secret123 \     # Optional: Add authentication
  -e SIMULATION=hierarchical \         # Simulation type
  -e LOG_LEVEL=info \                  # Logging verbosity
  -p 8080:8080 -p 8000:8000 \
  -v $(pwd)/results:/results \
  ghcr.io/zynerji/ivhl:hierarchical
```

### Port Mapping

- `8080`: Web monitoring interface (WebSocket + HTTP)
- `8000`: vLLM API endpoint (Qwen2.5-2B)

### Volume Mounts

- `/results`: Simulation outputs (frames, whitepaper PDF, metrics JSON)
- `/app/configs`: Custom configuration files (optional)

---

## GPU Requirements

**Minimum**: NVIDIA GPU with 20GB VRAM (works but limited quality)
**Recommended**: H100 (80GB) or H200 (141GB)

**Auto-scaling**:
The container automatically detects your GPU and adjusts simulation parameters:

| GPU | VRAM | Config |
|-----|------|--------|
| H200 | 141GB | 128³ grid, ultra quality, 1000 timesteps |
| H100 | 80GB | 64³ grid, high quality, 500 timesteps |
| A100 | 40GB | 32³ grid, medium quality, 300 timesteps |
| RTX 4090 | 24GB | 16³ grid, low quality, 100 timesteps |

**CPU Fallback**: If no GPU detected, runs on CPU (very slow, minimal config)

---

## Troubleshooting

### Image won't pull
```bash
# Make sure you're authenticated to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

### GPU not detected
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.5.0-base nvidia-smi

# If fails, install nvidia-container-toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### vLLM fails to start
```bash
# Check logs
docker logs <container_id>

# Common issue: Insufficient VRAM
# Solution: Image will automatically disable LLM if <20GB available
```

### Port already in use
```bash
# Use different ports
docker run --gpus all \
  -p 9080:8080 \   # Web UI on 9080 instead
  -p 9000:8000 \   # vLLM on 9000 instead
  ghcr.io/zynerji/ivhl:hierarchical
```

---

## Manual Control

### Start vLLM separately (for debugging)
```bash
# Inside container
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-2B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.6
```

### Run simulation without Docker
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python simulations/hierarchical_dynamics/run_simulation.py --device cuda

# Start web server
python -m uvicorn web_monitor.streaming_server:app --host 0.0.0.0 --port 8080
```

---

## Image Tags

| Tag | Description |
|-----|-------------|
| `hierarchical` | Latest stable (recommended) |
| `hierarchical-YYYYMMDD-HHmmss` | Timestamped builds |
| `hierarchical-sha-<commit>` | Specific commit |

**Pin to specific version for reproducibility**:
```bash
docker pull ghcr.io/zynerji/ivhl:hierarchical-20251215-143022
```

---

## Development

### Rebuild after code changes
```bash
git pull origin main
docker build -t ivhl-hierarchical -f docker/Dockerfile .
```

### Override entrypoint for debugging
```bash
docker run --gpus all -it \
  --entrypoint /bin/bash \
  ghcr.io/zynerji/ivhl:hierarchical
```

---

## Security Notes

1. **Network exposure**: The web interface has no authentication by default
   - Add `IVHL_ACCESS_TOKEN` for basic protection
   - Use firewall rules to restrict access

2. **LLM API**: vLLM endpoint on port 8000 is public
   - Consider not exposing `-p 8000:8000` if not needed externally

3. **Results volume**: Contains simulation data
   - Review before sharing (may include intermediate outputs)

---

## Performance Benchmarks

**H100 (80GB VRAM)**:
- Simulation: ~2-3 minutes (500 timesteps)
- GPU utilization: 75-85%
- VRAM usage: 68-72GB
- Frame rendering: 30 FPS (no frame drops)
- Whitepaper generation: ~30 seconds

**A100 (40GB VRAM)**:
- Simulation: ~4-5 minutes (300 timesteps)
- GPU utilization: 80-90%
- VRAM usage: 35-38GB
- Frame rendering: 30 FPS
- Whitepaper generation: ~45 seconds

**CPU (fallback)**:
- Simulation: ~45-60 minutes (50 timesteps)
- CPU utilization: 90-100% (all cores)
- RAM usage: ~4-6GB
- Rendering: Disabled (too slow)

---

**Last Updated**: 2025-12-15
**Dockerfile Version**: 1.0
**Image Registry**: ghcr.io/zynerji/ivhl
