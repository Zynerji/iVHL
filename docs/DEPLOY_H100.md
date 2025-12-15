# H100 Remote Deployment Guide

Complete guide for deploying the iVHL framework on a remote NVIDIA H100 GPU VM using Docker.

---

## Prerequisites

### Remote H100 VM Requirements

- **GPU**: NVIDIA H100 (80GB HBM3 recommended)
- **CUDA**: 12.1+ (driver version 535+)
- **RAM**: 128GB+ recommended
- **Storage**: 500GB+ SSD
- **OS**: Ubuntu 22.04 LTS
- **Docker**: 24.0+ with NVIDIA Container Toolkit

### Local Machine Requirements

- SSH access to H100 VM
- Modern web browser
- Stable internet connection

---

## Quick Start

### 1. Clone Repository on H100 VM

```bash
# SSH into H100 VM
ssh user@h100-vm.example.com

# Clone iVHL repository
git clone https://github.com/Zynerji/iVHL.git
cd iVHL
```

### 2. Build Docker Image

```bash
# Build image (takes 10-15 minutes)
docker build -t ivhl-h100:latest .

# Verify image
docker images | grep ivhl-h100
```

### 3. Run Container

```bash
# Run with GPU support
docker run --gpus all \
  -p 8501:8501 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  ivhl-h100:latest
```

### 4. Access Streamlit Interface

Open browser to: `http://h100-vm.example.com:8501`

---

## Detailed Setup

### Installing Docker on Ubuntu 22.04

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
```

### Installing NVIDIA Container Toolkit

```bash
# Setup NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA runtime
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Verify H100 Detection

```bash
# Check GPU
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.XX       Driver Version: 535.XX       CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA H100 80G...  Off  | 00000000:00:1E.0 Off |                    0 |
```

---

## Docker Build Options

### Standard Build

```bash
docker build -t ivhl-h100:latest .
```

### Build with Custom Tag

```bash
docker build -t ivhl-h100:v1.0.0 .
```

### Build with Build Args

```bash
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg CUDA_VERSION=12.5.1 \
  -t ivhl-h100:latest .
```

### Build with No Cache (clean build)

```bash
docker build --no-cache -t ivhl-h100:latest .
```

---

## Running the Container

### Basic Run

```bash
docker run --gpus all -p 8501:8501 ivhl-h100:latest
```

### Run with Volume Mounts

```bash
docker run --gpus all \
  -p 8501:8501 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  ivhl-h100:latest
```

### Run in Background (Detached)

```bash
docker run -d \
  --name ivhl-container \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  ivhl-h100:latest

# Check logs
docker logs -f ivhl-container

# Stop container
docker stop ivhl-container

# Restart container
docker start ivhl-container

# Remove container
docker rm ivhl-container
```

### Run with Environment Variables

```bash
docker run --gpus all \
  -p 8501:8501 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  -e STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  ivhl-h100:latest
```

### Run with Custom Streamlit App

```bash
docker run --gpus all \
  -p 8501:8501 \
  ivhl-h100:latest \
  custom_app.py --server.port=8501
```

---

## Accessing the Application

### From Local Machine

```bash
# SSH tunnel to H100 VM
ssh -L 8501:localhost:8501 user@h100-vm.example.com

# Open browser to:
http://localhost:8501
```

### From Remote Browser

Configure firewall to allow port 8501:

```bash
# On H100 VM
sudo ufw allow 8501/tcp

# Access from browser:
http://h100-vm.example.com:8501
```

---

## Performance Optimization

### GPU Memory Management

```bash
# Limit GPU memory
docker run --gpus '"device=0"' \
  --memory=64g \
  --memory-swap=128g \
  -p 8501:8501 \
  ivhl-h100:latest
```

### Multi-GPU Support

```bash
# Use all GPUs
docker run --gpus all -p 8501:8501 ivhl-h100:latest

# Use specific GPUs
docker run --gpus '"device=0,1"' -p 8501:8501 ivhl-h100:latest

# Use GPU with specific capabilities
docker run --gpus 'all,capabilities=compute' -p 8501:8501 ivhl-h100:latest
```

### CPU Limits

```bash
docker run --gpus all \
  --cpus=32 \
  --memory=128g \
  -p 8501:8501 \
  ivhl-h100:latest
```

---

## Monitoring

### Container Stats

```bash
# Real-time stats
docker stats ivhl-container

# GPU utilization
nvidia-smi -l 1

# Watch GPU memory
watch -n 1 nvidia-smi
```

### Logs

```bash
# Stream logs
docker logs -f ivhl-container

# Last 100 lines
docker logs --tail 100 ivhl-container

# Since specific time
docker logs --since 2025-12-15T10:00:00 ivhl-container
```

### Health Check

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' ivhl-container

# Manual health check
curl http://localhost:8501/_stcore/health
```

---

## Data Persistence

### Volume Strategy

**Recommended volumes to mount:**
- `/app/checkpoints` - Model checkpoints
- `/app/logs` - Training logs
- `/app/data` - Simulation data
- `/app/results` - Results and exports

```bash
# Create local directories
mkdir -p checkpoints logs data results

# Run with mounts
docker run --gpus all \
  -p 8501:8501 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  ivhl-h100:latest
```

### Backup Strategy

```bash
# Backup volumes
docker run --rm \
  -v ivhl-checkpoints:/checkpoints \
  -v $(pwd)/backup:/backup \
  ubuntu tar czf /backup/checkpoints-$(date +%Y%m%d).tar.gz -C /checkpoints .

# Restore volumes
docker run --rm \
  -v ivhl-checkpoints:/checkpoints \
  -v $(pwd)/backup:/backup \
  ubuntu tar xzf /backup/checkpoints-20251215.tar.gz -C /checkpoints
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, restart Docker
sudo systemctl restart docker

# Verify NVIDIA Container Toolkit
nvidia-container-cli info
```

### Port Already in Use

```bash
# Find process using port 8501
sudo lsof -i :8501

# Kill process
sudo kill -9 <PID>

# Or use different port
docker run --gpus all -p 8502:8501 ivhl-h100:latest
```

### Out of Memory

```bash
# Check available memory
free -h

# Increase swap
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Run with memory limit
docker run --gpus all \
  --memory=64g \
  --memory-swap=128g \
  -p 8501:8501 \
  ivhl-h100:latest
```

### Slow Build

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t ivhl-h100:latest .

# Use cache from previous build
docker build --cache-from ivhl-h100:latest -t ivhl-h100:latest .
```

---

## Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ivhl:
    image: ivhl-h100:latest
    container_name: ivhl-container
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8501:8501"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./data:/app/data
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Security Considerations

### Firewall Configuration

```bash
# Allow only specific IP
sudo ufw allow from 203.0.113.0/24 to any port 8501

# Or use SSH tunnel (recommended)
ssh -L 8501:localhost:8501 user@h100-vm.example.com
```

### SSL/TLS (Production)

Use nginx reverse proxy with SSL:

```bash
# Install nginx
sudo apt-get install nginx

# Configure SSL
sudo certbot --nginx -d h100.example.com
```

Example nginx config:

```nginx
server {
    listen 443 ssl;
    server_name h100.example.com;

    ssl_certificate /etc/letsencrypt/live/h100.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/h100.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Benchmarking

### Test GPU Performance

```bash
# Run PyTorch benchmark
docker exec -it ivhl-container python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')

# Simple benchmark
x = torch.randn(10000, 10000, device='cuda')
%timeit x @ x.T
"

# Run torch.compile benchmark
docker exec -it ivhl-container python benchmarks.py --size large
```

### Expected Performance

**H100 80GB HBM3:**
- Field superposition (8192 grid): ~3-5ms
- MERA contraction (128 tensors): ~8-12ms
- GFT evolution (64³ grid): ~1200-1800ms
- Full hybrid TD3-SAC update: ~15-25ms

---

## Maintenance

### Update Container

```bash
# Pull latest code
git pull origin main

# Rebuild image
docker build -t ivhl-h100:latest .

# Stop old container
docker stop ivhl-container
docker rm ivhl-container

# Run new container
docker run -d --name ivhl-container --gpus all -p 8501:8501 ivhl-h100:latest
```

### Clean Up

```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -f

# Remove all unused data
docker system prune -a -f

# Remove specific image
docker rmi ivhl-h100:latest
```

---

## Production Deployment Checklist

- [ ] H100 VM provisioned with Ubuntu 22.04
- [ ] NVIDIA drivers installed (535+)
- [ ] Docker and NVIDIA Container Toolkit installed
- [ ] Firewall configured
- [ ] SSL certificate configured (if public)
- [ ] Volume directories created
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Docker image built and tested
- [ ] Container running with health checks
- [ ] SSH tunnel or VPN configured
- [ ] Access tested from local machine

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/Zynerji/iVHL/issues
- Documentation: See LOCAL_STREAMLIT_HTML.md for local deployment

---

**Ready to deploy on H100! 🚀**
