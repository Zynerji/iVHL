# iVHL VM Quick Start Guide

Quick deployment guide for running iVHL Hierarchical Dynamics simulation on a remote VM with GPU.

---

## Prerequisites

- **VM with GPU**: H100/H200 recommended (or H100/A100 40GB+ VRAM)
- **SSH access**: Username, IP, and SSH key
- **Base Docker image**: PyTorch/Jupyter image with CUDA support

---

## Deployment Steps

### 1. Connect to VM

```bash
# SSH into VM
ssh -i /path/to/ssh_key username@VM_IP
```

### 2. Clone Repository

```bash
# Clone iVHL repository
git clone https://github.com/Zynerji/iVHL.git
cd iVHL
```

### 3. Install Python Dependencies

If using PyTorch/Jupyter base image, install only iVHL-specific requirements:

```bash
# Install requirements
pip install -r requirements.txt
```

If starting from scratch:

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install iVHL dependencies
pip install -r requirements.txt
```

### 4. Run Startup Validation

Verify your environment before launching:

```bash
# Run validation checks
python -m ivhl.utils.startup_validator

# Or with strict mode (exits on critical failures)
python -m ivhl.utils.startup_validator --strict
```

**Expected output:**
```
=========================================================
iVHL Framework Startup Validation
=========================================================

🔍 Critical Checks:
  ✅ Python Version: Python 3.10.12
  ✅ NumPy: numpy available
  ✅ PyTorch: torch available
  ✅ SciPy: scipy available
  ✅ Results Directory: /results is writable

🔧 Optional Checks:
  ✅ GPU: GPU available: NVIDIA H100 80GB (80.0 GB VRAM)
  ⚠️  vLLM Server: vLLM server unreachable (will start later)
  ✅ pdflatex: pdflatex available
  ✅ FastAPI: fastapi available
  ✅ PyVista: pyvista available
  ✅ Matplotlib: matplotlib available

=========================================================
Validation Summary
=========================================================
✅ Passed: 10
❌ Failed: 0
⚠️  Warnings: 1
```

### 5. Start vLLM Server (Optional, for LLM monitoring)

If you have >20GB free VRAM:

```bash
# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 4096 \
  &

# Wait for server to be ready (30 seconds)
sleep 30

# Test vLLM is running
curl http://localhost:8000/v1/models
```

If VRAM is limited, skip this step. The simulation will run in LLM-offline mode.

### 6. Start Web Monitoring Server

```bash
# Start FastAPI server
python -m uvicorn web_monitor.streaming_server:app \
  --host 0.0.0.0 \
  --port 8080 \
  --log-level info \
  &

# Server should be accessible at http://VM_IP:8080
```

### 7. Run Simulation

```bash
# Run hierarchical dynamics simulation
python simulations/hierarchical_dynamics/run_simulation.py \
  --device cuda \
  --timesteps 500 \
  --output-dir /results/hierarchical_run_1
```

**Optional flags:**
- `--device cpu` - Force CPU mode
- `--timesteps N` - Number of simulation steps
- `--base-dim N` - Base tensor dimension
- `--bond-dim N` - Bond dimension for compression
- `--checkpoint-every N` - Save checkpoint every N steps
- `--no-llm` - Disable LLM monitoring (offline mode)
- `--markdown-only` - Skip PDF, generate Markdown report only

---

## Accessing the Web Interface

Once the web server is running:

1. **Open browser**: Navigate to `http://VM_IP:8080/monitor`
2. **View simulation**: Real-time WebSocket stream (30 FPS)
3. **Ask questions**: Chat with Qwen2.5-2B about simulation (if LLM enabled)
4. **Download results**: Whitepaper PDF/MD and metrics JSON in `/results`

---

## Fault Tolerance Features

All 6 SPOFs have been fixed:

### ✅ SPOF #1: Checkpointing
- Automatic checkpoints saved every N steps
- Resume from checkpoint on failure:
  ```bash
  python simulations/hierarchical_dynamics/run_simulation.py \
    --resume /results/checkpoints/latest.ckpt
  ```

### ✅ SPOF #2: GPU OOM Handling
- Automatic memory monitoring
- Dynamic parameter downscaling on OOM
- CPU fallback after 3 OOM errors

### ✅ SPOF #3: LLM Fallback
- Offline rule-based analysis if vLLM unavailable
- Simulation continues without LLM
- Whitepaper still generated (template-based)

### ✅ SPOF #4: Numerical Stability
- NaN/Inf detection and auto-fixing
- Safe division/log/sqrt operations
- Simulation halts after 10 numerical failures (prevent garbage results)

### ✅ SPOF #5: Report Generation
- PDF generation via pdflatex (preferred)
- Markdown fallback if LaTeX unavailable
- Report always generated, format may vary

### ✅ SPOF #6: Startup Validation
- Pre-flight checks before simulation launch
- Clear error messages for missing dependencies
- Degraded configuration auto-generated

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# If missing, install NVIDIA drivers and CUDA toolkit
# (Varies by OS - see NVIDIA documentation)
```

### vLLM Fails to Start

```bash
# Check available VRAM
nvidia-smi

# If <20GB free, run simulation in offline mode:
python simulations/hierarchical_dynamics/run_simulation.py --no-llm
```

### Port Already in Use

```bash
# Change web server port
python -m uvicorn web_monitor.streaming_server:app \
  --host 0.0.0.0 \
  --port 9080  # Use different port
```

### Simulation Crashes Mid-Run

```bash
# Resume from last checkpoint
python simulations/hierarchical_dynamics/run_simulation.py \
  --resume /results/checkpoints/latest.ckpt
```

### LaTeX Not Installed

Simulation will automatically fall back to Markdown reports. To install LaTeX:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-extra

# macOS
brew install --cask mactex

# Then re-run simulation
```

---

## Performance Benchmarks

Expected performance on different hardware:

| GPU | VRAM | Grid Size | Timesteps | Duration | Output |
|-----|------|-----------|-----------|----------|--------|
| H100 | 80GB | 64³ | 500 | ~2-3 min | PDF + frames |
| A100 | 40GB | 32³ | 300 | ~4-5 min | PDF + frames |
| RTX 4090 | 24GB | 16³ | 100 | ~3-4 min | MD + frames |
| CPU | 16GB RAM | 8³ | 50 | ~45 min | MD only |

---

## Next Steps

After successful first run:

1. **Review Results**: Check `/results` directory
2. **Read Whitepaper**: Open PDF or Markdown report
3. **Analyze Metrics**: Load `metrics.json` for detailed data
4. **Multi-User Setup**: See `Hello_Claude.md` for Option A improvements
5. **Tune Parameters**: Adjust grid size, timesteps for your hardware

---

## Support

- **Issues**: https://github.com/Zynerji/iVHL/issues
- **Documentation**: See `docs/` directory
- **Architecture**: See `Hello_Claude.md` for technical details

---

**Last Updated**: 2025-12-15
**Version**: 1.0 (SPOF fixes applied)
