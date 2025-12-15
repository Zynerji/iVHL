#!/bin/bash
# iVHL Docker Entrypoint
# ======================
#
# Orchestrates:
# 1. GPU detection and auto-scaling
# 2. vLLM server launch (Qwen2.5-2B)
# 3. Simulation execution
# 4. Web monitoring server

set -e

echo "========================================"
echo "iVHL Hierarchical Dynamics Container"
echo "========================================"
echo ""

# 1. Detect GPU and scale parameters
echo "[1/4] Detecting GPU and scaling parameters..."
python3 /app/docker/gpu_detect_and_scale.py

# Load auto-configuration
if [ -f "/app/auto_config.json" ]; then
    echo ""
    echo "Configuration loaded from auto_config.json"
else
    echo "❌ Failed to generate configuration!"
    exit 1
fi

# Extract values from config
LLM_ENABLED=$(python3 -c "import json; print(json.load(open('/app/auto_config.json'))['llm_enabled'])")
DEVICE=$(python3 -c "import json; print(json.load(open('/app/auto_config.json'))['device'])")

# 2. Launch vLLM server (if LLM enabled)
if [ "$LLM_ENABLED" = "True" ]; then
    echo ""
    echo "[2/4] Launching vLLM server with Qwen2.5-2B..."

    # Start vLLM in background
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-2B-Instruct \
        --port 8000 \
        --gpu-memory-utilization 0.6 \
        --max-model-len 4096 \
        &

    VLLM_PID=$!
    echo "   vLLM server started (PID: $VLLM_PID)"

    # Wait for vLLM to be ready
    echo "   Waiting for vLLM to be ready..."
    sleep 30

    # Check if vLLM is responsive
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "   ✅ vLLM server ready!"
    else
        echo "   ⚠️ vLLM may not be fully ready, continuing anyway..."
    fi
else
    echo ""
    echo "[2/4] LLM disabled (CPU mode or insufficient VRAM)"
fi

# 3. Set simulation parameters from auto-config
echo ""
echo "[3/4] Preparing simulation..."

BASE_DIM=$(python3 -c "import json; print(json.load(open('/app/auto_config.json'))['simulation_params']['base_dimension'])")
BOND_DIM=$(python3 -c "import json; print(json.load(open('/app/auto_config.json'))['simulation_params']['bond_dimension'])")
LAYERS=$(python3 -c "import json; print(json.load(open('/app/auto_config.json'))['simulation_params']['num_layers'])")
TIMESTEPS=$(python3 -c "import json; print(json.load(open('/app/auto_config.json'))['simulation_params']['timesteps'])")

echo "   Simulation will run with:"
echo "   - Base dimension: $BASE_DIM"
echo "   - Bond dimension: $BOND_DIM"
echo "   - Layers: $LAYERS"
echo "   - Timesteps: $TIMESTEPS"
echo "   - Device: $DEVICE"

# 4. Start web monitoring server
echo ""
echo "[4/4] Starting web monitoring server..."

cd /app

# Start FastAPI server in background
python3 -m uvicorn web_monitor.streaming_server:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info \
    &

SERVER_PID=$!
echo "   Web server started (PID: $SERVER_PID)"
echo ""
echo "========================================"
echo "✅ All services running!"
echo "========================================"
echo ""
echo "Access web interface at:"
echo "   http://<VM_IP>:8080/monitor"
echo ""
if [ "$LLM_ENABLED" = "True" ]; then
    echo "LLM API available at:"
    echo "   http://<VM_IP>:8000/v1/chat/completions"
    echo ""
fi
echo "Press Ctrl+C to stop all services"
echo ""

# Keep container running
wait
