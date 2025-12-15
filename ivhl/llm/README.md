# iVHL Embedded LLM System

**Autonomous AI assistant running on H100 alongside simulations**

## Overview

The iVHL framework now includes an embedded small language model that:
- ✅ Monitors simulations in real-time
- ✅ Answers questions about simulation state
- ✅ Modifies parameters via natural language
- ✅ Generates white papers autonomously (using H100, not Claude API)
- ✅ Provides interactive chat interface
- ✅ Runs completely offline in Docker container

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container (H100)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌────────────────────────────────┐   │
│  │   LLM        │────▶│  Simulation Manager            │   │
│  │  (15% GPU)   │     │  (80% GPU)                     │   │
│  │              │     │                                 │   │
│  │ TinyLlama/   │     │  • Holographic Resonance       │   │
│  │ Phi-3 /      │     │  • GFT Dynamics                │   │
│  │ Llama-3.2    │     │  • MERA Construction           │   │
│  └──────────────┘     │  • GW Analysis                 │   │
│         │              └────────────────────────────────┘   │
│         │                          │                        │
│         ▼                          ▼                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           WebSocket Communication                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │  Web Browser  │
                   │  (User Chat)  │
                   └───────────────┘
```

## Key Features

### 1. LLM-First Startup

**CRITICAL**: The LLM loads FIRST and reserves GPU memory BEFORE simulations start.

```bash
# Recommended startup (handles everything)
python scripts/start_with_llm.py --full

# This ensures:
# 1. LLM loads and reserves 15% of H100 (12 GB)
# 2. Simulations use remaining 80% (64 GB)
# 3. No memory conflicts or OOM errors
```

### 2. Supported Models

| Model | Size | Memory | Speed | Quality |
|-------|------|--------|-------|---------|
| **TinyLlama-1.1B** | 1.1B | ~4 GB | Very Fast | Basic |
| **Phi-3-mini-4k** | 3.8B | ~8 GB | Fast | Good |
| **Llama-3.2-3B** | 3B | ~6 GB | Fast | Good |
| **Mistral-7B-Instruct-v0.2** | 7B (4-bit) | ~5 GB | Medium | Excellent |

**Default**: TinyLlama (fastest, leaves most memory for simulations)

### 3. Function Calling / Tool Use

The LLM can call these functions to control simulations:

```python
# Available tools:
- get_simulation_state()        # Check current status
- set_parameter(name, value)    # Modify parameters
- pause_simulation()            # Pause execution
- resume_simulation()           # Resume execution
- get_metrics_history(metric)   # Get historical data
- generate_white_paper()        # Create PDF report
- get_available_parameters()    # List modifiable params
```

### 4. Natural Language Control

**Example interactions**:

```
User: What's the current simulation status?
LLM:  The simulation is running at timestep 342/1000 (34% complete).
      Current metrics: correlation=0.95, fractal_dimension=2.1

User: The correlation seems high. Set GW amplitude to 5e-21
LLM:  [Calls: set_parameter("gw_amplitude", 5e-21)]
      Parameter updated. GW amplitude is now 5e-21.

User: Generate a white paper for this run
LLM:  [Calls: generate_white_paper()]
      White paper generated at: /app/whitepapers/report_20251215_153000/

User: What parameters can I change?
LLM:  [Calls: get_available_parameters()]
      Available parameters:
      - num_lattice_nodes (int): Number of boundary lattice nodes
      - gw_amplitude (float): Gravitational wave strain amplitude
      - gw_frequency (float): GW frequency in Hz
      - perturbation_type (string): Type of GW perturbation
      - temperature (float): RL exploration temperature
```

### 5. Autonomous White Paper Generation

The LLM generates white papers directly on the H100 (not using Claude API):

```python
from ivhl.llm.agent import iVHLAgent, LLMConfig

agent = iVHLAgent(config)

# Generate white paper from simulation data
latex_source = agent.generate_white_paper_with_llm({
    'parameters': {...},
    'results': {...},
    'analysis': {...}
})

# Save and compile
with open('whitepaper.tex', 'w') as f:
    f.write(latex_source)

# Compile in Docker (has pdflatex)
subprocess.run(['pdflatex', 'whitepaper.tex'])
```

### 6. Real-Time Monitoring

The LLM continuously monitors simulations and alerts on issues:

```python
# Start monitoring
await agent.monitor_simulation()

# Automatic alerts on:
# - Field divergence
# - High NaN fraction
# - Correlation drops
# - Numerical instability
```

## Usage

### Quick Start (Docker)

```bash
# Build container with LLM
docker build -t ivhl-llm:latest .

# Run with LLM-first startup
docker run --gpus all \
  -p 7860:7860 \
  -p 8501:8501 \
  -v $(pwd)/whitepapers:/app/whitepapers \
  ivhl-llm:latest \
  python scripts/start_with_llm.py --full

# Access:
# - LLM Chat: http://localhost:7860
# - Simulation Dashboard: http://localhost:8501
```

### Local Development

```bash
# Install dependencies
pip install vllm gradio huggingface_hub

# Start LLM server only
python scripts/start_with_llm.py --llm-only --port 7860

# Start simulations separately
streamlit run dashboards/resonance_dashboard.py
```

### Custom Configuration

```python
from ivhl.llm.agent import iVHLAgent, LLMConfig

# Configure LLM
config = LLMConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",  # Better quality
    max_tokens=2048,
    temperature=0.7,
    gpu_memory_utilization=0.2,  # 20% of GPU for LLM
    monitor_interval=30.0,  # Check sim every 30s
    auto_generate_papers=True
)

# Initialize agent
agent = iVHLAgent(config, simulation_manager=my_sim)

# Chat
response = agent.chat("What's the fractal dimension?")
```

## Memory Management

### H100 80GB Allocation

| Component | Memory | Percentage |
|-----------|--------|------------|
| **LLM** (TinyLlama) | 4-8 GB | 5-10% |
| **LLM** (Phi-3) | 8-12 GB | 10-15% |
| **Reserved** | 4 GB | 5% |
| **Simulations** | 64-68 GB | 80-85% |

**Total**: Never exceeds 100%

### Startup Sequence

```
1. [0s] Load LLM → reserves 12 GB (Phi-3)
2. [10s] LLM chat server starts → port 7860
3. [15s] Calculate remaining memory → 68 GB available
4. [15s] Start simulations → use 64 GB budget
5. [20s] System ready
```

## Chat Interface

### Gradio UI

![LLM Chat Interface](assets/llm_chat_screenshot.png)

Features:
- Real-time chat with markdown rendering
- Code block syntax highlighting
- Example prompt buttons
- Simulation status panel (live updates)
- White paper generation button
- Copy/export conversation

### WebSocket API

For custom integrations:

```javascript
// Connect to LLM WebSocket
const ws = new WebSocket('ws://localhost:7860/ws');

// Send message
ws.send(JSON.stringify({
  type: 'chat',
  message: 'What is the current correlation?'
}));

// Receive response
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.response);
};
```

## White Paper Generation

### Autonomous Mode

```python
# Enable auto-generation
config = LLMConfig(auto_generate_papers=True)
agent = iVHLAgent(config)

# Papers automatically generated when:
# - Simulation completes
# - User requests via chat
# - Monitoring detects interesting phenomena
```

### Manual Mode

```python
# Generate on demand
latex_source = agent.generate_white_paper_with_llm(simulation_data)

# Contains:
# - Title, abstract, intro
# - Methods with equations
# - Results tables
# - Analysis and discussion
# - Conclusions
# - References (Maldacena, LIGO, etc.)
```

## Performance

### Inference Speed

| Model | Tokens/sec | Latency (avg) |
|-------|------------|---------------|
| TinyLlama | 200-300 | ~50ms |
| Phi-3 | 100-150 | ~100ms |
| Llama-3.2 | 80-120 | ~120ms |
| Mistral-7B (4-bit) | 40-60 | ~250ms |

*On H100 80GB with vLLM*

### Memory Overhead

Running LLM alongside simulations adds:
- **Memory**: 4-12 GB (depending on model)
- **Compute**: <5% overhead (simulations still use 95%+ of GPU)
- **Network**: Minimal (WebSocket is lightweight)

## Troubleshooting

### OOM (Out of Memory) Errors

```bash
# Reduce LLM memory allocation
python scripts/start_with_llm.py --llm-memory 0.10

# Or use smaller model
python scripts/start_with_llm.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Or run LLM on CPU (slow)
export CUDA_VISIBLE_DEVICES=""
python dashboards/llm_chat.py
```

### LLM Not Responding

```bash
# Check if vLLM is installed
pip install vllm

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check logs
tail -f logs/llm_server.log
```

### Slow Inference

```bash
# Use quantized model
python scripts/start_with_llm.py --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# Enable torch.compile (faster after warmup)
export TORCH_COMPILE=1
python scripts/start_with_llm.py
```

## Advanced Features

### Custom System Prompt

```python
agent.system_prompt = """You are a quantum gravity expert.
Focus on GFT, tensor networks, and holography.
Be precise with physics terminology."""
```

### Multi-Turn Conversations

```python
# Context is preserved automatically
agent.chat("Start a simulation with 1000 nodes")
agent.chat("Now increase GW amplitude")  # Remembers context
agent.chat("Generate a paper")  # Knows which sim to document
```

### Event-Driven Alerts

```python
# Define custom alert conditions
agent.config.alert_thresholds = {
    'divergence': 1e12,
    'nan_fraction': 0.05,
    'correlation_drop': 0.3
}

# Start monitoring
await agent.monitor_simulation()

# Alerts sent to user via WebSocket
```

## Security Considerations

- LLM runs in isolated Docker container
- No external API calls (fully offline)
- WebSocket connections are local-only by default
- Function calling is sandboxed to safe operations only
- No file system access beyond /app/whitepapers

## Future Enhancements

- [ ] Multi-model support (switch models on-the-fly)
- [ ] Voice interface (Whisper integration)
- [ ] Visual analysis (describe plots/visualizations)
- [ ] Collaborative mode (multiple users, one LLM)
- [ ] Fine-tuning on iVHL-specific physics
- [ ] Knowledge base integration (RAG over documentation)

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Gradio Chat Interface](https://www.gradio.app/docs/chatbot)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [TinyLlama Paper](https://arxiv.org/abs/2401.02385)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)

---

**Last Updated**: 2025-12-15
**Status**: Production Ready
**Docker**: H100 Optimized
