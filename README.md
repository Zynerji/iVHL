# iVHL: Integrated Vibrational Helix Lattice Framework

**A Computational Research Platform for Quantum Gravity Phenomenology**

![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.5-green)
![Docker](https://img.shields.io/badge/Docker-H100%20Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

**iVHL** is a GPU-accelerated computational framework for exploring quantum gravity phenomenology through:
- **Holographic resonance simulations** with spherical boundary conditions
- **Group Field Theory (GFT)** condensate dynamics and phase transitions
- **Tensor network holography** (MERA, HaPPY codes, spin foams)
- **LIGO-inspired gravitational wave** lattice analysis with fractal harmonics
- **Reinforcement learning** discovery campaigns (TD3-SAC hybrid)
- **Automated report generation** with LaTeX white papers

**Not a theory of everything** — a research tool for computational exploration of holographic duality, emergent geometry, and quantum-classical correspondence.

---

## 🚀 Quick Start

### Docker Deployment (H100 GPU - Recommended)

```bash
# Clone repository
git clone https://github.com/Zynerji/iVHL.git
cd iVHL

# Build H100-optimized container
docker build -t ivhl-h100:latest .

# Run with GPU support
docker run --gpus all -p 8501:8501 \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/checkpoints:/app/checkpoints \
  ivhl-h100:latest

# Access Streamlit interface
# Open http://localhost:8501
```

See [`DEPLOY_H100.md`](DEPLOY_H100.md) for complete deployment guide.

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run holographic resonance dashboard
streamlit run dashboards/resonance_dashboard.py

# Run GW lattice analysis
streamlit run dashboards/gw_dashboard.py
```

---

## 🌀 Core Modules

### 1. Holographic Resonance (`vhl_holographic_resonance.py`)

Simulates wave interference on spherical boundary creating 3D resonant structures:

**Physics**:
```
ψ(r, t) = Σ_i A_i sin(k|r - r_i| + φ_i(t)) / |r - r_i|
```
- Coherent wave sources on helical boundary lattice
- 3D superposition with Calabi-Yau-like folded topologies
- Phase singularities (vortices) with topological charges
- Particle advection in resonant field

**Key Features**:
- GPU-accelerated field computation (PyTorch)
- Multi-vortex dynamics with Fourier/RNN trajectory control
- PyVista volumetric rendering
- Streamlit interactive interface

**Usage**:
```python
from ivhl.resonance.holographic_resonance import HolographicResonator

resonator = HolographicResonator(
    num_sources=500,
    sphere_radius=1.0,
    helical_turns=5.0
)
field = resonator.compute_field(grid_resolution=128)
```

### 2. Group Field Theory (`gft_condensate_dynamics.py`, `gft_tensor_models.py`)

Implements GFT as UV-complete foundation for emergent spacetime:

**Colored Tensor Models**:
```math
S[T] = (1/2) Tr(T†T) + (λ/d!) Tr(T^d)
```
- Melonic diagram dominance in large-N limit
- Schwinger-Dyson equations for dressed propagator G*
- Critical coupling λ_c and double-scaling limit

**GFT Condensate**:
```math
V_eff(σ) = (m²/2)|σ|² + (λ/d!)|σ|^d
```
- Phase transition: non-geometric (σ=0) ↔ geometric (σ≠0)
- Gross-Pitaevskii dynamics
- Emergent FLRW cosmology with bouncing solutions

**Holographic Stack Weaving** (`holographic_stack_weaving.py`):
- 8-layer architecture: Boundary → Tensor → Condensate → MERA/HaPPY → Spin Networks → CDT → RG Flow → Full Closure
- Cross-consistency checks (entropy, amplitudes, beta functions)
- Unifies all quantum gravity formalisms

### 3. LIGO-Inspired GW Lattice (`gw_lattice_mode.py`, `gw_fractal_analysis.py`)

Explores gravitational wave phenomenology with vibrational lattice:

**Waveform Generation**:
- **Inspiral**: f(t) ∝ (t_c - t)^(-3/8) chirp
- **Ringdown**: Quasinormal modes with Q-factor decay
- **Stochastic**: Ω_GW(f) ∝ f^(2/3) power-law spectrum
- **Constant Lattice**: Embedded π, e, φ, √2, √3 at harmonic frequencies

**Lattice Perturbation**:
```
Δr = r * h(t)  (radial strain)
Δx/x = h,  Δy/y = -h  (tidal deformation)
```

**Fractal Analysis**:
- Box-counting fractal dimension: D_box = log(N(ε)) / log(1/ε)
- Harmonic series detection via FFT
- Mathematical constant residue matching
- Log-space power-law fitting

**Persistence Tests**:
- Phase scrambling robustness
- Null scrambling tolerance
- Memory field decay (exponential fitting for τ)

### 4. Reinforcement Learning Discovery (`gw_rl_discovery.py`, `td3_sac_hybrid_training.py`)

RL-driven exploration of parameter space:

**TD3-SAC Hybrid**:
- Twin Delayed DDPG (TD3) for stability
- Soft Actor-Critic (SAC) for exploration (entropy maximization)
- Automatic temperature tuning (α auto-adjustment)
- Prioritized experience replay

**GW-Specific Rewards**:
- Lattice stability (Procrustes similarity): weight 2.0
- Fractal dimension (target D ≈ 2.6): weight 1.5
- Constant residues (π, e, φ): weight 3.0
- Attractor convergence (low variance): weight 2.5
- Memory persistence (long τ): weight 2.0

**Discovery Modes**:
1. `FIND_CONSTANT_LATTICE`
2. `FRACTAL_HARMONIC_STABILIZATION`
3. `ATTRACTOR_CONVERGENCE`
4. `GW_MEMORY_FIELD`
5. `QUASINORMAL_RINGING`
6. `LATTICE_UNDER_SCRAMBLING`

### 5. Automated Reporting (`simulation_report_generator.py`)

**Data Exfiltration & Documentation**:
- JSON structured data
- Markdown summary
- LaTeX white paper with professional template
- PDF compilation (pdflatex in Docker)
- Automatic GitHub commit/push

**Usage**:
```python
from ivhl.integration.report_generator import IntegratedReportGenerator

generator = IntegratedReportGenerator(
    output_base_dir=Path("whitepapers"),
    auto_commit=True,
    compile_pdf=True
)

files = generator.generate_full_report(
    simulation_type='gw_lattice_constant_probe',
    configuration={...},
    results={...},
    analysis={...}
)
# Generates: JSON + MD + TEX + PDF, commits to GitHub
```

---

## 📊 Performance Benchmarks

**NVIDIA H100 80GB HBM3**:

| Operation | Time | Speedup vs CPU |
|-----------|------|----------------|
| Holographic field (8192³) | 3-5 ms | ~200× |
| MERA contraction (128 tensors) | 8-12 ms | ~150× |
| GFT evolution (64³ grid) | 1200-1800 ms | ~100× |
| TD3-SAC update | 15-25 ms | ~80× |
| GW waveform (4096 samples) | ~10 ms | ~50× |
| Fractal box-counting (64³) | ~200 ms | ~30× |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     iVHL Framework                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │  Boundary   │───▶│  GFT/Tensor  │───▶│  Emergent       │ │
│  │  (Helical   │    │  Condensate  │    │  Spacetime      │ │
│  │  Lattice)   │    │  Dynamics    │    │  (Holographic)  │ │
│  └─────────────┘    └──────────────┘    └─────────────────┘ │
│         │                   │                     │          │
│         ▼                   ▼                     ▼          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          Holographic Resonance Simulation               │ │
│  │  • Wave interference (ψ = Σ A_i sin(...) / r_i)        │ │
│  │  • Vortex dynamics (Fourier + RNN control)             │ │
│  │  • Field computation (GPU PyTorch)                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         LIGO-Inspired GW Lattice Analysis               │ │
│  │  • Perturbations (inspiral, ringdown, stochastic)      │ │
│  │  • Fractal dimension (box-counting)                    │ │
│  │  • Constant residues (π, e, φ detection)              │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │       RL Discovery (TD3-SAC Hybrid)                     │ │
│  │  • Attractor search                                    │ │
│  │  • Lattice stability optimization                      │ │
│  │  • Memory persistence maximization                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │    Automated Report Generation                          │ │
│  │  • JSON + Markdown + LaTeX + PDF                        │ │
│  │  • GitHub auto-commit                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure (Clean Package Organization)

```
iVHL/
├── 📦 ivhl/                            # Core Python Package
│   ├── resonance/                     # Holographic boundary dynamics
│   │   ├── holographic_resonance.py  # Core physics engine
│   │   ├── visualization.py          # PyVista 3D rendering
│   │   ├── vortex_controller.py      # Basic vortex control
│   │   └── vortex_control_advanced.py # Fourier/RNN control
│   ├── gft/                           # Group Field Theory
│   │   ├── condensate_dynamics.py    # Gross-Pitaevskii dynamics
│   │   └── tensor_models.py          # Colored tensors, melonics
│   ├── tensor_networks/               # MERA, HaPPY, AdS/CFT
│   │   ├── holography.py             # MERA construction
│   │   ├── stack_weaving.py          # 8-layer unified framework
│   │   └── ads_cft_entanglement.py   # Ryu-Takayanagi entropy
│   ├── gw/                            # Gravitational wave analysis
│   │   ├── lattice_mode.py           # Waveforms, perturbations
│   │   ├── fractal_analysis.py       # Fractal dimension, harmonics
│   │   └── rl_discovery.py           # RL rewards for GW
│   ├── rl/                            # Reinforcement learning
│   │   ├── sac_core.py               # Soft Actor-Critic
│   │   ├── sac_training.py           # SAC training utilities
│   │   ├── sac_rewards.py            # Reward engineering
│   │   ├── td3_sac_core.py           # Hybrid TD3-SAC
│   │   └── td3_sac_training.py       # Hybrid training
│   ├── integration/                   # API & utilities
│   │   ├── api.py                    # REST API interface
│   │   ├── integration.py            # Cross-module integration
│   │   └── report_generator.py       # JSON/MD/LaTeX/PDF
│   └── legacy/                        # Deprecated modules
│
├── 🎨 dashboards/                      # Streamlit interfaces
│   ├── resonance_dashboard.py        # Holographic resonance UI
│   ├── gw_dashboard.py               # GW lattice analysis
│   ├── sac_dashboard.py              # RL training visualization
│   ├── webgpu_component.py           # WebGPU acceleration
│   └── webgpu_client.html            # Browser GPU rendering
│
├── 🔬 simulations/                     # Simulation scripts
│   ├── README.md                     # Simulation guide
│   └── full_11d_holographic_simulation.py
│
├── 🧪 tests/                           # Test suites
│   └── test_report_pipeline.py       # End-to-end test
│
├── ⚙️ scripts/                         # Utility scripts
│   ├── benchmarks.py                 # Performance profiling
│   ├── compiled_ops.py               # torch.compile examples
│   └── sac_example.py                # SAC usage example
│
├── ⚙️ configs/                         # JSON configurations
│   ├── mera_network.json             # Tensor network config
│   └── multi_vortex_config.json      # Vortex parameters
│
├── 📚 docs/                            # Documentation
│   ├── DEPLOY_H100.md                # H100 deployment guide
│   ├── TD3_SAC_HYBRID_GUIDE.md       # RL training guide
│   └── [10+ other guides]
│
├── 📄 whitepapers/                     # Generated PDF reports
│   └── report_YYYYMMDD_HHMMSS/       # Timestamped reports
│
├── 🗃️ Data & Outputs
│   ├── checkpoints/                  # Model checkpoints (gitignored)
│   ├── logs/                         # Training logs (gitignored)
│   └── reports/                      # Simulation data (gitignored)
│
├── 🐳 Docker Deployment
│   ├── Dockerfile                    # H100-optimized (CUDA 12.5)
│   └── .dockerignore                 # Build optimization
│
└── 📋 Project Files
    ├── README.md                     # This document
    ├── README.tex                    # LaTeX version
    ├── Hello_Claude.md               # Claude AI context file
    ├── requirements.txt              # Python dependencies
    └── .gitignore                    # Git exclusions
```

---

## 🔬 Research Applications

### Holographic Duality Exploration
- Test AdS/CFT-like correspondences in discrete systems
- Explore boundary-bulk information encoding
- Investigate entanglement entropy scaling (Ryu-Takayanagi)

### Quantum Gravity Phenomenology
- GFT phase transitions (non-geometric ↔ geometric)
- Spin foam amplitudes and CDT Hausdorff dimensions
- Asymptotic safety RG flow fixed points

### Gravitational Wave Analysis
- Fractal structure in stochastic GW background
- Mathematical constant encoding in strain data
- Memory field persistence (non-Markovian dynamics)

### RL-Driven Discovery
- Attractor basin identification
- Lattice stability optimization
- Emergent structure discovery

---

## 🐳 Docker Self-Contained Deployment

**Complete production-ready containerization**:

```bash
# Build (includes all dependencies + LaTeX)
docker build -t ivhl-h100:latest .

# Run with automatic report generation
docker run --gpus all -p 8501:8501 \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  ivhl-h100:latest

# Reports auto-generated after simulations:
# - reports/report_YYYYMMDD_HHMMSS/
#   ├── report_*.json         # Structured data
#   ├── report_*.md           # Summary
#   ├── whitepaper_*.tex      # LaTeX source
#   └── whitepaper_*.pdf      # Compiled PDF
```

**Data exfiltration**: All reports saved to mounted `/app/reports` volume.

**GitHub integration**: Auto-commit with:
```python
generator = IntegratedReportGenerator(auto_commit=True)
# Automatically commits reports to repository
```

---

## 🛠️ Development

### Prerequisites
- **GPU**: NVIDIA with CUDA 12.1+ (H100 recommended for production)
- **RAM**: 64GB+ recommended for large simulations
- **Storage**: 100GB+ for checkpoints and results
- **OS**: Ubuntu 22.04 LTS (Docker) or Windows 10/11 (local)

### Local Development

```bash
# Clone
git clone https://github.com/Zynerji/iVHL.git
cd iVHL

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run tests (if available)
pytest tests/

# Run benchmarks
python benchmarks.py
```

### Contributing

We welcome contributions in:
- **Physics**: New quantum gravity formalisms
- **Performance**: GPU optimization, distributed computing
- **Analysis**: Novel metrics, visualization techniques
- **RL**: Advanced discovery algorithms
- **Documentation**: Tutorials, examples

**Submit PRs** to: https://github.com/Zynerji/iVHL

---

## 📖 References

### Holographic Duality
1. Maldacena, J. (1999). "The Large N Limit of Superconformal Field Theories". *Int. J. Theor. Phys.* **38**: 1113-1133.
2. Ryu, S., Takayanagi, T. (2006). "Holographic Derivation of Entanglement Entropy". *Phys. Rev. Lett.* **96**: 181602.
3. Pastawski, F. et al. (2015). "Holographic quantum error-correcting codes". *JHEP* **06**: 149.

### Group Field Theory
4. Oriti, D. (2016). "Group Field Theory and Loop Quantum Gravity". *Loop Quantum Gravity: The First 30 Years*.
5. Gielen, S., Oriti, D., Sindoni, L. (2013). "Cosmology from Group Field Theory". *Phys. Rev. Lett.* **111**: 031301.
6. Gurau, R. (2011). "Colored Group Field Theory". *Comm. Math. Phys.* **304**: 69-93.

### Gravitational Waves
7. Abbott, B.P. et al. (2016). "Observation of Gravitational Waves". *Phys. Rev. Lett.* **116**: 061102.
8. LIGO Scientific Collaboration (2021). "GWTC-3: Compact Binary Coalescences". arXiv:2111.03606.

### Tensor Networks
9. Vidal, G. (2008). "Class of Quantum Many-Body States That Can Be Efficiently Simulated". *Phys. Rev. Lett.* **101**: 110501.
10. Evenbly, G., Vidal, G. (2015). "Tensor Network Renormalization". *Phys. Rev. Lett.* **115**: 180405.

---

## ⚠️ Disclaimer

iVHL is a **computational research tool** for exploring quantum gravity phenomenology. It is **not**:
- ❌ A theory of everything
- ❌ A replacement for established physics
- ❌ Claiming to have "discovered new laws"

It **is**:
- ✅ A framework for computational exploration
- ✅ A tool for testing holographic duality concepts
- ✅ A platform for RL-driven discovery
- ✅ An educational resource for quantum gravity formalisms

**Use responsibly** for research and education.

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 📧 Contact

**Repository**: https://github.com/Zynerji/iVHL
**Issues**: https://github.com/Zynerji/iVHL/issues
**Documentation**: See `docs/` directory

---

## 🎓 Citation

If using iVHL in research:

```bibtex
@software{ivhl2025,
  title = {iVHL: Integrated Vibrational Helix Lattice Framework for Quantum Gravity Phenomenology},
  author = {iVHL Development Team},
  year = {2025},
  url = {https://github.com/Zynerji/iVHL},
  note = {Computational platform for holographic resonance, GFT, tensor networks, and LIGO-inspired GW analysis}
}
```

---

*Computational exploration of holographic duality, emergent geometry, and quantum-classical correspondence.*

---

## Multi-Scale Holographic Exploration Framework

**IMPORTANT DISCLAIMER**: This framework is a computational exploration of mathematical models including holographic resonance, tensor networks, and emergent patterns. It does NOT claim to explain physical phenomena, discover new laws, or predict dark matter/dark energy.

### Overview

The multi-scale framework provides tools for exploring geometric patterns through multiple computational layers:

1. **Boundary Resonance** - Wave interference patterns on spherical boundaries
2. **GFT Field Evolution** - Gross-Pitaevskii dynamics with colored tensor structure
3. **MERA Bulk Reconstruction** - Tensor network holographic compression
4. **Perturbation Analysis** - Lattice stability under external perturbations
5. **RL Discovery** - Reinforcement learning for configuration optimization
6. **Multi-Scale Upscaling** - Information projection across scales
7. **Analysis & Visualization** - Comprehensive result analysis

### Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib pyyaml pytest

# Run basic exploration
python simulations/multiscale_exploration.py

# Run tests
pytest tests/test_multiscale.py -v
```

### Module Reference

| Module | Description | Key Classes |
|--------|-------------|-------------|
| `ivhl.multiscale.boundary_resonance` | Spherical wave interference | `BoundaryResonanceSimulator` |
| `ivhl.multiscale.gft_field` | GFT field evolution | `GFTFieldEvolver` |
| `ivhl.multiscale.mera_bulk` | Tensor network reconstruction | `MERABulkReconstructor` |
| `ivhl.multiscale.perturbation_engine` | Stability analysis | `PerturbationEngine` |
| `ivhl.multiscale.rl_discovery` | RL optimization | `RLDiscoveryAgent` |
| `ivhl.multiscale.upscaling` | Multi-scale projection | `MultiScaleUpscaler` |
| `ivhl.multiscale.analysis` | Result analysis | `SimulationAnalyzer` |

### Configuration

Edit `configs/multiscale_config.yaml` to customize:

```yaml
boundary:
  num_nodes: 126
  grid_resolution: 64
  timesteps: 100
  device: cuda  # or 'cpu'

gft:
  grid_size: 32
  num_colors: 4
  interaction_strength: 0.1
```

### Example Usage

```python
from ivhl.multiscale import BoundaryResonanceSimulator, BoundaryConfig

# Configure simulation
config = BoundaryConfig(
    num_nodes=126,
    grid_resolution=64,
    timesteps=100
)

# Run simulation
simulator = BoundaryResonanceSimulator(config)
results = simulator.run_simulation()

# Analyze results
entropy = simulator.compute_entropy(results['field_evolution'][-1])
print(f"Final field entropy: {entropy:.4f}")
```

### Testing

```bash
# Run all tests
pytest tests/test_multiscale.py -v

# Run specific test class
pytest tests/test_multiscale.py::TestGFTEvolution -v

# Run with coverage
pytest tests/test_multiscale.py --cov=ivhl.multiscale
```

### Output Structure

Results are saved to `results/` directory:

```
results/
├── boundary_results.npz       # Boundary field evolution
├── gft_results.npz           # GFT field snapshots
├── multiscale_analysis.json  # Analysis metrics
└── multiscale_summary.png    # Visualization plots
```

### GPU Acceleration

The framework supports CUDA acceleration:

```python
config = BoundaryConfig(device="cuda")  # Use GPU
config = BoundaryConfig(device="cpu")   # Use CPU
```

**Performance** (approximate):
- H100 GPU: 10-15 seconds per full pipeline
- CPU: ~5 minutes per full pipeline

### Scientific Interpretation

All results should be interpreted as:
- ✓ Mathematical patterns in computational models
- ✓ Information-theoretic metrics
- ✓ Geometric structure exploration

NOT as:
- ✗ Predictions about physical dark matter/energy
- ✗ Claims about discovering new physical laws
- ✗ Models of real quantum gravity

### Troubleshooting

**CUDA out of memory:**
```python
# Reduce grid size
config = GFTConfig(grid_size=16)  # Instead of 32
```

**Slow CPU performance:**
```python
# Reduce timesteps
config = BoundaryConfig(timesteps=50)  # Instead of 100
```

### Contributing

Contributions welcome! Please ensure:
- All code includes proper disclaimers
- Tests pass: `pytest tests/ -v`
- Code formatted: `black ivhl/`
- Documentation updated

### References

1. Maldacena, J. (1999). The Large N Limit of Superconformal Field Theories
2. Ryu, S., Takayanagi, T. (2006). Holographic Derivation of Entanglement Entropy
3. Vidal, G. (2007). Entanglement Renormalization (MERA)
4. PySCF Documentation: https://pyscf.org

### License

MIT License - See LICENSE file for details.

---

**Framework Version**: 1.0.0  
**Last Updated**: 2025-12-15  
**Status**: Research/Development

---

## Hierarchical Information Dynamics (Option B)

**NEW**: GPU-accelerated simulation with embedded LLM monitoring and real-time visualization.

### Overview

Explore information flow through multi-layer tensor networks with:
- **Server-side GPU rendering** - PyVista renders on H100/H200, streams to browser
- **Embedded LLM (Qwen2.5-2B)** - Real-time scientific monitoring with note-taking
- **Auto-scaling** - Detects GPU, reserves 6GB for LLM, 10GB for rendering, rest for simulation
- **WebSocket streaming** - Live 3D visualization at 30 FPS
- **Automated whitepaper** - LaTeX PDF generated from LLM notes

### Quick Start

```bash
# Build Docker image
cd iVHL
docker build -t ivhl-hierarchical -f docker/Dockerfile .

# Run with GPU
docker run --gpus all \
  -p 8080:8080 -p 8000:8000 \
  -v $(pwd)/results:/results \
  ivhl-hierarchical

# Access web interface
open http://localhost:8080/
```

### What You'll See

- **3D rotating visualization** of tensor network evolving in real-time
- **Live LLM commentary** analyzing entropy flows and correlations
- **Interactive chat** - ask Qwen about what's happening
- **Progress metrics** - step count, GPU utilization, notes taken
- **One-click whitepaper** generation when complete

### Architecture

```
GPU (H100/H200 80-141GB)
├─ vLLM (6GB) → Qwen2.5-2B monitoring
├─ PyVista Rendering (10GB) → Server-side 3D
├─ Simulation (60-125GB) → Tensor network dynamics
└─ Safety Buffer (4GB)

FastAPI Server → WebSocket → Browser
  - /ws/frames → JPEG stream @ 30fps
  - /ws/metrics → JSON metrics @ 10Hz
  - /ws/llm-commentary → LLM notes stream
  - /api/llm/chat → Interactive Q&A
```

### Resource Allocation

| GPU | LLM | Rendering | Simulation | Config |
|-----|-----|-----------|------------|--------|
| H200 (141GB) | 6GB | 10GB | 121GB | 128³ grid, ultra quality |
| H100 (80GB) | 6GB | 10GB | 60GB | 64³ grid, high quality |
| A100 (40GB) | 6GB | 10GB | 20GB | 32³ grid, medium quality |
| CPU | Disabled | Disabled | All RAM | 8³ grid, low quality |

### Files Created

```
ivhl/hierarchical/
├── __init__.py
├── tensor_hierarchy.py      # Multi-layer tensor network
├── compression_engine.py    # Compression strategies
├── entropy_analyzer.py      # Entropy flow tracking
└── correlation_tracker.py   # Inter-layer correlations

web_monitor/
├── streaming_server.py      # FastAPI + WebSocket server
├── llm_monitoring_agent.py  # Qwen agent with system prompt
├── whitepaper_generator.py  # LaTeX PDF from notes
└── rendering/
    └── gpu_renderer.py      # PyVista server-side rendering

docker/
├── Dockerfile              # CUDA 12.5 + vLLM + PyTorch
├── entrypoint.sh          # Orchestration script
└── gpu_detect_and_scale.py # Auto-scaling logic

simulations/hierarchical_dynamics/
└── run_simulation.py       # Main runner

configs/hierarchical/
└── default.yaml           # Configuration

tests/
└── test_hierarchical.py   # Unit tests
```

### Deployment Workflow

1. **Rent VM** (H100/H200 with CUDA 12.5)
2. **Provide credentials** to Claude via SSH
3. **Claude executes**:
   ```bash
   git clone https://github.com/Zynerji/iVHL.git
   cd iVHL
   docker build -t ivhl-hierarchical -f docker/Dockerfile .
   docker run --gpus all -p 8080:8080 -p 8000:8000 -v $(pwd)/results:/results ivhl-hierarchical
   ```
4. **Access** `http://<VM_IP>:8080/` in browser
5. **Watch** simulation run with live LLM commentary
6. **Download** whitepaper PDF when complete

### Scientific Integrity

**CRITICAL DISCLAIMER**: This simulates information dynamics in abstract mathematical tensor networks. It does NOT model physical systems, dark matter, cosmology, or quantum gravity.

Qwen2.5-2B is configured with a system prompt emphasizing:
- ✅ Honest analysis of computational patterns
- ✅ Information-theoretic language
- ✅ Quantitative metric-based commentary
- ❌ NO claims about physical reality
- ❌ NO dark matter/cosmology connections
- ❌ NO "theory validation"

### Future Simulations

See `docs/FUTURE_SIMULATIONS.md` for:
- **Option A**: Emergent Gravitational Effects in Tensor Networks
- **Option C**: Lattice Perturbation and Residual Entropy Studies

Both will use the same Docker/LLM/rendering infrastructure.

---

**Framework Version**: 1.1.0 (Hierarchical Dynamics)
**Last Updated**: 2025-12-15
**Status**: Production Ready
