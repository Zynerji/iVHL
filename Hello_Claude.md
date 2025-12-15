# Hello Claude - iVHL Framework Context

**Last Updated**: 2025-12-15
**Project**: iVHL (Vibrational Helical Lattice) Framework
**Repository**: https://github.com/Zynerji/iVHL
**Status**: Production-Ready (Docker H100 deployment)

---

## CRITICAL: What This Project IS and ISN'T

### ❌ NOT:
- A theory of everything
- A replacement for established physics
- Claiming to explain or predict real physical phenomena

### ✅ IS:
- A computational research platform for quantum gravity phenomenology
- A tool for testing holographic duality concepts (AdS/CFT-inspired)
- An exploration framework for Group Field Theory (GFT) and tensor networks
- A LIGO-inspired gravitational wave lattice analysis system
- A reinforcement learning discovery engine for emergent spacetime phenomena

---

## Project Goal

**Primary Objective**: Simulate emergent spacetime from holographic resonance on a spherical boundary, using:
- Vibrational helical lattice as holographic boundary
- Group Field Theory (GFT) condensate dynamics
- Tensor network holography (MERA/multiscale entanglement renormalization)
- LIGO-inspired gravitational wave (GW) perturbation analysis
- Reinforcement learning for discovering stable configurations

**Key Question**: Can structured resonance patterns on a 2D spherical boundary encode 3D (or higher-dimensional) bulk spacetime geometry via holographic principles?

---

## Core Architecture (11-Dimensional Framework)

The iVHL framework operates in an **11-dimensional space**:

### Boundary Dimensions (2D + 1 time)
1. **θ (theta)**: Spherical coordinate (polar angle, 0 to π)
2. **φ (phi)**: Spherical coordinate (azimuthal angle, 0 to 2π)
3. **t (time)**: Evolution parameter

### Bulk Emergent Dimensions (3D spatial)
4. **x**: Emergent spatial coordinate (from entanglement structure)
5. **y**: Emergent spatial coordinate
6. **z**: Emergent spatial coordinate (radial from origin)

### Field/Tensor Dimensions (5D internal)
7. **Color index c₁**: GFT field color label (0-3, representing SU(2) or graph connectivity)
8. **Color index c₂**: Second color label
9. **Color index c₃**: Third color label
10. **Spin/Helicity s**: Internal angular momentum quantum number
11. **Tensor rank r**: Position in MERA hierarchy (coarse-graining scale)

**Holographic Encoding**: The 2D+1 boundary (dims 1-3) encodes 3D+5D=8D information through:
- Resonant field amplitude ψ(θ,φ,t)
- GFT condensate wave function Φ(c₁,c₂,c₃)
- Tensor network structure T(r,s)
- Vortex configurations (topological charges)

---

## Core Concepts

### 1. Holographic Resonance
- **Source**: Acoustic wave interference on spherical shell
- **Equation**: `ψ(r,t) = Σᵢ Aᵢ sin(k|r-rᵢ|) / |r-rᵢ|`
- **Nodes**: Helical lattice points on boundary sphere
- **Vortices**: Phase singularities where Re(ψ)=0 and Im(ψ)=0
- **File**: `vhl_holographic_resonance.py`

### 2. Group Field Theory (GFT) Condensate
- **Purpose**: Pre-geometric quantum spacetime from colored tensor fields
- **Dynamics**: Gross-Pitaevskii equation with quartic interaction
  - `iℏ ∂Φ/∂t = [-ℏ²∇²/(2m) + V + λ|Φ|²]Φ`
- **Phase transition**: Disordered → Condensate → Emergent geometry
- **Melonic diagrams**: Dominant Feynman graphs (D-dimensional melons)
- **Files**: `gft_condensate_dynamics.py`, `gft_tensor_models.py`

### 3. Tensor Network Holography
- **MERA**: Multiscale Entanglement Renormalization Ansatz
- **Structure**: Binary tree with disentanglers (U) and isometries (W)
- **RT Formula**: Entanglement entropy S(A) = Area(γₐ)/(4G) (Ryu-Takayanagi)
- **Bulk reconstruction**: Geodesics in AdS ↔ Minimal surfaces in tensor network
- **File**: `tensor_network_holography.py`, `holographic_stack_weaving.py`

### 4. LIGO-Inspired GW Lattice Analysis
- **Waveforms**: Inspiral chirp, ringdown (quasinormal modes), constant lattice
- **Strain**: h(t) = perturbation of lattice node positions
  - Radial: `Δr = r * h(t)`
  - Tidal: Plus (+) and cross (×) polarizations
- **Constant residues**: Embedded π, e, φ, √2, √3 in harmonic frequencies
- **Fractal analysis**: Box-counting dimension, log-space harmonics
- **Memory field**: Exponential decay τ after perturbation (GW memory effect)
- **Files**: `gw_lattice_mode.py`, `gw_fractal_analysis.py`, `gw_rl_discovery.py`, `gw_streamlit_dashboard.py`

### 5. Reinforcement Learning Discovery
- **Algorithm**: TD3-SAC hybrid (Twin Delayed DDPG + Soft Actor-Critic)
- **State**: Lattice configuration (node positions, vortex charges, field amplitudes)
- **Action**: Adjust source amplitudes, phases, lattice geometry
- **Rewards**:
  - Lattice stability (Procrustes similarity after GW perturbation)
  - Fractal dimension (target ~1.5-2.0 for self-similar structure)
  - Constant residue detection (π, e, φ in frequency peaks)
  - Attractor convergence (fixed-point dynamics)
  - Memory persistence (quality factor Q > threshold)
  - Harmonic richness (number of peaks in power spectrum)
- **Discovery modes**:
  - FIND_CONSTANT_LATTICE
  - FRACTAL_HARMONIC_STABILIZATION
  - ATTRACTOR_CONVERGENCE
  - GW_MEMORY_FIELD
  - VORTEX_PINNING_OPTIMIZATION
  - ENTANGLEMENT_MAXIMIZATION
- **Files**: `td3_sac_hybrid_core.py`, `td3_sac_hybrid_training.py`, `gw_rl_discovery.py`, `sac_*.py`

### 6. Automated Report Generation
- **Pipeline**: Simulation → JSON + Markdown + LaTeX → PDF → GitHub commit
- **LaTeX Template**: Professional academic white paper with:
  - Abstract, configuration tables, results sections
  - Analysis with equations (rendered LaTeX math)
  - Conclusions and implications
  - References (Maldacena, Ryu-Takayanagi, Pastawski, Oriti, LIGO)
  - Appendices (raw data, reproducibility)
- **Data Exfiltration**: Timestamped reports in `whitepapers/report_YYYYMMDD_HHMMSS/`
- **File**: `ivhl/integration/report_generator.py`

### 7. **NEW: Embedded LLM System** (2025-12-15)
- **Purpose**: Autonomous AI assistant running on H100 alongside simulations
- **Models**: TinyLlama (1.1B), Phi-3 (3.8B), Llama-3.2 (3B), Mistral-7B (4-bit)
- **Key Features**:
  - Real-time simulation monitoring
  - Natural language control (modify parameters via chat)
  - Autonomous white paper generation (using H100, not Claude API)
  - Interactive chat interface (Gradio)
  - Function calling to control simulations
  - Completely offline (no external APIs)
- **Architecture**: LLM loads FIRST (reserves 10-15% GPU), simulations use remaining 80-85%
- **Startup**: `python scripts/start_with_llm.py --full`
- **Ports**: LLM Chat (7860), Simulation Dashboard (8501)
- **Files**: `ivhl/llm/agent.py`, `dashboards/llm_chat.py`, `scripts/start_with_llm.py`
- **Documentation**: See `ivhl/llm/README.md` for complete guide

---

## Key Modules Reference

### Core Physics
| File | Purpose |
|------|---------|
| `vhl_holographic_resonance.py` | Acoustic resonance on spherical boundary |
| `gft_condensate_dynamics.py` | Gross-Pitaevskii dynamics, phase transitions |
| `gft_tensor_models.py` | Colored tensor models, melonic diagrams |
| `tensor_network_holography.py` | MERA construction, RT formula |
| `holographic_stack_weaving.py` | Multi-layer holographic encoding |
| `vhl_ads_cft_entanglement.py` | AdS/CFT correspondence, entanglement |

### GW Lattice Analysis
| File | Purpose |
|------|---------|
| `gw_lattice_mode.py` | GW waveform generation, lattice perturbation |
| `gw_fractal_analysis.py` | Fractal dimension, harmonic detection |
| `gw_rl_discovery.py` | RL rewards for GW phenomena |
| `gw_streamlit_dashboard.py` | Interactive GW analysis dashboard |

### Reinforcement Learning
| File | Purpose |
|------|---------|
| `td3_sac_hybrid_core.py` | TD3-SAC agent implementation |
| `td3_sac_hybrid_training.py` | Training loops, discovery campaigns |
| `sac_core.py` | Pure SAC agent (entropy-based) |
| `sac_training.py` | SAC-specific training |
| `sac_rewards.py` | Reward engineering for various phenomena |

### Visualization & Interaction
| File | Purpose |
|------|---------|
| `vhl_resonance_streamlit.py` | Main Streamlit UI for resonance control |
| `gw_streamlit_dashboard.py` | GW-specific dashboard |
| `sac_streamlit.py` | RL training visualization |
| `vhl_resonance_viz.py` | 3D visualization utilities |
| `streamlit_webgpu_component.py` | WebGPU browser acceleration |
| `vhl_webgpu.html` | Client-side GPU-accelerated rendering |

### Utilities
| File | Purpose |
|------|---------|
| `simulation_report_generator.py` | Automated JSON/MD/LaTeX/PDF reports |
| `compiled_ops.py` | torch.compile optimizations |
| `benchmarks.py` | Performance benchmarking |
| `vhl_vortex_controller.py` | Vortex topology management |

---

## Recent Work (Context for Continuation)

### 1. LIGO Integration (2025-12-15)
- **Goal**: Incorporate gravitational wave analysis insights
- **Implementation**:
  - 4 waveform types (inspiral, ringdown, stochastic, constant_lattice)
  - Fractal harmonic detection in log-space
  - Lattice perturbation engine (radial, tidal, phase scrambling)
  - Persistence analysis (Procrustes similarity, exponential decay)
  - RL discovery modes for GW phenomena
- **Files Added**: `gw_*.py` (4 modules)
- **Commits**: 4677d0b, 29b0205, 99fb9b7, c23decc

### 2. Automated Reporting System (2025-12-15)
- **Goal**: Data exfiltration from Docker containers
- **Implementation**:
  - IntegratedReportGenerator (JSON + MD + LaTeX + PDF)
  - Professional LaTeX template with academic citations
  - GitHub auto-commit capability
  - Timestamped report directories
- **File Added**: `simulation_report_generator.py`
- **Dockerfile Updated**: Added texlive-latex-base, extra, fonts
- **Commit**: e4f45b7

### 3. Documentation Cleanup (2025-12-15)
- **Goal**: Clarify project scope and objectives
- **Changes**:
  - Added explicit disclaimer about project scope
  - Reframed as computational research platform
  - Focused on holographic resonance, GFT, tensor networks, LIGO analysis
  - Moved outdated documentation to `archive/`
- **Commits**: e4f45b7, multiple

### 4. Repository Organization (2025-12-15)
- **Goal**: Clean up root directory clutter
- **Changes**:
  - Moved guide .md files → `docs/`
  - Moved JSON configs → `configs/`
  - Moved tests → `tests/`
  - Moved assets → `assets/`
  - Created `simulations/` folder

---

## Docker Deployment (H100 Production-Ready)

### Build & Run
```bash
# Build container
docker build -t ivhl-h100:latest .

# Run with GPU and data exfiltration
docker run --gpus all \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/simulations:/app/simulations \
  -p 8501:8501 \
  ivhl-h100:latest
```

### Container Specs
- **Base**: nvidia/cuda:12.5.1-cudnn9-devel-ubuntu22.04
- **Python**: 3.11
- **PyTorch**: 2.5.1 with CUDA 12.1 (H100 compatible)
- **Compute Capability**: 9.0 (TORCH_CUDA_ARCH_LIST="9.0")
- **LaTeX**: texlive-latex-base, extra, fonts (for PDF generation)
- **Ports**: 8501 (Streamlit), 8080 (trame visualization)

### Performance
- **H100 80GB HBM3**: ~1000 nodes in <1s per timestep
- **CPU Fallback**: ~500 nodes in ~5s per timestep
- **Optimizations**: torch.compile, CUDA graphs, mixed precision

---

## How to Continue Development

### Quick Start
1. Read this file completely
2. Check `README.md` for current project state
3. Review `docs/` for detailed guides on specific topics
4. Run `tests/test_report_pipeline.py` to validate environment
5. Launch Streamlit dashboard: `streamlit run vhl_resonance_streamlit.py`

###Common Tasks

#### Add New Waveform Type
1. Edit `ivhl/gw/lattice_mode.py` → `GWWaveformGenerator` class
2. Add method like `def my_waveform(self) -> torch.Tensor`
3. Update `perturbation_type` in `GWLatticeConfig`
4. Test with `GWLatticeProbe`

#### Add New RL Reward
1. Edit `ivhl/gw/rl_discovery.py` or `ivhl/rl/sac_rewards.py`
2. Add method to `GWRewardComputer` or `SACRewardComputer`
3. Update `weights` dictionary
4. Add to discovery campaign in training script

#### Create New Simulation
1. Create Python file in `simulations/`
2. Import modules using new package structure:
   ```python
   from ivhl.resonance import holographic_resonance
   from ivhl.gft import condensate_dynamics
   from ivhl.gw import lattice_mode
   from ivhl.integration import report_generator
   ```
3. Configure parameters
4. Run simulation
5. Generate report with `IntegratedReportGenerator`
6. Export results to `whitepapers/`

#### Modify Holographic Encoding
1. Edit `ivhl/resonance/holographic_resonance.py` for boundary dynamics
2. Edit `ivhl/tensor_networks/holography.py` for bulk reconstruction
3. Ensure consistency between boundary field and bulk geometry
4. Validate with entanglement entropy checks (RT formula)

#### Run Dashboards
1. Resonance: `streamlit run dashboards/resonance_dashboard.py`
2. GW Analysis: `streamlit run dashboards/gw_dashboard.py`
3. RL Training: `streamlit run dashboards/sac_dashboard.py`

### Testing
- **Unit tests**: (TODO - to be added in `tests/`)
- **Integration test**: `tests/test_report_pipeline.py`
- **Manual testing**: Streamlit dashboards (interactive)

---

## Important Constraints & Design Decisions

### 1. No Over-Engineering
- Keep implementations simple and focused
- Only add features explicitly requested
- Don't add excessive error handling for impossible cases
- Trust internal code and framework guarantees
- No backwards-compatibility hacks for unused code

### 2. GPU Acceleration
- All tensor operations use PyTorch (CPU/GPU agnostic)
- Device detection: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Move tensors to device: `tensor.to(device)`
- Use `torch.compile()` for production code (3-10x speedup)

### 3. Reproducibility
- Set random seeds for deterministic results
- Save all configurations in JSON
- Include git commit hash in reports
- Log all hyperparameters

### 4. Data Exfiltration from Docker
- Always generate reports after simulations
- Use volume mounts: `-v $(pwd)/reports:/app/reports`
- Enable auto-commit for GitHub backup (optional)
- JSON for programmatic access, PDF for human review

### 5. Coordinate Systems
- **Spherical (boundary)**: (r, θ, φ) with r = sphere_radius (usually 1.0)
- **Cartesian (bulk)**: (x, y, z) reconstructed from tensor network
- **Helical parameterization**: θ(s) = 2π * helical_turns * s, φ(s) = 2π * s/num_helices
- **GFT color indices**: (c₁, c₂, c₃) ∈ {0,1,2,3}

### 6. Physical Units
- **Length**: Arbitrary (normalized to sphere radius = 1.0)
- **Time**: Simulation timesteps (dt configurable)
- **Mass**: Atomic units (for GFT field, m_eff)
- **GW Strain**: Dimensionless (typical LIGO range: 10⁻²³ to 10⁻²¹)
- **Frequency**: Hz for GW, arbitrary for resonance modes

---

## Glossary of Key Terms

- **AdS/CFT**: Anti-de Sitter / Conformal Field Theory (holographic duality)
- **CCSD**: Coupled Cluster Singles Doubles (quantum chemistry, legacy)
- **GFT**: Group Field Theory (pre-geometric quantum spacetime)
- **MERA**: Multiscale Entanglement Renormalization Ansatz (tensor network)
- **QNM**: Quasinormal Mode (ringdown oscillation)
- **RT Formula**: Ryu-Takayanagi formula (entanglement entropy = area)
- **SAC**: Soft Actor-Critic (entropy-regularized RL)
- **TD3**: Twin Delayed Deep Deterministic Policy Gradient (RL algorithm)
- **VHL**: Vibrational Helical Lattice (boundary structure)

---

## Communication Style Preferences

- **Concise**: CLI-appropriate, no fluff
- **No emojis** unless explicitly requested
- **Technical accuracy** over user validation
- **Direct answers** without excessive praise
- **Git commit messages**: Detailed, professional, with Claude Code attribution
- **Code comments**: Only where logic isn't self-evident
- **No over-engineering**: Simplest solution that works

---

## Next Steps (Typical Workflow)

1. **Run existing simulations** to understand current capabilities
2. **Experiment with parameters** via Streamlit dashboards
3. **Analyze reports** in `reports/` to see what metrics are captured
4. **Identify gaps** in current implementation
5. **Add features incrementally** with immediate testing
6. **Generate reports** after each major change
7. **Commit to GitHub** with descriptive messages
8. **Deploy to Docker** for production runs on H100

---

## File Structure Summary (CLEAN PACKAGE ORGANIZATION)

```
iVHL/
├── Hello_Claude.md           ← YOU ARE HERE
├── README.md                 ← Public-facing overview
├── README.tex                ← LaTeX version of README
├── Dockerfile                ← H100-optimized container
├── requirements.txt          ← Python dependencies
│
├── ivhl/                     ← CORE PYTHON PACKAGE
│   ├── __init__.py
│   ├── resonance/           ← Holographic boundary dynamics
│   │   ├── holographic_resonance.py
│   │   ├── visualization.py
│   │   ├── vortex_controller.py
│   │   └── vortex_control_advanced.py
│   ├── gft/                 ← Group Field Theory
│   │   ├── condensate_dynamics.py
│   │   └── tensor_models.py
│   ├── tensor_networks/     ← MERA, RT formula, AdS/CFT
│   │   ├── holography.py
│   │   ├── stack_weaving.py
│   │   └── ads_cft_entanglement.py
│   ├── gw/                  ← GW lattice analysis
│   │   ├── lattice_mode.py
│   │   ├── fractal_analysis.py
│   │   └── rl_discovery.py
│   ├── rl/                  ← Reinforcement learning
│   │   ├── sac_core.py
│   │   ├── sac_training.py
│   │   ├── sac_rewards.py
│   │   ├── td3_sac_core.py
│   │   └── td3_sac_training.py
│   ├── integration/         ← API, reports, utilities
│   │   ├── api.py
│   │   ├── integration.py
│   │   └── report_generator.py
│   └── legacy/              ← Deprecated modules
│       └── [old code]
│
├── dashboards/               ← Streamlit interfaces
│   ├── resonance_dashboard.py
│   ├── gw_dashboard.py
│   ├── sac_dashboard.py
│   ├── webgpu_component.py
│   └── webgpu_client.html
│
├── scripts/                  ← Utility scripts
│   ├── benchmarks.py
│   ├── compiled_ops.py
│   └── sac_example.py
│
├── simulations/              ← Simulation scripts
│   ├── README.md
│   └── full_11d_holographic_simulation.py
│
├── tests/                    ← Test scripts
│   └── test_report_pipeline.py
│
├── configs/                  ← JSON configurations
│   ├── mera_network.json
│   ├── multi_vortex_config.json
│   └── vhl_ccsd_data.json
│
├── docs/                     ← All documentation
│   ├── DEPLOY_H100.md
│   ├── TD3_SAC_HYBRID_GUIDE.md
│   └── [10 other guides]
│
├── whitepapers/              ← Generated PDF reports
│   └── README.md
│
├── assets/                   ← Images, diagrams
├── archive/                  ← Deprecated code
└── utils/                    ← Helper utilities
```

**IMPORTANT**: The root directory is now CLEAN - no scattered .py files!
All Python code is organized in the `ivhl/` package with logical submodules.

---

## Questions to Ask User When Resuming

1. "What aspect of the iVHL framework would you like to work on?"
   - Holographic encoding
   - GW lattice analysis
   - RL discovery
   - Tensor network reconstruction
   - Visualization
   - Performance optimization

2. "Are you planning to run simulations locally or in Docker on H100?"
   - Affects optimization strategies and resource allocation

3. "Do you want to explore existing phenomena or discover new configurations?"
   - Determines whether to use existing simulations or create new ones

4. "Should I generate reports after completing tasks?"
   - Important for Docker deployments where data exfiltration is needed

---

## Final Notes

- **This is a research tool, not a physics theory**
- The framework is production-ready for computational experiments
- All LIGO integration is LIGO-*inspired*, not claiming to explain real GW data
- GFT and tensor networks are well-established theoretical frameworks we're implementing computationally
- Holographic resonance is an acoustic analogy to AdS/CFT, not a literal equivalence
- The goal is to explore whether structured boundary dynamics can encode emergent bulk geometry

**Most Important**: Read the current `README.md` after this file to understand the latest state and any updates made after this Hello_Claude.md was written.

---

**End of Hello_Claude.md**

When you (Claude) reconnect to this project:
1. Read this file first
2. Check git log for recent commits
3. Review README.md for current state
4. Ask user what they want to work on
5. Proceed with confidence knowing the full context

Good luck! 🚀
