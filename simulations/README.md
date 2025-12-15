# iVHL Simulations

This folder contains complete simulation scripts demonstrating the full iVHL framework capabilities.

## Available Simulations

### 1. Full 11D Holographic Simulation
**File**: `full_11d_holographic_simulation.py`

**Description**: Flagship simulation integrating all 11 dimensions of the iVHL framework with GPU auto-scaling.

**Dimensions**:
- **Boundary (2D+1)**: θ, φ, t (spherical coordinates + time)
- **Emergent Bulk (3D)**: x, y, z (reconstructed from holography)
- **GFT Colors (3D)**: c₁, c₂, c₃ (field color indices)
- **Spin (1D)**: s (helicity/angular momentum)
- **Tensor Rank (1D)**: r (MERA hierarchy level)

**Features**:
- Auto-detects GPU and scales parameters to saturate available memory
- H100 80GB: 2000 nodes, 64³ GFT grid, depth-5 MERA
- CPU fallback: 200 nodes, 16³ GFT grid, depth-2 MERA
- Generates comprehensive white paper with all analysis
- Integrates: resonance, GFT, tensor networks, GW perturbations, RL

**Usage**:
```bash
# Run with auto-scaling (recommended)
python simulations/full_11d_holographic_simulation.py

# Or from Docker
docker run --gpus all \
  -v $(pwd)/whitepapers:/app/whitepapers \
  ivhl-h100:latest \
  python simulations/full_11d_holographic_simulation.py
```

**Output**:
- White paper PDF in `whitepapers/report_YYYYMMDD_HHMMSS/`
- JSON data export
- Markdown summary
- LaTeX source

**Expected Runtime**:
- H100: ~30-60 seconds
- A100: ~45-90 seconds
- RTX 3090: ~2-3 minutes
- CPU: ~10-15 minutes

---

## How to Create New Simulations

### Template Structure

```python
#!/usr/bin/env python3
"""
Your Simulation Name
Description
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import iVHL modules
from vhl_holographic_resonance import HolographicResonanceSimulator
from gft_condensate_dynamics import GFTCondensateSimulator
# ... other imports

class YourSimulation:
    def __init__(self):
        # Setup
        pass

    def run(self):
        # Main simulation logic
        pass

    def generate_report(self):
        # Report generation
        from simulation_report_generator import IntegratedReportGenerator

        generator = IntegratedReportGenerator(
            output_base_dir='./whitepapers',
            auto_commit=False
        )

        report_files = generator.generate_full_report(
            simulation_type='your_simulation_name',
            configuration={...},
            results={...},
            analysis={...}
        )

if __name__ == '__main__':
    sim = YourSimulation()
    sim.run()
    sim.generate_report()
```

### Best Practices

1. **GPU Auto-Scaling**: Always detect GPU and scale parameters
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   if torch.cuda.is_available():
       memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
       # Scale params based on memory_gb
   ```

2. **Progress Reporting**: Print status at major phases
   ```python
   print(f"[1/5] Phase 1 name...")
   # ... work
   print(f"[2/5] Phase 2 name...")
   ```

3. **Always Generate Reports**: Use `IntegratedReportGenerator`
   ```python
   from simulation_report_generator import IntegratedReportGenerator
   generator = IntegratedReportGenerator(output_base_dir='./whitepapers')
   generator.generate_full_report(...)
   ```

4. **Save to whitepapers/**: All PDFs go to top-level whitepapers folder

5. **Handle CPU Fallback**: Reduce parameters if no GPU available

---

## Simulation Ideas

### Physics Explorations
- **Vortex Stability**: Test vortex pinning under GW perturbations
- **Phase Transitions**: GFT condensate formation dynamics
- **Entanglement Growth**: MERA depth vs entanglement entropy
- **Fractal Harmonics**: Systematic scan of constant residues

### RL Discovery
- **Optimal Lattice**: Use RL to find stable configurations
- **Memory Maximization**: Discover configurations with longest GW memory
- **Constant Emergence**: Train agent to produce specific mathematical constants

### Holographic Tests
- **RT Formula Validation**: Check S(A) = Area(γ)/4G numerically
- **Bulk Reconstruction**: Test geodesic vs tensor network predictions
- **AdS/CFT**: Boundary correlation functions vs bulk propagators

### Performance
- **Scaling Tests**: Benchmark across different GPUs
- **torch.compile**: Compare compiled vs eager mode
- **Memory Optimization**: Find optimal batch sizes

---

## Output Organization

```
whitepapers/
├── report_20251215_143000/          ← 11D simulation run 1
│   ├── report_20251215_143000.json  ← Full data
│   ├── report_20251215_143000.md    ← Summary
│   ├── whitepaper_20251215_143000.tex  ← LaTeX source
│   └── whitepaper_20251215_143000.pdf  ← Final PDF
├── report_20251215_150000/          ← 11D simulation run 2
│   └── ...
└── README.md
```

---

## Troubleshooting

### CUDA Out of Memory
- Simulation auto-scales, but if you still get OOM:
  ```python
  torch.cuda.empty_cache()
  ```
- Reduce batch sizes manually
- Use CPU fallback: `device = 'cpu'`

### PDF Compilation Fails
- LaTeX not installed locally (OK - Docker has it)
- Check `whitepaper_*.log` for pdflatex errors
- Verify texlive packages in Docker

### Slow Performance
- Check GPU utilization: `nvidia-smi`
- Enable torch.compile: `model = torch.compile(model)`
- Reduce unnecessary intermediate tensors
- Use in-place operations where possible

### Import Errors
- Run from repo root: `python simulations/your_sim.py`
- Or use absolute imports
- Check `sys.path.insert(0, ...)` is correct

---

## Contributing New Simulations

1. Create new Python file in `simulations/`
2. Follow template structure above
3. Include GPU auto-scaling
4. Generate white paper report
5. Add entry to this README
6. Test on CPU and GPU
7. Commit with descriptive message

---

**Last Updated**: 2025-12-15
**Framework**: iVHL (Vibrational Helical Lattice)
**Repository**: https://github.com/Zynerji/iVHL
