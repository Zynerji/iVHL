# 🌀 Vibrational Helix Lattice (VHL) Simulation

**Fusion of the Periodic Table, Cymatics, and Quantum Mechanics**

A computational framework reimagining the periodic table as a 3D helical structure, integrating Walter Russell's octave cosmology, holographic physics, and speculative fifth-force dynamics.

![VHL Concept](https://img.shields.io/badge/Status-Research-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

The VHL Simulation explores a speculative 3D helical model of the periodic table where:
- Elements emerge as **resonant nodes** in polarized waves (inspired by Walter Russell, 1926)
- Geometry incorporates **hyperbolic folding** (sinh/cosh saddle creases, evoking AdS-CFT holographic boundaries)
- **Polarity charges** (+expansion, -contraction, 0-equilibrium) drive a Yukawa fifth-force
- **Quantum energies** computed via PySCF (Hartree-Fock) with holographic gap reconstruction
- **Cymatics integration**: Vibrational patterns mirror Chladni figures and spherical harmonics
- **Interactive 3D visualization** with Streamlit and Plotly

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Zynerji/Vibrational-Helix-Lattice.git
cd Vibrational-Helix-Lattice
pip install -r requirements.txt
```

### Run Python Simulation (Streamlit)

```bash
streamlit run vhl_sim.py
```

Browser opens at `http://localhost:8501`

### Run WebGPU Version (Browser-Based) 🆕

**Standalone (No Installation Required)**:
```bash
# Serve with simple HTTP server
python -m http.server 8000
# Open http://localhost:8000/vhl_webgpu.html
```

**With Backend API** (for enhanced quantum calculations):
```bash
# Terminal 1: Start Flask API
python vhl_api.py

# Terminal 2: Serve HTML
python -m http.server 8000
# Open http://localhost:8000/vhl_webgpu.html
```

**Browser Requirements**:
- ✅ Chrome/Edge 113+ (WebGPU support)
- ✅ Firefox/Safari (WebGL fallback)
- 60fps real-time rendering
- GPU-accelerated force computations

**See [WEBGPU_GUIDE.md](WEBGPU_GUIDE.md) for detailed usage**

## ✨ Features

### Core Capabilities
- **126 Elements**: Known elements (Z=1-118) + speculative superheavies (Z=119-126)
- **Hyperbolic Helix**: sinh/cosh modulated 3D geometry with configurable folding
- **Polarity System**: Maps to valence configurations (alkalis +, halogens -, nobles 0)
- **Quantum Integration**: PySCF Hartree-Fock energies for Z≤36 with holographic extrapolation
- **Fifth Force**: Yukawa potential from polarity mismatches (hints from Ca isotope anomalies)
- **Dynamics Simulation**: Overdamped Langevin evolution under fifth-force
- **FFT Spectrum**: Vibrational mode analysis (phonon analog)

### Interactive UI (Streamlit)
- **3D Plotly Visualization**: Rotate, zoom, explore the helical lattice
- **Element Focus**: Camera zoom to any element (Z=1 to Z=126)
- **Real-Time Controls**: Adjust helix radius, fold frequency, force parameters
- **Analysis Plots**: Energy curves, FFT spectra, force distributions
- **Data Export**: Download trajectory as CSV for external analysis

### WebGPU Features 🚀 NEW
- **GPU-Accelerated Forces**: WebGPU compute shaders (1-2ms for 126 elements)
- **60fps Three.js Rendering**: Smooth real-time 3D visualization with orbital controls
- **TensorFlow.js ML**: Client-side superheavy element predictions
- **Multi-Body Forces**: 3-body Axilrod-Teller-Muto corrections
- **Relativistic API**: X2C corrections via Flask backend (optional)
- **Standalone Operation**: Works offline with embedded quantum data
- **Universal Compatibility**: WebGL fallback for older browsers
- **Interactive Labels**: Click elements for detailed info, force vector display
- **Live Statistics**: FPS counter, force magnitudes, simulation time

## 📊 Scientific Foundations

### 1. Helical Geometry
Base helix with hyperbolic folding:
```
r(θ) = r_base + a·sinh(ω·θ)
z(θ) = h·t + b·cosh(ω·θ)
```

### 2. Polarity Assignment
Elements mapped to {-1, 0, +1} based on valence:
- **+1**: Alkalis, alkaline earths (Li, Na, K, Ca, ...)
- **-1**: Halogens, chalcogens (F, Cl, O, S, ...)
- **0**: Noble gases, transition metals (He, Ne, Ar, ...)

### 3. Quantum Calculations
Hartree-Fock via PySCF:
```python
mol = gto.M(atom='H 0 0 0', basis='sto-3g')
mf = scf.RHF(mol)
energy = mf.kernel()  # ~80% of experimental energy
```

### 4. Holographic Gap Filling
Missing energies (Z>36, superheavies) reconstructed via cubic spline:
```
E_recon(Z) = Spline(E_known)[Z] × (1 + 0.2·(Z/Z_max)^1.5)
```

### 5. Fifth Force (Yukawa)
```
F_ij = G5 · exp(-r/λ) / r · q_i·q_j · r̂
```
- `G5 = -5.01` (calibrated to ~10^-10 scale)
- `λ = 22 Å` (atomic/nuclear range)

### 6. Vibrational Spectrum
FFT on dynamics:
```
FFT[x(t)] → Power(ω)
```
Analogous to Raman/IR spectroscopy for crystals.

## 📁 Project Structure

```
Vibrational-Helix-Lattice/
├── vhl_sim.py           # Python/Streamlit simulation engine
├── vhl_webgpu.html      # WebGPU/WebGL browser version (standalone)
├── vhl_api.py           # Flask REST API backend (optional)
├── requirements.txt     # Python dependencies
├── DOCUMENTATION.md     # Python implementation technical docs
├── WEBGPU_GUIDE.md      # WebGPU implementation guide
└── README.md            # This file
```

## 🎮 Usage Examples

### 1. Explore Octave Structure
Focus on noble gases (He, Ne, Ar, Kr, Xe, Rn, Og) to see Russell's "stillness points"

### 2. Study Polarity Patterns
Select alkali metals (Li, Na, K, Rb, Cs) - all show +1 (red) polarity

### 3. Analyze Superheavies
Zoom to Z=119-126 and examine holographically extrapolated energies

### 4. Tune Fifth Force
- Increase `|G5|` → stronger clustering
- Decrease `λ` → more localized interactions
- Observe FFT spectrum changes

### 5. Export Data
Download CSV for custom analysis (matplotlib animations, network graphs, etc.)

## 🔬 Reality Correlations

| VHL Component | Physical Anchor |
|--------------|-----------------|
| Helical positions | SE radial functions (R_nl), cymatic nodal lines |
| Polarity | Ionization energies (NIST), Pauling electronegativity |
| HF energies | PySCF calculations (~80% of exp. values) |
| Gap reconstruction | Multi-body/relativistic corrections (~20%) |
| Fifth force | Ca isotope charge radii hints (ETH Zurich 2025) |
| FFT spectrum | Phonon modes (Raman shifts) |

## 🛠️ Extensions

### ✅ Implemented
- [x] GPU acceleration (WebGPU compute shaders)
- [x] Relativistic corrections (PySCF X2C via Flask API)
- [x] Multi-body forces (3-body Axilrod-Teller-Muto)
- [x] Machine learning (TensorFlow.js for superheavy predictions)
- [x] WebGL/WebGPU real-time rendering (60fps)

### 🔮 Planned
- [ ] VR/AR mode (WebXR integration)
- [ ] Quantum orbital visualization (electron density)
- [ ] Experimental validation (Raman spectra comparison)
- [ ] Jupyter notebook tutorials
- [ ] Mobile touch controls
- [ ] Collaborative multi-user mode (WebRTC)

## 📚 References

1. Russell, W. (1926). *The Universal One*. University of Science and Philosophy.
2. Maldacena, J. (1999). *AdS-CFT Correspondence*. arXiv:hep-th/9711200.
3. Ryu, S., Takayanagi, T. (2006). *Holographic Entanglement Entropy*. Phys. Rev. Lett.
4. PySCF Documentation: https://pyscf.org
5. Chladni, E. (1787). *Entdeckungen über die Theorie des Klanges*.

## ⚠️ Disclaimer

This is a **research/educational tool** exploring speculative physics:
- Superheavy elements (Z>118) are theoretical extrapolations
- Fifth-force parameters are calibrated to hypothetical anomalies
- Holographic gap-filling is a mathematical interpolation, not a physical theory

Results should be interpreted as **exploratory models**, not validated predictions.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Improved quantum calculations (DFT, coupled-cluster)
- Alternative polarity schemes (machine-learned from data)
- Integration with experimental databases (NIST, CODATA)
- Visualization enhancements (VR, animations)

## 📄 License

MIT License - See LICENSE file for details.

## 📧 Contact

**Project**: Vibrational Helix Lattice
**Repository**: https://github.com/Zynerji/Vibrational-Helix-Lattice
**Issues**: https://github.com/Zynerji/Vibrational-Helix-Lattice/issues

---

*"The universe is a symphony of vibrating strings, and the mind of God that Einstein eloquently wrote about for thirty years would be cosmic music resonating through eleven-dimensional hyperspace."* — Michio Kaku
