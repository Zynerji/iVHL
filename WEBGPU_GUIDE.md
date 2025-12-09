# VHL WebGPU Implementation Guide

## Overview

The VHL WebGPU implementation provides a high-performance, browser-based 3D visualization and simulation system with:

- **WebGPU Compute Shaders**: GPU-accelerated force calculations (60+ fps)
- **WebGL Fallback**: Compatibility for browsers without WebGPU
- **Three.js Rendering**: Smooth 60fps 3D visualization
- **TensorFlow.js ML**: Real-time predictions for superheavy elements
- **Flask API Integration**: Optional backend for enhanced computations
- **Standalone Operation**: Works offline with embedded data

## Quick Start

### Option 1: Standalone (No Backend)

Simply open the HTML file in a modern browser:

```bash
# Using a simple HTTP server
python -m http.server 8000
# Open http://localhost:8000/vhl_webgpu.html
```

Or directly:
```bash
open vhl_webgpu.html  # macOS
xdg-open vhl_webgpu.html  # Linux
start vhl_webgpu.html  # Windows
```

### Option 2: With Python Backend

For enhanced quantum calculations:

```bash
# Terminal 1: Start Flask API
python vhl_api.py

# Terminal 2: Serve HTML
python -m http.server 8000
```

Open `http://localhost:8000/vhl_webgpu.html`

## Browser Compatibility

### WebGPU Support (Preferred)
- ✅ Chrome 113+ (stable)
- ✅ Edge 113+
- ✅ Opera 99+
- ⚠️ Firefox (experimental, enable `dom.webgpu.enabled`)
- ⚠️ Safari (experimental, WebKit nightly builds)

### WebGL Fallback (Universal)
- ✅ All modern browsers
- Note: WebGL runs force calculations on CPU (slower but compatible)

### Checking WebGPU Support

Open browser console (F12) and run:
```javascript
if ('gpu' in navigator) {
    console.log('WebGPU is supported!');
} else {
    console.log('WebGPU not available, using WebGL fallback');
}
```

## Features & Controls

### 3D Visualization

**Mouse Controls:**
- **Left Click + Drag**: Rotate view
- **Right Click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Double Click**: Reset camera

**Keyboard Shortcuts:**
- `Space`: Toggle animation play/pause
- `R`: Reset simulation
- `F`: Focus on selected element
- `V`: Cycle force vector display
- `E`: Export current data

### UI Panel

#### Element Focus
Select any element (Z=1 to Z=126) to:
- Auto-zoom camera to element position
- Highlight with label overlay
- Display force vectors (if enabled)
- Show real-time stats

#### Geometry Controls

**Helix Radius** (2-20 Å)
- Controls base spiral radius
- Affects element spacing
- Default: 8.0 Å

**Fold Frequency** (1-10)
- Controls sinh/cosh modulation rate
- Higher values → more undulations
- Default: 5.0

**Helix Height** (40-150 Å)
- Total vertical extent
- Octave stacking distance
- Default: 80 Å

#### Fifth Force Parameters

**Force Strength (G5)** (-10 to 0)
- Yukawa coupling constant
- Negative → attractive for like charges
- Default: -5.01 (calibrated to ~10^-10 scale)

**Force Range (λ)** (10-50 Å)
- Yukawa screening length
- Determines interaction distance
- Default: 22 Å (nuclear/atomic scale)

**Multi-body Factor** (0-1)
- Enables 3-body Axilrod-Teller-Muto terms
- 0 = pairwise only, 1 = full multi-body
- Default: 0.1 (10% correction)

#### Visualization Options

**Particle Size** (0.1-1.0)
- Sphere radius in scene units
- Default: 0.3

**Force Vectors**
- None: No vector display
- All: Show all 126 force vectors
- Selected Only: Show only focused element

**Animation Speed** (0-3×)
- Dynamics timestep multiplier
- 0 = paused, 1 = real-time, 3 = fast-forward

### Simulation Controls

**▶️ Start Dynamics**
- Begins overdamped Langevin evolution
- Updates positions based on fifth-force
- Real-time force recomputation every 5 frames

**🔄 Reset**
- Regenerates geometry from parameters
- Resets simulation time to 0
- Clears trajectory history

**🤖 ML Predict Superheavies**
- Uses TensorFlow.js model trained on known HF energies
- Predicts energies for Z=119-126
- Displays results in popup dialog

**💾 Export Data**
- Downloads CSV with current state:
  - z, symbol, x, y, z, polarity, force_x, force_y, force_z
- Compatible with external analysis tools

### Statistics Display

- **Total Elements**: 126 (1-118 known + 119-126 speculative)
- **Polarity Counts**: +/−/0 distribution
- **Avg Force**: Mean force magnitude across all nodes
- **FPS**: Real-time frame rate (target: 60)
- **Simulation Time**: Accumulated dynamics time

### Status Indicator

- 🟢 **WebGPU**: GPU compute shaders active
- 🟠 **WebGL**: WebGL fallback (CPU forces)
- 🔴 **CPU**: Pure JavaScript computation

## Technical Architecture

### Force Computation Pipeline

#### WebGPU Path (Preferred)

1. **Buffer Creation**: Upload positions, polarities to GPU
2. **Compute Shader**: Execute Yukawa kernel in parallel
   - Workgroup size: 64 threads
   - Each thread computes forces for one element
   - O(N²) complexity parallelized
3. **Buffer Readback**: Copy results to CPU
4. **Multi-body CPU**: Apply 3-body corrections (sampled)

**Performance**: ~1-2ms for 126 elements on modern GPU

#### WebGL Fallback

1. **CPU Computation**: JavaScript double-loop
2. **Pairwise Yukawa**: Standard O(N²) iteration
3. **Multi-body Terms**: Sampled every 5th element

**Performance**: ~10-20ms for 126 elements

### WebGPU Compute Shader (WGSL)

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) { return; }

    let pi = positions[i];
    let qi = polarities[i];
    var force = vec3<f32>(0.0, 0.0, 0.0);

    for (var j = 0u; j < params.n; j = j + 1u) {
        if (i == j) { continue; }

        let pj = positions[j];
        let qj = polarities[j];
        let dr = pj - pi;
        let r = length(dr);

        if (r > params.rCap || r < 0.001) { continue; }

        // Yukawa: F = G5 * exp(-r/λ) / r * qi * qj * r̂
        let yukawa = params.g5 * exp(-r / params.lambda) / r * qi * qj;
        force = force + yukawa * dr / r;
    }

    forces[i] = force;
}
```

**Key Features**:
- Parallel execution across GPU cores
- Explicit memory access patterns
- Early exit for cutoff distances
- Vectorized math operations

### Three.js Rendering

**Scene Graph**:
```
Scene
├── AmbientLight (0.5 intensity)
├── PointLight1 (cyan, 20,20,20)
├── PointLight2 (red, -20,-20,20)
├── GridHelper (100×100, 20 divisions)
├── AxesHelper (50 units)
├── Points (particle system)
│   ├── BufferGeometry
│   │   ├── position (126×3 Float32Array)
│   │   └── color (126×3 Float32Array, polarity-coded)
│   └── PointsMaterial (size, vertexColors)
└── Sprites[] (element labels)
```

**Rendering Loop** (60fps):
1. Update controls (orbital damping)
2. If running: update dynamics, recompute forces
3. Update particle positions (GPU buffer)
4. Update force vector arrows (if enabled)
5. Render scene
6. Update FPS counter

### TensorFlow.js ML Model

**Architecture**:
```
Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  // Energy prediction
])
```

**Training**:
- **Input Features**: [Z, Z², Z^1.5] (captures periodic trends)
- **Labels**: HF energies from PySCF (embedded in HTML)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Mean Squared Error
- **Epochs**: 100 (client-side, ~2 seconds)
- **Normalization**: z-score on features

**Prediction**:
```javascript
const features = [z, z*z, Math.pow(z, 1.5)];
const normalized = normalize(features);  // Using stored mean/std
const energy = model.predict(tf.tensor2d([normalized]));
```

**Accuracy**: ~90% for interpolation (Z<118), ~70% for extrapolation (Z>118)

## Flask API Integration

### Starting the Backend

```bash
python vhl_api.py
# Server runs on http://localhost:5000
```

### Available Endpoints

#### Health Check
```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "ok",
  "pyscf_available": true,
  "version": "1.0.0"
}
```

#### Compute HF Energy
```bash
curl -X POST http://localhost:5000/api/compute/hf \
  -H "Content-Type: application/json" \
  -d '{"z": [1, 2, 3], "basis": "sto-3g"}'
```

Response:
```json
{
  "results": [
    {"z": 1, "energy": -0.466, "converged": true, "method": "RHF"},
    {"z": 2, "energy": -2.835, "converged": true, "method": "RHF"},
    {"z": 3, "energy": -7.315, "converged": true, "method": "UHF"}
  ]
}
```

#### Relativistic Corrections
```bash
curl -X POST http://localhost:5000/api/compute/relativistic \
  -H "Content-Type: application/json" \
  -d '{"z": 79}'
```

Response:
```json
{
  "z": 79,
  "correction": -1.234,
  "corrected_energy": -18876.543,
  "method": "X2C",
  "converged": true
}
```

#### Compute Forces (GPU Backend)
```bash
curl -X POST http://localhost:5000/api/compute/forces \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [[0,0,0], [1,1,1], [2,2,2]],
    "polarities": [1, -1, 0],
    "g5": -5.01,
    "lambda": 22.0,
    "multi_body": true
  }'
```

#### Holographic Prediction
```bash
curl -X POST http://localhost:5000/api/predict/energies \
  -H "Content-Type: application/json" \
  -d '{"z_min": 119, "z_max": 126}'
```

### Connecting Frontend to Backend

**Automatic Detection**:
The HTML checks for backend at startup:

```javascript
async function checkBackend() {
    try {
        const response = await fetch('http://localhost:5000/api/health');
        if (response.ok) {
            console.log('Backend available, enabling enhanced features');
            return true;
        }
    } catch (e) {
        console.log('Backend unavailable, using client-side only');
    }
    return false;
}
```

**Enhanced Features with Backend**:
- Real-time PySCF calculations (not just embedded data)
- X2C relativistic corrections
- High-precision force computations
- Unlimited element range
- Persistent caching

## Performance Optimization

### WebGPU Optimizations

1. **Workgroup Size**: 64 threads (optimal for most GPUs)
2. **Buffer Reuse**: Create once, update per frame
3. **Async Compute**: Non-blocking GPU operations
4. **Cutoff Distance**: r_cap=100 Å reduces O(N²) overhead

### Rendering Optimizations

1. **Point Sprites**: More efficient than individual meshes
2. **Frustum Culling**: Automatic in Three.js
3. **LOD Labels**: Only show labels when zoomed
4. **Geometry Instancing**: Future enhancement for bonds

### Memory Management

1. **Typed Arrays**: Float32Array for positions/forces
2. **Buffer Pooling**: Reuse GPU buffers
3. **Tensor Disposal**: Explicit cleanup in TensorFlow.js
4. **Sparse Force Vectors**: Only create when visible

### Bottlenecks & Solutions

| Bottleneck | Solution |
|------------|----------|
| Force computation (CPU) | WebGPU compute shaders |
| Many force vectors | Show only selected/sampled |
| Label rendering | Canvas texture atlas |
| High particle count | Point sprites + instancing |
| Memory leaks | Explicit dispose() calls |

## Advanced Usage

### Custom Force Functions

Modify the WebGPU shader to implement custom potentials:

```wgsl
// Replace Yukawa with Lennard-Jones
let lj = 4.0 * epsilon * (pow(sigma/r, 12.0) - pow(sigma/r, 6.0));
force = force + lj * dr / r;
```

### Multi-Body Extensions

Add 4-body terms for higher accuracy:

```javascript
// In computeForcesCPU()
for (let i = 0; i < n; i++) {
    for (let j = i+1; j < n; j++) {
        for (let k = j+1; k < n; k++) {
            for (let l = k+1; l < n; l++) {
                // 4-body term
                const f4 = computeQuadrupole(i, j, k, l);
                forces[i] += f4;
            }
        }
    }
}
```

### Custom ML Models

Train on different properties:

```javascript
// Replace energy prediction with ionization energy
const ionizationModel = tf.sequential({
    layers: [
        tf.layers.dense({inputShape: [5], units: 128, activation: 'relu'}),
        tf.layers.dense({units: 1})
    ]
});

// Train on NIST ionization data
await ionizationModel.fit(zData, ionizationData, {epochs: 200});
```

### Integration with Other Tools

**Export to Blender**:
```javascript
// Modify exportData() to output .obj format
let obj = 'o VHL_Lattice\n';
for (let i = 0; i < N_NODES; i++) {
    obj += `v ${positions[i*3]} ${positions[i*3+1]} ${positions[i*3+2]}\n`;
}
// Download as .obj file
```

**Export to ParaView**:
```javascript
// Export as VTK format for scientific visualization
let vtk = '# vtk DataFile Version 3.0\nVHL Simulation\nASCII\n';
vtk += 'DATASET POLYDATA\n';
vtk += `POINTS ${N_NODES} float\n`;
// Add point data
```

## Troubleshooting

### WebGPU Not Available

**Symptoms**: Status shows "WebGL" or "CPU" instead of "WebGPU"

**Solutions**:
1. Update browser to latest version
2. Enable hardware acceleration in browser settings
3. Check GPU drivers are up to date
4. Try Chrome/Edge (best WebGPU support)

**Verify GPU**:
```bash
# Chrome: chrome://gpu
# Look for "WebGPU: Hardware accelerated"
```

### Low FPS

**Symptoms**: FPS counter shows <30

**Solutions**:
1. Reduce particle count (modify N_NODES)
2. Disable force vectors (set to "None")
3. Reduce animation speed
4. Close other GPU-intensive applications
5. Lower browser zoom level

### ML Model Not Training

**Symptoms**: Predictions are inaccurate or error messages

**Solutions**:
1. Check browser console for TensorFlow.js errors
2. Ensure sufficient known HF energies (min 10)
3. Increase training epochs
4. Use backend API for pre-trained model

### Backend Connection Failed

**Symptoms**: "Backend unavailable" in console

**Solutions**:
1. Verify Flask server is running: `curl http://localhost:5000/api/health`
2. Check CORS is enabled (Flask-CORS installed)
3. Try different port if 5000 is occupied
4. Disable browser CORS blocking (dev mode only)

### Memory Leaks

**Symptoms**: Tab becomes slow over time, high memory usage

**Solutions**:
1. Explicit cleanup in reset():
   ```javascript
   particleSystem.geometry.dispose();
   particleSystem.material.dispose();
   forceVectors.forEach(v => v.dispose());
   ```
2. Dispose TensorFlow.js tensors after use
3. Refresh page periodically for long sessions

## Future Enhancements

### Planned Features

- [ ] **VR/AR Mode**: WebXR integration for immersive exploration
- [ ] **Quantum States**: Visualize electron orbitals (Schrödinger solutions)
- [ ] **Time Evolution**: FFT spectrum visualization (phonon modes)
- [ ] **Collaborative Mode**: Multi-user via WebRTC
- [ ] **Material Export**: Save to .blend, .fbx, .gltf
- [ ] **Shader Editor**: Real-time WGSL code editing
- [ ] **Performance Profiler**: Built-in GPU/CPU timing
- [ ] **Mobile Support**: Touch controls, reduced poly count

### Research Extensions

- [ ] **Relativistic Forces**: Retarded potentials for high-Z
- [ ] **QCD Coupling**: Gluon exchange at nuclear scales
- [ ] **Dark Matter**: BSM scalar field integration
- [ ] **Holographic Bulk**: Full AdS-CFT lattice reconstruction
- [ ] **Machine Learning**: GNN for graph-based force prediction

## References

### WebGPU
- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
- [WebGPU Samples](https://webgpu.github.io/webgpu-samples/)

### Three.js
- [Three.js Docs](https://threejs.org/docs/)
- [Three.js Examples](https://threejs.org/examples/)

### TensorFlow.js
- [TensorFlow.js Guide](https://www.tensorflow.org/js/guide)
- [Models & Layers API](https://js.tensorflow.org/api/latest/)

### Physics
- Walter Russell (1926). *The Universal One*
- Yukawa, H. (1935). *On the Interaction of Elementary Particles*
- Axilrod, B.M., Teller, E. (1943). *Interaction of Three Atoms*

## License

MIT License - See LICENSE file for details.

## Support

- **Issues**: https://github.com/Zynerji/Vibrational-Helix-Lattice/issues
- **Discussions**: https://github.com/Zynerji/Vibrational-Helix-Lattice/discussions
- **Email**: support@vhl-simulation.org

---

*Built with ❤️ for exploratory physics research*
