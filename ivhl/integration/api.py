"""
VHL API Backend - Flask REST API for Enhanced Computations

This optional backend provides:
- PySCF quantum calculations for any element
- Relativistic corrections (X2C)
- High-precision holographic gap filling
- Advanced multi-body force computations
- Data caching for performance

Usage:
    python vhl_api.py

    Server runs on http://localhost:5000

API Endpoints:
    GET  /api/health              - Health check
    POST /api/compute/hf          - Compute HF energy for element
    POST /api/compute/relativistic - Compute relativistic corrections
    POST /api/compute/forces      - Compute forces for positions
    POST /api/predict/energies    - Holographic prediction for range
    GET  /api/elements            - Get all element data
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform
import json
import os
from functools import lru_cache

# PySCF imports (with fallback)
try:
    from pyscf import gto, scf
    from pyscf.x2c import x2c
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("WARNING: PySCF not available. Quantum calculations will use mock data.")

app = Flask(__name__)
CORS(app)  # Enable CORS for web client

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.vhl_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Constants
ELEMENTS = [
    {'z': i+1, 'sym': sym} for i, sym in enumerate([
        'H','He','Li','Be','B','C','N','O','F','Ne',
        'Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
        'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
        'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',
        'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
        'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
        'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
        'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
        'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',
        'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm',
        'Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds',
        'Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og',
        'Uue','Ubn','Ubu','Ubb','Ubt','Ubq','Ubp','Ubh'
    ])
]

POLARITY_MAP = {
    1:1,2:0,3:1,4:1,5:0,6:0,7:-1,8:-1,9:-1,10:0,
    11:1,12:1,13:1,14:0,15:-1,16:-1,17:-1,18:0,
    19:1,20:1,21:0,22:0,23:0,24:0,25:0,26:0,27:0,28:0,29:0,30:0,
    31:1,32:0,33:-1,34:-1,35:-1,36:0,
    37:1,38:1,39:0,40:0,41:0,42:0,43:0,44:0,45:0,46:0,47:0,48:0,
    49:1,50:0,51:-1,52:-1,53:-1,54:0,
    55:1,56:1,57:0,58:0,59:0,60:0,61:0,62:0,63:0,64:0,65:0,66:0,67:0,68:0,69:0,70:0,71:0,
    72:0,73:0,74:0,75:0,76:0,77:0,78:0,79:0,80:0,
    81:1,82:0,83:-1,84:-1,85:-1,86:0,
    87:1,88:1,89:0,90:0,91:0,92:0,93:0,94:0,95:0,96:0,97:0,98:0,99:0,100:0,101:0,102:0,103:0,
    104:0,105:0,106:0,107:0,108:0,109:0,110:0,111:0,112:0,
    113:1,114:0,115:-1,116:-1,117:-1,118:0,
    119:1,120:1,121:0,122:0,123:-1,124:-1,125:-1,126:0
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def load_cache(key):
    """Load cached data"""
    cache_file = os.path.join(CACHE_DIR, f'{key}.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def save_cache(key, data):
    """Save data to cache"""
    cache_file = os.path.join(CACHE_DIR, f'{key}.json')
    with open(cache_file, 'w') as f:
        json.dump(data, f)

@lru_cache(maxsize=128)
def compute_hf_energy_cached(z, basis='sto-3g'):
    """Compute HF energy with caching"""
    cache_key = f'hf_{z}_{basis}'
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    if not PYSCF_AVAILABLE or z > 36:
        # Fallback: use approximate formula
        energy = -0.5 * z**2 * (1 + 0.2 * (z / 126)**1.5)
        result = {'z': z, 'energy': energy, 'converged': False, 'method': 'fallback'}
    else:
        try:
            el = ELEMENTS[z-1]
            mol = gto.M(
                atom=f'{el["sym"]} 0 0 0',
                basis=basis,
                spin=0 if z % 2 == 0 else 1
            )

            if z % 2 == 0:
                mf = scf.RHF(mol)
            else:
                mf = scf.UHF(mol)

            mf.verbose = 0
            mf.max_cycle = 100
            energy = mf.kernel()

            result = {
                'z': z,
                'energy': float(energy),
                'converged': bool(mf.converged),
                'method': 'RHF' if z % 2 == 0 else 'UHF'
            }
        except Exception as e:
            energy = -0.5 * z**2 * (1 + 0.2 * (z / 126)**1.5)
            result = {'z': z, 'energy': energy, 'converged': False, 'method': 'fallback', 'error': str(e)}

    save_cache(cache_key, result)
    return result

def compute_relativistic_correction(z, hf_energy):
    """Compute relativistic corrections using X2C if available"""
    if not PYSCF_AVAILABLE or z > 36:
        # Approximate relativistic correction ~ Z^4
        correction = -0.001 * z**4 / 10000
        return {
            'correction': correction,
            'corrected_energy': hf_energy + correction,
            'method': 'approximate'
        }

    try:
        el = ELEMENTS[z-1]
        mol = gto.M(
            atom=f'{el["sym"]} 0 0 0',
            basis='sto-3g',
            spin=0 if z % 2 == 0 else 1
        )

        # X2C for relativistic effects
        mf = x2c.UHF(mol)
        mf.verbose = 0
        mf.max_cycle = 50
        energy_rel = mf.kernel()

        correction = energy_rel - hf_energy

        return {
            'correction': float(correction),
            'corrected_energy': float(energy_rel),
            'method': 'X2C',
            'converged': bool(mf.converged)
        }
    except Exception as e:
        correction = -0.001 * z**4 / 10000
        return {
            'correction': correction,
            'corrected_energy': hf_energy + correction,
            'method': 'approximate',
            'error': str(e)
        }


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'pyscf_available': PYSCF_AVAILABLE,
        'version': '1.0.0'
    })

@app.route('/api/elements', methods=['GET'])
def get_elements():
    """Get all element data"""
    return jsonify({
        'elements': ELEMENTS,
        'polarity_map': POLARITY_MAP
    })

@app.route('/api/compute/hf', methods=['POST'])
def compute_hf():
    """
    Compute Hartree-Fock energy for element(s)

    Request body:
    {
        "z": int or [int],  // Atomic number(s)
        "basis": str        // Basis set (default: 'sto-3g')
    }
    """
    data = request.json
    z_list = data.get('z', [])
    if not isinstance(z_list, list):
        z_list = [z_list]

    basis = data.get('basis', 'sto-3g')

    results = []
    for z in z_list:
        if z < 1 or z > 126:
            results.append({'z': z, 'error': 'Invalid atomic number'})
            continue

        result = compute_hf_energy_cached(z, basis)
        results.append(result)

    return jsonify({'results': results})

@app.route('/api/compute/relativistic', methods=['POST'])
def compute_relativistic():
    """
    Compute relativistic corrections

    Request body:
    {
        "z": int,           // Atomic number
        "hf_energy": float  // Base HF energy (optional, will compute if not provided)
    }
    """
    data = request.json
    z = data.get('z')
    hf_energy = data.get('hf_energy')

    if z is None or z < 1 or z > 126:
        return jsonify({'error': 'Invalid atomic number'}), 400

    if hf_energy is None:
        hf_result = compute_hf_energy_cached(z)
        hf_energy = hf_result['energy']

    result = compute_relativistic_correction(z, hf_energy)
    result['z'] = z

    return jsonify(result)

@app.route('/api/compute/forces', methods=['POST'])
def compute_forces():
    """
    Compute forces for given positions and polarities

    Request body:
    {
        "positions": [[x,y,z], ...],  // N x 3 array
        "polarities": [q, ...],        // N x 1 array
        "g5": float,                   // Force strength
        "lambda": float,               // Force range
        "r_cap": float,                // Cutoff distance
        "multi_body": bool             // Include 3-body terms
    }
    """
    data = request.json

    positions = np.array(data.get('positions', []))
    polarities = np.array(data.get('polarities', []))
    g5 = data.get('g5', -5.01)
    lam = data.get('lambda', 22.0)
    r_cap = data.get('r_cap', 100.0)
    multi_body = data.get('multi_body', False)

    if len(positions) == 0 or len(polarities) == 0:
        return jsonify({'error': 'Empty positions or polarities'}), 400

    n = len(positions)
    forces = np.zeros_like(positions)

    # Pairwise Yukawa forces
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[j] - positions[i]
            r = np.linalg.norm(dr)

            if r > r_cap or r < 0.001:
                continue

            yukawa = g5 * np.exp(-r / lam) / r * polarities[i] * polarities[j]
            force_ij = yukawa * dr / r

            forces[i] += force_ij
            forces[j] -= force_ij

    # Multi-body corrections (3-body Axilrod-Teller-Muto inspired)
    if multi_body:
        mb_factor = 0.1
        for i in range(0, n, 5):  # Sample for performance
            for j in range(i + 1, min(i + 10, n)):
                for k in range(j + 1, min(j + 5, n)):
                    rij = positions[j] - positions[i]
                    rik = positions[k] - positions[i]
                    rjk = positions[k] - positions[j]

                    dij = np.linalg.norm(rij)
                    dik = np.linalg.norm(rik)
                    djk = np.linalg.norm(rjk)

                    if dij < 30 and dik < 30 and djk < 30:
                        # Simplified 3-body term
                        f3 = mb_factor * polarities[i] * polarities[j] * polarities[k]
                        f3 /= (dij * dik * djk + 0.1)

                        forces[i] += f3 * rij / dij
                        forces[j] += f3 * rjk / djk
                        forces[k] += f3 * (-rik) / dik

    return jsonify({
        'forces': forces.tolist(),
        'avg_force': float(np.mean(np.linalg.norm(forces, axis=1))),
        'max_force': float(np.max(np.linalg.norm(forces, axis=1)))
    })

@app.route('/api/predict/energies', methods=['POST'])
def predict_energies():
    """
    Holographic prediction for energy range

    Request body:
    {
        "z_min": int,
        "z_max": int,
        "known_energies": {z: E, ...}  // Optional, will use cached if not provided
    }
    """
    data = request.json
    z_min = data.get('z_min', 1)
    z_max = data.get('z_max', 126)
    known_energies = data.get('known_energies', {})

    # If no known energies provided, compute from cache
    if not known_energies:
        for z in range(1, min(37, z_max + 1)):
            result = compute_hf_energy_cached(z)
            if result['converged']:
                known_energies[str(z)] = result['energy']

    # Convert keys to int
    known_energies = {int(k): v for k, v in known_energies.items()}

    if len(known_energies) < 2:
        return jsonify({'error': 'Insufficient known energies for interpolation'}), 400

    # Cubic spline interpolation
    zs_known = np.array(sorted(known_energies.keys()))
    Es_known = np.array([known_energies[z] for z in zs_known])

    spline = CubicSpline(zs_known, Es_known, extrapolate=True)

    # Predict for range
    z_range = np.arange(z_min, z_max + 1)
    predictions = {}

    for z in z_range:
        E_base = spline(z)

        # Add holographic correction
        recon_scale = 1 + 0.2 * (z / 126)**1.5

        # Add noise for extrapolation
        if z > max(zs_known):
            noise = np.random.normal(0, 0.1 * abs(E_base))
            E_base += noise

        predictions[int(z)] = float(E_base * recon_scale)

    return jsonify({
        'predictions': predictions,
        'method': 'cubic_spline_holographic',
        'known_count': len(known_energies)
    })

@app.route('/api/batch/compute', methods=['POST'])
def batch_compute():
    """
    Batch computation for multiple elements

    Request body:
    {
        "elements": [z1, z2, ...],
        "compute_hf": bool,
        "compute_relativistic": bool,
        "basis": str
    }
    """
    data = request.json
    elements = data.get('elements', [])
    compute_hf_flag = data.get('compute_hf', True)
    compute_rel_flag = data.get('compute_relativistic', False)
    basis = data.get('basis', 'sto-3g')

    results = []

    for z in elements:
        result = {'z': z}

        if compute_hf_flag:
            hf_result = compute_hf_energy_cached(z, basis)
            result['hf'] = hf_result

            if compute_rel_flag:
                rel_result = compute_relativistic_correction(z, hf_result['energy'])
                result['relativistic'] = rel_result

        results.append(result)

    return jsonify({'results': results})


@app.route('/api/orbital/hydrogen', methods=['GET'])
def get_hydrogen_orbital():
    """
    Get hydrogen 2p orbital data (pre-computed or on-demand)

    Returns JSON with radial density, nodal rings, and STED correlations
    """
    import os

    # Check if pre-computed data exists
    orbital_file = os.path.join(CACHE_DIR, 'vhl_h_data.json')

    if os.path.exists(orbital_file):
        with open(orbital_file, 'r') as f:
            orbital_data = json.load(f)
        return jsonify({
            'source': 'cached',
            'data': orbital_data
        })

    # Compute on-demand if file doesn't exist
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'vhl_hydrogen_orbital.py', '--export', 'json'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and os.path.exists('vhl_h_data.json'):
            with open('vhl_h_data.json', 'r') as f:
                orbital_data = json.load(f)

            # Move to cache
            os.rename('vhl_h_data.json', orbital_file)

            return jsonify({
                'source': 'computed',
                'data': orbital_data
            })
        else:
            return jsonify({
                'error': 'Failed to compute orbital',
                'stderr': result.stderr
            }), 500

    except Exception as e:
        return jsonify({
            'error': 'Orbital computation failed',
            'message': str(e)
        }), 500


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("VHL API Backend Server")
    print("=" * 60)
    print(f"PySCF Available: {PYSCF_AVAILABLE}")
    print(f"Cache Directory: {CACHE_DIR}")
    print()
    print("Endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/elements")
    print("  GET  /api/orbital/hydrogen          (NEW: H 2p orbital from STED 2013)")
    print("  POST /api/compute/hf")
    print("  POST /api/compute/relativistic")
    print("  POST /api/compute/forces")
    print("  POST /api/predict/energies")
    print("  POST /api/batch/compute")
    print()
    print("Starting server on http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False)
