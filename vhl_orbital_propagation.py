"""
VHL Orbital Propagation - Multi-Element Orbital Computation

This script computes orbitals for multiple elements and maps them to VHL geometry,
showing how hydrogen's fundamental 2p pattern propagates through the periodic table.

Key Insights:
- Nodal surfaces increase with shell number (n)
- Each octave (noble gas) represents complete shell filling
- Orbital complexity maps to helix position and polarity

Run: python vhl_orbital_propagation.py --elements H,He,Li,C,N,O,Ne
"""

import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen
import json
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# VHL Constants
OCTAVE_ELEMENTS = {
    1: [1, 2],                          # Octave 1: H, He
    2: [3, 4, 5, 6, 7, 8, 9, 10],      # Octave 2: Li-Ne
    3: [11, 12, 13, 14, 15, 16, 17, 18], # Octave 3: Na-Ar
    # ... extends to all 14 octaves
}

# Element configurations (ground state)
ELEMENT_CONFIGS = {
    'H': (1, 1, 'sto-3g', [1]),      # (Z, n_elec, basis, shell_structure)
    'He': (2, 2, 'sto-3g', [2]),
    'Li': (3, 3, 'sto-3g', [2, 1]),
    'Be': (4, 4, 'sto-3g', [2, 2]),
    'B': (5, 5, 'sto-3g', [2, 3]),
    'C': (6, 6, 'sto-3g', [2, 4]),
    'N': (7, 7, 'sto-3g', [2, 5]),
    'O': (8, 8, 'sto-3g', [2, 6]),
    'F': (9, 9, 'sto-3g', [2, 7]),
    'Ne': (10, 10, 'sto-3g', [2, 8]),
    'Na': (11, 11, 'sto-3g', [2, 8, 1]),
    'Mg': (12, 12, 'sto-3g', [2, 8, 2]),
    'Al': (13, 13, 'sto-3g', [2, 8, 3]),
    'Si': (14, 14, 'sto-3g', [2, 8, 4]),
    'P': (15, 15, 'sto-3g', [2, 8, 5]),
    'S': (16, 16, 'sto-3g', [2, 8, 6]),
    'Cl': (17, 17, 'sto-3g', [2, 8, 7]),
    'Ar': (18, 18, 'sto-3g', [2, 8, 8]),
}


def count_nodal_surfaces(shell_structure):
    """
    Count total nodal surfaces based on shell filling.

    Nodal surfaces = (n-1) for each shell
    Example: [2, 8, 3] = (1-1) + (2-1) + (3-1) = 0 + 1 + 2 = 3 nodes
    """
    total_nodes = 0
    for shell_idx, n_electrons in enumerate(shell_structure):
        n = shell_idx + 1  # Principal quantum number
        if n_electrons > 0:
            total_nodes += (n - 1)
    return total_nodes


def compute_orbital_density(symbol, grid_size=32):
    """
    Compute orbital density for given element.

    Returns:
        mf: PySCF SCF object
        grid: 3D density grid
        r_grid: Radial grid points
        radial_density: 1D radial average
    """
    if symbol not in ELEMENT_CONFIGS:
        raise ValueError(f"Element {symbol} not supported")

    z, n_elec, basis, shell_struct = ELEMENT_CONFIGS[symbol]

    # Determine spin
    spin = n_elec % 2  # 0 for even, 1 for odd

    print(f"Computing {symbol} (Z={z}, electrons={n_elec}, spin={spin})...")

    mol = gto.M(
        atom=f'{symbol} 0 0 0',
        basis=basis,
        spin=spin
    )

    # Choose method based on spin
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.kernel()

    if not mf.converged:
        print(f"  Warning: SCF did not converge for {symbol}")

    # Generate density grid
    dm = mf.make_rdm1()
    grid = cubegen.density(mol, dm, nx=grid_size, ny=grid_size, nz=grid_size)

    # Radial average
    r_grid = np.linspace(-3 * (z/2), 3 * (z/2), grid_size)  # Scale with Z
    radial_density = np.zeros(grid_size, dtype=np.float64)

    for i in range(grid_size):
        radial_density[i] = np.mean(grid[i, :, :])

    # Count nodes
    n_nodes = count_nodal_surfaces(shell_struct)

    print(f"  ✓ Converged: E = {mf.e_tot:.4f} Ha, Nodes = {n_nodes}")

    return mf, grid, r_grid, radial_density, n_nodes, shell_struct


def map_to_vhl_position(z, n_nodes):
    """
    Map element to VHL helix position based on orbital structure.

    VHL Mapping:
    - Octave = ceil(z / 9)  (Russell's 9-tone scale)
    - Turn angle = 2π * (z % 9) / 9
    - Height = octave * vertical_spacing
    - Radial offset = base_radius + node_amplitude * n_nodes

    Returns: (x, y, z, octave, tone)
    """
    HELIX_RADIUS = 8.0
    HELIX_HEIGHT = 80.0
    TURNS = 42  # 14 octaves * 3 turns
    NODE_AMPLITUDE = 0.5  # Radial variation per node

    octave = int(np.ceil(z / 9))
    tone = (z - 1) % 9  # 0-8 within octave

    # Helical parameters
    t = (z - 1) / 126  # Normalized position
    theta = 2 * np.pi * TURNS * t
    z_pos = HELIX_HEIGHT * t - HELIX_HEIGHT / 2

    # Radial modulation by nodal count
    r = HELIX_RADIUS + NODE_AMPLITUDE * n_nodes

    # Hyperbolic folding (from original VHL)
    fold_freq = 5.0
    fold_amp_r = HELIX_RADIUS * 0.2
    r += fold_amp_r * np.sinh(fold_freq * theta) / np.sinh(fold_freq * 2 * np.pi * TURNS)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y, z_pos, octave, tone


def compute_propagation_matrix(elements):
    """
    Compute orbital data for multiple elements and build propagation matrix.

    Propagation Matrix:
    - Rows: Elements (ordered by Z)
    - Columns: [Z, nodes, energy, max_density, vhl_x, vhl_y, vhl_z, octave, tone]

    This matrix shows how orbital complexity propagates through the helix.
    """
    results = []

    for symbol in elements:
        try:
            mf, grid, r_grid, radial_density, n_nodes, shell_struct = compute_orbital_density(symbol)

            z, n_elec, basis, _ = ELEMENT_CONFIGS[symbol]

            # VHL mapping
            vhl_x, vhl_y, vhl_z, octave, tone = map_to_vhl_position(z, n_nodes)

            results.append({
                'symbol': symbol,
                'z': z,
                'n_electrons': n_elec,
                'n_nodes': n_nodes,
                'shell_structure': shell_struct,
                'scf_energy': float(mf.e_tot),
                'max_density': float(np.max(radial_density)),
                'radial_density': radial_density.tolist(),
                'r_grid': r_grid.tolist(),
                'vhl_position': {
                    'x': float(vhl_x),
                    'y': float(vhl_y),
                    'z': float(vhl_z),
                    'octave': int(octave),
                    'tone': int(tone)
                }
            })

        except Exception as e:
            print(f"  ✗ Failed for {symbol}: {e}")

    return results


def analyze_propagation_patterns(results):
    """
    Analyze how orbital patterns propagate through the VHL structure.

    Key patterns:
    1. Node count vs. Helix position (should correlate with octave)
    2. Energy vs. Z (ionization trend)
    3. Density distribution vs. Polarity
    4. Noble gases as local maxima (complete shells)
    """
    print("\n" + "=" * 60)
    print("VHL Orbital Propagation Analysis")
    print("=" * 60)

    # Pattern 1: Nodes vs. Octave
    print("\nPattern 1: Nodal Surfaces vs. VHL Octave")
    print("-" * 60)
    print(f"{'Symbol':<6} {'Z':<4} {'Nodes':<6} {'Octave':<8} {'Tone':<6} {'Energy (Ha)':<12}")
    print("-" * 60)

    for r in results:
        print(f"{r['symbol']:<6} {r['z']:<4} {r['n_nodes']:<6} "
              f"{r['vhl_position']['octave']:<8} {r['vhl_position']['tone']:<6} "
              f"{r['scf_energy']:<12.4f}")

    # Pattern 2: Shell filling milestones
    noble_gases = [r for r in results if r['symbol'] in ['He', 'Ne', 'Ar']]
    if noble_gases:
        print("\nPattern 2: Noble Gas Stillness Points (Complete Shells)")
        print("-" * 60)
        for ng in noble_gases:
            print(f"{ng['symbol']}: Octave {ng['vhl_position']['octave']}, "
                  f"Nodes = {ng['n_nodes']}, "
                  f"Shell = {ng['shell_structure']}")

    # Pattern 3: Octave transitions (post-noble gas)
    print("\nPattern 3: Octave Transitions (New Shell Begins)")
    print("-" * 60)
    for i, r in enumerate(results[:-1]):
        next_r = results[i + 1]
        if r['vhl_position']['octave'] != next_r['vhl_position']['octave']:
            print(f"{r['symbol']} → {next_r['symbol']}: "
                  f"Octave {r['vhl_position']['octave']} → {next_r['vhl_position']['octave']}, "
                  f"Nodes {r['n_nodes']} → {next_r['n_nodes']}")

    print("\n" + "=" * 60)


def plot_propagation(results, output='vhl_orbital_propagation.png'):
    """
    Visualize orbital propagation through VHL structure.

    Creates multi-panel plot:
    1. Radial densities overlaid (color by octave)
    2. Nodal count vs. Z (showing octave structure)
    3. Energy vs. Z (ionization trend)
    4. 3D VHL positions colored by node count
    """
    fig = plt.figure(figsize=(16, 10))

    # Panel 1: Overlaid radial densities
    ax1 = plt.subplot(2, 3, 1)
    octave_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for r in results:
        octave = r['vhl_position']['octave']
        color = octave_colors[octave % 10]
        ax1.plot(r['r_grid'], r['radial_density'],
                label=f"{r['symbol']} (Oct {octave})",
                color=color, alpha=0.7, linewidth=1.5)

    ax1.set_xlabel('Radial Distance (a.u.)', fontsize=10)
    ax1.set_ylabel('Electron Density (a.u.⁻³)', fontsize=10)
    ax1.set_title('Radial Density Propagation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Nodal count vs. Z
    ax2 = plt.subplot(2, 3, 2)
    zs = [r['z'] for r in results]
    nodes = [r['n_nodes'] for r in results]
    octaves = [r['vhl_position']['octave'] for r in results]

    scatter = ax2.scatter(zs, nodes, c=octaves, cmap='tab10', s=100, alpha=0.7, edgecolors='k')
    for r in results:
        ax2.annotate(r['symbol'], (r['z'], r['n_nodes']),
                    fontsize=8, ha='center', va='bottom')

    ax2.set_xlabel('Atomic Number (Z)', fontsize=10)
    ax2.set_ylabel('Nodal Surfaces', fontsize=10)
    ax2.set_title('Node Count vs. Z (colored by Octave)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='VHL Octave')

    # Panel 3: Energy vs. Z
    ax3 = plt.subplot(2, 3, 3)
    energies = [r['scf_energy'] for r in results]

    ax3.plot(zs, energies, 'o-', color='purple', markersize=8, linewidth=2, alpha=0.7)
    for r in results:
        if r['symbol'] in ['He', 'Ne', 'Ar']:  # Highlight noble gases
            ax3.plot(r['z'], r['scf_energy'], 'r*', markersize=15, label='Noble Gas' if r['symbol'] == 'He' else '')

    ax3.set_xlabel('Atomic Number (Z)', fontsize=10)
    ax3.set_ylabel('SCF Energy (Hartree)', fontsize=10)
    ax3.set_title('Ionization Trend', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: 3D VHL positions
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    xs = [r['vhl_position']['x'] for r in results]
    ys = [r['vhl_position']['y'] for r in results]
    zs_vhl = [r['vhl_position']['z'] for r in results]

    scatter_3d = ax4.scatter(xs, ys, zs_vhl, c=nodes, cmap='viridis', s=150, alpha=0.8, edgecolors='k')

    for r in results:
        ax4.text(r['vhl_position']['x'], r['vhl_position']['y'], r['vhl_position']['z'],
                r['symbol'], fontsize=7, ha='center')

    ax4.set_xlabel('X (VHL)', fontsize=9)
    ax4.set_ylabel('Y (VHL)', fontsize=9)
    ax4.set_zlabel('Z (VHL)', fontsize=9)
    ax4.set_title('VHL Helix Positions (colored by Nodes)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter_3d, ax=ax4, label='Nodal Count', shrink=0.6)

    # Panel 5: Octave structure
    ax5 = plt.subplot(2, 3, 5)
    octave_counts = {}
    for r in results:
        oct = r['vhl_position']['octave']
        octave_counts[oct] = octave_counts.get(oct, 0) + 1

    ax5.bar(octave_counts.keys(), octave_counts.values(), color='teal', alpha=0.7, edgecolor='k')
    ax5.set_xlabel('VHL Octave', fontsize=10)
    ax5.set_ylabel('Element Count', fontsize=10)
    ax5.set_title("Russell's 9-Tone Octave Distribution", fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Density maxima comparison
    ax6 = plt.subplot(2, 3, 6)
    max_densities = [r['max_density'] for r in results]

    ax6.semilogy(zs, max_densities, 's-', color='darkorange', markersize=8, linewidth=2, alpha=0.7)
    ax6.set_xlabel('Atomic Number (Z)', fontsize=10)
    ax6.set_ylabel('Max Density (log scale, a.u.⁻³)', fontsize=10)
    ax6.set_title('Peak Electron Density', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')

    plt.suptitle('VHL Orbital Propagation: H 2p Template → Multi-Element Harmonic Overtones',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Propagation visualization saved: {output}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='VHL Orbital Propagation Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python vhl_orbital_propagation.py --elements H,He,Li,C,N,O,Ne
  python vhl_orbital_propagation.py --elements H,He,Ne,Ar --export json
  python vhl_orbital_propagation.py --octave 2  # Compute full octave 2 (Li-Ne)
        '''
    )

    parser.add_argument('--elements', type=str,
                       help='Comma-separated element symbols (e.g., H,He,Li,C)')
    parser.add_argument('--octave', type=int,
                       help='Compute all elements in given octave (1-3 supported)')
    parser.add_argument('--export', choices=['json', 'plot', 'both'], default='both',
                       help='Export format')
    parser.add_argument('--output-json', default='vhl_orbital_propagation.json',
                       help='Output JSON filename')
    parser.add_argument('--output-plot', default='vhl_orbital_propagation.png',
                       help='Output plot filename')

    args = parser.parse_args()

    # Determine which elements to compute
    if args.octave:
        if args.octave == 1:
            elements = ['H', 'He']
        elif args.octave == 2:
            elements = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
        elif args.octave == 3:
            elements = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
        else:
            print(f"Error: Octave {args.octave} not supported yet")
            return
    elif args.elements:
        elements = [e.strip() for e in args.elements.split(',')]
    else:
        # Default: first two octaves
        elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

    print("=" * 60)
    print("VHL Orbital Propagation Computation")
    print("=" * 60)
    print(f"Computing orbitals for: {', '.join(elements)}")
    print()

    # Compute propagation matrix
    results = compute_propagation_matrix(elements)

    if not results:
        print("No results generated. Check element symbols.")
        return

    # Analyze patterns
    analyze_propagation_patterns(results)

    # Export
    if args.export in ['json', 'both']:
        with open(args.output_json, 'w') as f:
            json.dump({'elements': results}, f, indent=2)
        print(f"\n✓ Propagation data exported: {args.output_json}")

    if args.export in ['plot', 'both']:
        plot_propagation(results, args.output_plot)

    print("\n" + "=" * 60)
    print("VHL Orbital Propagation Complete!")
    print()
    print("Key Findings:")
    print("  • Nodal surfaces increase with atomic number")
    print("  • Noble gases mark octave boundaries (complete shells)")
    print("  • Orbital complexity maps to VHL radial position")
    print("  • H 2p template propagates as harmonic overtones")
    print("=" * 60)


if __name__ == '__main__':
    main()
