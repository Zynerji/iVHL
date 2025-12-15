"""
VHL Holographic Resonance - Framework Integration

Bridges the holographic resonance simulation with existing VHL framework:
- Maps element properties to wave source parameters
- Uses VHL helical lattice generation
- Connects to VHL unification (Laws 1, 5) for holographic compression
- Integrates nodal spacing, polarity, and orbital structure
- Provides unified API for element-aware holographic simulation

This module makes holographic resonance "VHL-aware" by translating
quantum properties (shell structure, nodal surfaces, polarity) into
resonance parameters (frequencies, phases, vortex charges).

Usage:
    from vhl_integration import VHLElementMapper

    mapper = VHLElementMapper()
    resonator = mapper.create_element_resonator('F', num_sources=500)
    mapper.add_element_vortices(resonator, 'F', num_vortices=2)
    resonator.compute_field()

Author: Zynerji
Date: 2025-12-15
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json

# Import existing VHL modules
try:
    from vhl_orbital_propagation import ELEMENT_CONFIGS, count_nodal_surfaces
    from vhl_unification_v2 import VHLUnificationV2
except ImportError:
    print("Warning: Could not import VHL orbital/unification modules")
    ELEMENT_CONFIGS = {}

# Import holographic resonance modules
from vhl_holographic_resonance import (
    HolographicResonator,
    SphericalHelixLattice
)
from vhl_vortex_controller import (
    FourierTrajectory,
    MultiVortexChoreographer
)

# Extended element data with VHL properties
VHL_ELEMENT_DATA = {
    'H': {
        'z': 1, 'octave': 1, 'tone': 1, 'polarity': 1,
        'shell_structure': [1],
        'nodal_surfaces': 0,
        'base_frequency': 1.0,  # Normalized
        'orbital_radius': 0.53,  # Å
    },
    'He': {
        'z': 2, 'octave': 1, 'tone': 2, 'polarity': 0,
        'shell_structure': [2],
        'nodal_surfaces': 0,
        'base_frequency': 1.414,  # sqrt(2)
        'orbital_radius': 0.31,
    },
    'Li': {
        'z': 3, 'octave': 2, 'tone': 1, 'polarity': 1,
        'shell_structure': [2, 1],
        'nodal_surfaces': 1,
        'base_frequency': 1.732,  # sqrt(3)
        'orbital_radius': 1.67,
    },
    'Be': {
        'z': 4, 'octave': 2, 'tone': 2, 'polarity': 1,
        'shell_structure': [2, 2],
        'nodal_surfaces': 1,
        'base_frequency': 2.0,
        'orbital_radius': 1.12,
    },
    'C': {
        'z': 6, 'octave': 2, 'tone': 4, 'polarity': 0,
        'shell_structure': [2, 4],
        'nodal_surfaces': 1,
        'base_frequency': 2.449,  # sqrt(6)
        'orbital_radius': 0.77,
    },
    'N': {
        'z': 7, 'octave': 2, 'tone': 5, 'polarity': -1,
        'shell_structure': [2, 5],
        'nodal_surfaces': 1,
        'base_frequency': 2.646,  # sqrt(7)
        'orbital_radius': 0.75,
    },
    'O': {
        'z': 8, 'octave': 2, 'tone': 6, 'polarity': -1,
        'shell_structure': [2, 6],
        'nodal_surfaces': 1,
        'base_frequency': 2.828,  # sqrt(8)
        'orbital_radius': 0.73,
    },
    'F': {
        'z': 9, 'octave': 2, 'tone': 7, 'polarity': -1,
        'shell_structure': [2, 7],
        'nodal_surfaces': 1,
        'base_frequency': 3.0,  # sqrt(9)
        'orbital_radius': 0.71,
    },
    'Ne': {
        'z': 10, 'octave': 2, 'tone': 8, 'polarity': 0,
        'shell_structure': [2, 8],
        'nodal_surfaces': 1,
        'base_frequency': 3.162,  # sqrt(10)
        'orbital_radius': 0.69,
    },
    'Na': {
        'z': 11, 'octave': 3, 'tone': 1, 'polarity': 1,
        'shell_structure': [2, 8, 1],
        'nodal_surfaces': 3,
        'base_frequency': 3.317,  # sqrt(11)
        'orbital_radius': 1.90,
    },
    'Mg': {
        'z': 12, 'octave': 3, 'tone': 2, 'polarity': 1,
        'shell_structure': [2, 8, 2],
        'nodal_surfaces': 3,
        'base_frequency': 3.464,  # sqrt(12)
        'orbital_radius': 1.60,
    },
    'Fe': {
        'z': 26, 'octave': 4, 'tone': 8, 'polarity': 0,
        'shell_structure': [2, 8, 14, 2],
        'nodal_surfaces': 6,
        'base_frequency': 5.099,  # sqrt(26)
        'orbital_radius': 1.26,
    },
    'Cu': {
        'z': 29, 'octave': 4, 'tone': 11, 'polarity': -1,
        'shell_structure': [2, 8, 18, 1],
        'nodal_surfaces': 6,
        'base_frequency': 5.385,  # sqrt(29)
        'orbital_radius': 1.28,
    },
    'Au': {
        'z': 79, 'octave': 6, 'tone': 11, 'polarity': 1,
        'shell_structure': [2, 8, 18, 32, 18, 1],
        'nodal_surfaces': 15,
        'base_frequency': 8.888,  # sqrt(79)
        'orbital_radius': 1.44,
    },
}


class VHLElementMapper:
    """
    Maps element properties from VHL framework to holographic resonance parameters.

    Key Mappings:
    - Atomic number → Base frequency (via sqrt(Z))
    - Nodal surfaces → Helical turns
    - Polarity → Phase offsets and vortex charges
    - Shell structure → Frequency spread pattern
    - Orbital radius → Sphere radius scaling
    """

    def __init__(self):
        """Initialize mapper with VHL unification framework."""
        self.unification = None
        try:
            self.unification = VHLUnificationV2()
        except:
            print("VHL Unification V2 not available, using simplified mapping")

    def get_element_data(self, symbol: str) -> Dict:
        """
        Get comprehensive element data.

        Args:
            symbol: Element symbol (e.g., 'H', 'He', 'F')

        Returns:
            Dictionary with all element properties
        """
        if symbol not in VHL_ELEMENT_DATA:
            raise ValueError(f"Element {symbol} not in database")

        return VHL_ELEMENT_DATA[symbol].copy()

    def compute_resonance_frequency(self, symbol: str,
                                   use_nodal_correction: bool = True) -> float:
        """
        Compute base resonance frequency for element.

        Rationale:
        - Base: ω = sqrt(Z) (follows hydrogen-like scaling)
        - Correction: Multiply by (n_nodes + 1) for shell-dependent shifts
        - VHL Law 5: Frequency encodes nodal spacing information

        Args:
            symbol: Element symbol
            use_nodal_correction: Apply nodal surface correction

        Returns:
            Base angular frequency
        """
        data = self.get_element_data(symbol)

        # Base frequency from atomic number
        omega_base = np.sqrt(data['z'])

        # Nodal correction (higher shells → higher harmonics)
        if use_nodal_correction:
            nodal_factor = 1.0 + 0.1 * data['nodal_surfaces']
            omega_base *= nodal_factor

        return omega_base

    def compute_frequency_spread(self, symbol: str) -> float:
        """
        Compute frequency spread for source ensemble.

        Shell structure creates natural frequency distribution:
        - More shells → broader spread
        - Represents orbital energy level spacing

        Args:
            symbol: Element symbol

        Returns:
            Frequency spread parameter
        """
        data = self.get_element_data(symbol)

        # Spread increases with shell complexity
        n_shells = len(data['shell_structure'])
        spread = 0.05 * np.sqrt(n_shells)

        return spread

    def compute_sphere_radius(self, symbol: str,
                             scale_by_atomic_radius: bool = True) -> float:
        """
        Compute holographic sphere radius.

        Options:
        1. Scale by atomic radius (physical correspondence)
        2. Fixed unit sphere (normalized visualization)

        Args:
            symbol: Element symbol
            scale_by_atomic_radius: Use atomic radius for scaling

        Returns:
            Sphere radius
        """
        data = self.get_element_data(symbol)

        if scale_by_atomic_radius:
            # Scale to angstroms (normalized to hydrogen)
            return data['orbital_radius'] / 0.53
        else:
            return 1.0

    def compute_helical_turns(self, symbol: str) -> int:
        """
        Compute helical turns from nodal structure.

        Rationale:
        - Each nodal surface represents a phase wind
        - Helical turns = f(nodal_surfaces)
        - Minimum 3 turns for stability

        Args:
            symbol: Element symbol

        Returns:
            Number of helical turns
        """
        data = self.get_element_data(symbol)

        # Map nodal surfaces to helical turns
        # More nodes → more spiral winds
        turns = 3 + 2 * data['nodal_surfaces']

        return turns

    def get_polarity_phase_offset(self, symbol: str) -> float:
        """
        Convert VHL polarity to phase offset.

        Mapping:
        - Polarity +1 (expansion) → phase = 0
        - Polarity -1 (contraction) → phase = π
        - Polarity 0 (neutral) → phase = π/2

        Args:
            symbol: Element symbol

        Returns:
            Phase offset in radians
        """
        data = self.get_element_data(symbol)
        polarity = data['polarity']

        if polarity > 0:
            return 0.0
        elif polarity < 0:
            return np.pi
        else:
            return np.pi / 2

    def create_element_resonator(self, symbol: str,
                                num_sources: int = 500,
                                grid_resolution: int = 64) -> HolographicResonator:
        """
        Create holographic resonator configured for specific element.

        All parameters derived from element properties via VHL framework.

        Args:
            symbol: Element symbol
            num_sources: Number of wave sources
            grid_resolution: Field grid resolution

        Returns:
            Configured HolographicResonator instance
        """
        data = self.get_element_data(symbol)

        # Compute VHL-derived parameters
        sphere_radius = self.compute_sphere_radius(symbol)
        helical_turns = self.compute_helical_turns(symbol)
        base_frequency = self.compute_resonance_frequency(symbol)
        freq_spread = self.compute_frequency_spread(symbol)

        print(f"Creating resonator for {symbol} (Z={data['z']}):")
        print(f"  Sphere radius: {sphere_radius:.3f}")
        print(f"  Helical turns: {helical_turns}")
        print(f"  Base frequency: {base_frequency:.3f}")
        print(f"  Freq spread: {freq_spread:.3f}")

        # Create resonator
        resonator = HolographicResonator(
            sphere_radius=sphere_radius,
            grid_resolution=grid_resolution,
            num_sources=num_sources,
            helical_turns=helical_turns
        )

        # Initialize sources with element-specific parameters
        polarity_phase = (data['polarity'] != 0)  # Use polarity modulation if non-zero

        resonator.initialize_sources_from_lattice(
            base_frequency=base_frequency,
            frequency_spread=freq_spread,
            polarity_phase=polarity_phase
        )

        return resonator

    def add_element_vortices(self, resonator: HolographicResonator,
                           symbol: str, num_vortices: int = 1,
                           trajectory_type: str = 'circle') -> MultiVortexChoreographer:
        """
        Add vortices representing element's topological features.

        Vortex configuration based on:
        - Polarity → vortex charge sign
        - Shell structure → number of vortices
        - Nodal surfaces → trajectory complexity

        Args:
            resonator: HolographicResonator instance
            symbol: Element symbol
            num_vortices: Number of vortices (overrides auto-determination)
            trajectory_type: Trajectory preset ('circle', 'figure8', etc.)

        Returns:
            MultiVortexChoreographer managing vortices
        """
        data = self.get_element_data(symbol)

        # Auto-determine vortex count if not specified
        if num_vortices is None:
            # One vortex per valence electron (simplified)
            valence = data['shell_structure'][-1]
            num_vortices = min(valence, 4)  # Cap at 4 for visualization

        # Create choreographer
        choreographer = MultiVortexChoreographer(num_vortices=num_vortices)

        # Vortex charge from polarity
        base_charge = data['polarity'] if data['polarity'] != 0 else 1

        for i in range(num_vortices):
            # Distribute around sphere
            angle = 2 * np.pi * i / num_vortices
            radius = 0.3 * resonator.sphere_radius

            center = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ])

            # Alternating charges
            charge = base_charge * ((-1) ** i)

            # Phase offset for coordination
            phase_offset = self.get_polarity_phase_offset(symbol) + angle

            # Add vortex to choreographer
            choreographer.add_fourier_vortex(
                preset=trajectory_type,
                omega=1.0,
                amplitude=0.3 * resonator.sphere_radius,
                phase_offset=phase_offset
            )

            # Add vortex mode to resonator
            resonator.add_vortex(center=center, topological_charge=charge)

        print(f"Added {num_vortices} vortices with base charge {base_charge}")

        return choreographer

    def compute_holographic_compression(self, symbol: str) -> Dict:
        """
        Apply VHL Law 5 holographic compression to element.

        Uses VHL unification framework to compress 3D orbital structure
        to 2D boundary information (AdS/CFT inspired).

        Args:
            symbol: Element symbol

        Returns:
            Dictionary with compression analysis
        """
        if self.unification is None:
            return {
                'error': 'VHL Unification framework not available',
                'symbol': symbol
            }

        data = self.get_element_data(symbol)

        # Compute nodal spacing
        nodal_spacing, n_nodes, r_extent = self.unification.get_nodal_spacing(data['z'])

        # Holographic entropy (boundary → bulk encoding)
        # Surface ~ 4πR², volume ~ 4πR³/3
        # Holographic: S_boundary = S_bulk / R

        surface_area = 4 * np.pi * r_extent**2
        volume = (4/3) * np.pi * r_extent**3

        compression_ratio = volume / surface_area  # ~ R

        return {
            'symbol': symbol,
            'z': data['z'],
            'nodal_spacing': nodal_spacing,
            'n_nodes': n_nodes,
            'radial_extent': r_extent,
            'surface_area': surface_area,
            'volume': volume,
            'compression_ratio': compression_ratio,
            'holographic_entropy': surface_area / compression_ratio
        }

    def export_element_config(self, symbol: str, filepath: str):
        """
        Export complete element configuration to JSON.

        Args:
            symbol: Element symbol
            filepath: Output JSON file path
        """
        data = self.get_element_data(symbol)

        config = {
            'element': symbol,
            'atomic_number': data['z'],
            'octave': data['octave'],
            'tone': data['tone'],
            'polarity': data['polarity'],
            'shell_structure': data['shell_structure'],
            'nodal_surfaces': data['nodal_surfaces'],
            'resonance_parameters': {
                'base_frequency': self.compute_resonance_frequency(symbol),
                'frequency_spread': self.compute_frequency_spread(symbol),
                'sphere_radius': self.compute_sphere_radius(symbol),
                'helical_turns': self.compute_helical_turns(symbol),
                'phase_offset': self.get_polarity_phase_offset(symbol)
            },
            'holographic_compression': self.compute_holographic_compression(symbol)
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[OK] Exported {symbol} configuration to {filepath}")


# Convenience function for quick element simulation
def simulate_element(symbol: str, num_sources: int = 500,
                     num_vortices: int = 2, grid_resolution: int = 64,
                     compute_field: bool = True) -> Tuple[HolographicResonator,
                                                          MultiVortexChoreographer]:
    """
    One-line function to create and configure element simulation.

    Args:
        symbol: Element symbol
        num_sources: Number of wave sources
        num_vortices: Number of vortices
        grid_resolution: Field grid resolution
        compute_field: Compute field immediately

    Returns:
        (resonator, choreographer) tuple
    """
    mapper = VHLElementMapper()

    # Create resonator
    resonator = mapper.create_element_resonator(
        symbol, num_sources, grid_resolution
    )

    # Add vortices
    choreographer = mapper.add_element_vortices(
        resonator, symbol, num_vortices
    )

    # Compute field
    if compute_field:
        print("Computing holographic field...")
        resonator.compute_field(time=0.0)
        resonator.compute_intensity()
        print(f"[OK] Field computed: max intensity = {resonator.intensity.max():.4f}")

    return resonator, choreographer


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("VHL HOLOGRAPHIC RESONANCE - FRAMEWORK INTEGRATION")
    print("=" * 70)

    # Test element mapping
    mapper = VHLElementMapper()

    print("\n1. Testing Element Mappings:")
    test_elements = ['H', 'He', 'C', 'O', 'F', 'Ne', 'Fe', 'Au']

    for symbol in test_elements:
        data = mapper.get_element_data(symbol)
        freq = mapper.compute_resonance_frequency(symbol)
        turns = mapper.compute_helical_turns(symbol)

        print(f"\n  {symbol:3s} (Z={data['z']:3d}): "
              f"octave={data['octave']}, nodes={data['nodal_surfaces']}, "
              f"pol={data['polarity']:+2d}")
        print(f"       => w={freq:.3f}, turns={turns}, "
              f"radius={data['orbital_radius']:.2f}A")

    # Test holographic compression
    print("\n2. Testing Holographic Compression (VHL Law 5):")
    for symbol in ['H', 'F', 'Fe']:
        compression = mapper.compute_holographic_compression(symbol)
        if 'error' not in compression:
            print(f"\n  {symbol}: "
                  f"nodes={compression['n_nodes']}, "
                  f"spacing={compression['nodal_spacing']*1e10:.2f}Å")
            print(f"       compression_ratio={compression['compression_ratio']*1e10:.3f}Å")

    # Test full element simulation
    print("\n3. Testing Full Element Simulation (Fluorine):")
    resonator, choreographer = simulate_element('F', num_sources=200,
                                                num_vortices=2,
                                                grid_resolution=48)

    print("\n[OK] VHL integration ready!")
