"""
VHL Holographic Resonance Simulation Module

This module implements advanced holographic resonance simulations for the Vibrational Helix Lattice (VHL) framework.
It treats the VHL as a "holographic sphere" where:
- Wave sources are arranged in a helical lattice on a spherical boundary
- Inward-propagating waves create 3D interference patterns
- Resonant standing patterns form complex "folded topology" structures
- Multi-vortex dynamics enable splittable phase singularities

Physics Connections:
- Law 1 (Nodal Uncertainty): Nodal spacing determines natural frequencies
- Law 5 (Holographic Duality): Boundary sources encode bulk information
- AdS/CFT-inspired: Boundary vibrations → emergent bulk geometry
- Cymatic resonance: Standing wave patterns reveal hidden symmetries

Author: Zynerji
Date: 2025-12-15
"""

import numpy as np
from scipy.spatial import distance_matrix
from typing import Tuple, List, Optional, Dict
import warnings

# Suppress runtime warnings for cleaner output
warnings.filterwarnings('ignore')


class SphericalHelixLattice:
    """
    Generates a helical lattice of points on a sphere surface.

    Uses Fibonacci spiral for uniform distribution with helical twist.
    Parameterizable for different helical modes and densities.
    """

    def __init__(self, num_points: int = 500, radius: float = 1.0,
                 helical_turns: int = 5, offset_angle: float = 0.0):
        """
        Initialize spherical helix lattice.

        Args:
            num_points: Number of lattice points on sphere
            radius: Sphere radius
            helical_turns: Number of helical turns around sphere
            offset_angle: Global phase offset (radians)
        """
        self.num_points = num_points
        self.radius = radius
        self.helical_turns = helical_turns
        self.offset_angle = offset_angle
        self.points = None
        self.generate()

    def generate(self) -> np.ndarray:
        """
        Generate lattice points using Fibonacci spiral with helical twist.

        Returns:
            Array of shape (num_points, 3) with (x, y, z) coordinates
        """
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(self.num_points)

        # Fibonacci spiral latitude
        theta = np.arccos(1 - 2 * (indices + 0.5) / self.num_points)

        # Helical longitude
        phi = 2 * np.pi * indices / golden_ratio
        phi += self.helical_turns * theta + self.offset_angle

        # Convert to Cartesian coordinates
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        self.points = np.column_stack([x, y, z])
        return self.points

    def get_points(self) -> np.ndarray:
        """Return lattice points."""
        if self.points is None:
            self.generate()
        return self.points


class WaveSource:
    """
    Represents a wave emitter at a fixed position.

    Emits spherical waves with tunable frequency, amplitude, and phase.
    """

    def __init__(self, position: np.ndarray, frequency: float = 1.0,
                 amplitude: float = 1.0, phase: float = 0.0):
        """
        Initialize wave source.

        Args:
            position: 3D position (x, y, z)
            frequency: Wave frequency (Hz or dimensionless)
            amplitude: Wave amplitude
            phase: Initial phase offset (radians)
        """
        self.position = np.array(position)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.k = 2 * np.pi * frequency  # Wave number (assuming c=1)

    def compute_field(self, positions: np.ndarray, time: float = 0.0,
                      use_complex: bool = True) -> np.ndarray:
        """
        Compute wave field at given positions.

        Uses spherical wave: A * exp(i*(k*r - ω*t + φ)) / r

        Args:
            positions: Array of shape (N, 3) with evaluation points
            time: Current time
            use_complex: If True, return complex field; else return real part

        Returns:
            Array of shape (N,) with field values
        """
        # Distance from source
        distances = np.linalg.norm(positions - self.position, axis=1)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)

        # Phase: k*r - ω*t + φ
        omega = self.k  # Assuming ω = k for c=1
        phases = self.k * distances - omega * time + self.phase

        # Complex field
        field = (self.amplitude / np.sqrt(distances)) * np.exp(1j * phases)

        if use_complex:
            return field
        else:
            return np.real(field)


class VortexMode:
    """
    Represents an optical/acoustic vortex with topological charge.

    Creates phase singularities (dark spots) that can be dynamically positioned.
    """

    def __init__(self, center: np.ndarray, topological_charge: int = 1,
                 amplitude: float = 1.0, global_phase: float = 0.0):
        """
        Initialize vortex mode.

        Args:
            center: 3D center position (x, y, z)
            topological_charge: Integer winding number (±l)
            amplitude: Overall amplitude
            global_phase: Global phase offset
        """
        self.center = np.array(center)
        self.l = topological_charge  # Topological charge
        self.amplitude = amplitude
        self.global_phase = global_phase

    def compute_field(self, positions: np.ndarray, k: float = 1.0) -> np.ndarray:
        """
        Compute vortex field at given positions.

        Field: A * exp(i*(k*r + l*φ + phase)) / sqrt(r)
        where φ is azimuthal angle around vortex center

        Args:
            positions: Array of shape (N, 3) with evaluation points
            k: Wave number

        Returns:
            Complex array of shape (N,)
        """
        # Vector from center to positions
        rel_positions = positions - self.center

        # Radial distance
        r = np.linalg.norm(rel_positions, axis=1)
        r = np.maximum(r, 1e-10)

        # Azimuthal angle (in xy-plane relative to center)
        azimuthal = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])

        # Phase: k*r + l*φ + global_phase
        phases = k * r + self.l * azimuthal + self.global_phase

        # Complex vortex field
        field = (self.amplitude / np.sqrt(r)) * np.exp(1j * phases)

        return field


class HolographicResonator:
    """
    Main holographic resonance simulation engine.

    Combines:
    - Spherical boundary with helical lattice of wave sources
    - Multi-vortex superposition
    - 3D volumetric field computation
    - Dynamic evolution and particle advection
    """

    def __init__(self, sphere_radius: float = 1.0, grid_resolution: int = 64,
                 num_sources: int = 500, helical_turns: int = 5):
        """
        Initialize holographic resonator.

        Args:
            sphere_radius: Radius of boundary sphere
            grid_resolution: Grid resolution for volumetric field (e.g., 64 → 64³ grid)
            num_sources: Number of wave sources on boundary
            helical_turns: Number of helical turns for lattice
        """
        self.sphere_radius = sphere_radius
        self.grid_resolution = grid_resolution
        self.num_sources = num_sources
        self.helical_turns = helical_turns

        # Initialize lattice
        self.lattice = SphericalHelixLattice(
            num_points=num_sources,
            radius=sphere_radius,
            helical_turns=helical_turns
        )

        # Wave sources (initialized with default params)
        self.sources: List[WaveSource] = []

        # Vortex modes
        self.vortices: List[VortexMode] = []

        # 3D grid for field computation
        self.grid_points = None
        self.field = None
        self.intensity = None

        # Initialize grid
        self._create_grid()

    def _create_grid(self):
        """Create 3D volumetric grid inside sphere."""
        # Grid spans [-radius, radius] in each dimension
        n = self.grid_resolution
        coords = np.linspace(-self.sphere_radius, self.sphere_radius, n)

        # Create meshgrid
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

        # Flatten for vectorized computation
        self.grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Mask points outside sphere
        distances = np.linalg.norm(self.grid_points, axis=1)
        self.inside_mask = distances <= self.sphere_radius

    def initialize_sources_from_lattice(self, base_frequency: float = 1.0,
                                       frequency_spread: float = 0.1,
                                       amplitude: float = 1.0,
                                       polarity_phase: bool = True,
                                       seed: Optional[int] = 42):
        """
        Create wave sources at lattice points.

        Args:
            base_frequency: Base frequency for all sources
            frequency_spread: Random variation in frequency (±spread)
            amplitude: Source amplitude
            polarity_phase: If True, assign phase based on polarity pattern
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        lattice_points = self.lattice.get_points()
        self.sources = []

        for i, pos in enumerate(lattice_points):
            # Frequency variation
            freq = base_frequency + np.random.uniform(-frequency_spread, frequency_spread)

            # Phase: polarity-based or random
            if polarity_phase:
                # Simple polarity: alternate based on index
                phase = (i % 2) * np.pi
            else:
                phase = np.random.uniform(0, 2*np.pi)

            source = WaveSource(
                position=pos,
                frequency=freq,
                amplitude=amplitude,
                phase=phase
            )
            self.sources.append(source)

    def add_vortex(self, center: np.ndarray, topological_charge: int = 1,
                   amplitude: float = 1.0, phase: float = 0.0):
        """
        Add a vortex mode to the system.

        Args:
            center: Vortex center position
            topological_charge: Winding number
            amplitude: Amplitude
            phase: Global phase
        """
        vortex = VortexMode(center, topological_charge, amplitude, phase)
        self.vortices.append(vortex)

    def compute_field(self, time: float = 0.0, include_vortices: bool = True) -> np.ndarray:
        """
        Compute total 3D field via superposition of all sources and vortices.

        Args:
            time: Current time
            include_vortices: Whether to include vortex contributions

        Returns:
            Complex field array of shape (grid_resolution³,)
        """
        # Initialize field
        field = np.zeros(len(self.grid_points), dtype=complex)

        # Superpose contributions from all wave sources
        for source in self.sources:
            field += source.compute_field(self.grid_points, time, use_complex=True)

        # Add vortex contributions
        if include_vortices and len(self.vortices) > 0:
            k_avg = 2 * np.pi * 1.0  # Average wave number
            for vortex in self.vortices:
                field += vortex.compute_field(self.grid_points, k=k_avg)

        # Apply mask (zero outside sphere)
        field[~self.inside_mask] = 0

        self.field = field
        return field

    def compute_intensity(self) -> np.ndarray:
        """
        Compute intensity |field|² from current field.

        Returns:
            Real intensity array
        """
        if self.field is None:
            raise ValueError("Field not computed. Run compute_field() first.")

        self.intensity = np.abs(self.field)**2
        return self.intensity

    def get_field_3d(self) -> np.ndarray:
        """Reshape field to 3D grid."""
        n = self.grid_resolution
        return self.field.reshape((n, n, n))

    def get_intensity_3d(self) -> np.ndarray:
        """Reshape intensity to 3D grid."""
        n = self.grid_resolution
        return self.intensity.reshape((n, n, n))

    def extract_isosurfaces(self, levels: List[float]) -> Dict:
        """
        Extract isosurface data for visualization.

        Args:
            levels: List of intensity threshold values

        Returns:
            Dictionary with isosurface information
        """
        from skimage import measure

        intensity_3d = self.get_intensity_3d()
        isosurfaces = {}

        for level in levels:
            try:
                # Marching cubes
                verts, faces, normals, values = measure.marching_cubes(
                    intensity_3d, level=level
                )

                # Convert indices to world coordinates
                n = self.grid_resolution
                scale = 2 * self.sphere_radius / n
                offset = -self.sphere_radius
                verts_world = verts * scale + offset

                isosurfaces[level] = {
                    'vertices': verts_world,
                    'faces': faces,
                    'normals': normals,
                    'values': values
                }
            except:
                # Skip if isosurface doesn't exist at this level
                pass

        return isosurfaces


class ParticleAdvector:
    """
    Particle entrainment and advection in resonant field.

    Particles flow along field gradients or oscillate with local field.
    """

    def __init__(self, num_particles: int = 1000, sphere_radius: float = 1.0,
                 seed: Optional[int] = 42):
        """
        Initialize particle system.

        Args:
            num_particles: Number of particles
            sphere_radius: Confine particles within sphere
            seed: Random seed
        """
        self.num_particles = num_particles
        self.sphere_radius = sphere_radius

        # Initialize positions randomly inside sphere
        if seed is not None:
            np.random.seed(seed)

        # Uniform random in sphere
        self.positions = self._random_sphere_points(num_particles, sphere_radius * 0.8)
        self.velocities = np.zeros_like(self.positions)

    def _random_sphere_points(self, n: int, radius: float) -> np.ndarray:
        """Generate n random points uniformly distributed in sphere."""
        # Method: rejection sampling
        points = []
        while len(points) < n:
            candidate = np.random.uniform(-radius, radius, size=3)
            if np.linalg.norm(candidate) <= radius:
                points.append(candidate)
        return np.array(points)

    def advect(self, resonator: HolographicResonator, dt: float = 0.01,
               mode: str = 'gradient', damping: float = 0.1):
        """
        Update particle positions based on field.

        Args:
            resonator: Holographic resonator with computed field
            dt: Time step
            mode: 'gradient' or 'oscillation'
            damping: Velocity damping factor
        """
        if mode == 'gradient':
            # Advect along gradient of intensity
            self._advect_gradient(resonator, dt, damping)
        elif mode == 'oscillation':
            # Oscillate with local field phase
            self._advect_oscillation(resonator, dt, damping)

    def _advect_gradient(self, resonator: HolographicResonator, dt: float, damping: float):
        """Advect particles along intensity gradient."""
        # Compute gradient at particle positions via finite differences
        h = 0.01  # Small offset

        intensity_at_pos = self._interpolate_intensity(resonator, self.positions)

        # Gradient estimation
        gradient = np.zeros_like(self.positions)
        for dim in range(3):
            offset = np.zeros(3)
            offset[dim] = h

            pos_plus = self.positions + offset
            pos_minus = self.positions - offset

            intensity_plus = self._interpolate_intensity(resonator, pos_plus)
            intensity_minus = self._interpolate_intensity(resonator, pos_minus)

            gradient[:, dim] = (intensity_plus - intensity_minus) / (2 * h)

        # Update velocities
        self.velocities += gradient * dt
        self.velocities *= (1 - damping)

        # Update positions
        self.positions += self.velocities * dt

        # Confine to sphere
        self._confine_to_sphere()

    def _advect_oscillation(self, resonator: HolographicResonator, dt: float, damping: float):
        """Oscillate particles with local field phase."""
        # Get complex field at particle positions
        field_at_pos = self._interpolate_field(resonator, self.positions)

        # Velocity proportional to field gradient direction
        # Simplified: use real and imaginary parts
        self.velocities[:, 0] += np.real(field_at_pos) * dt
        self.velocities[:, 1] += np.imag(field_at_pos) * dt
        self.velocities *= (1 - damping)

        # Update positions
        self.positions += self.velocities * dt
        self._confine_to_sphere()

    def _interpolate_intensity(self, resonator: HolographicResonator, positions: np.ndarray) -> np.ndarray:
        """Interpolate intensity at arbitrary positions."""
        from scipy.interpolate import RegularGridInterpolator

        n = resonator.grid_resolution
        coords = np.linspace(-resonator.sphere_radius, resonator.sphere_radius, n)
        intensity_3d = resonator.get_intensity_3d()

        interpolator = RegularGridInterpolator(
            (coords, coords, coords),
            intensity_3d,
            bounds_error=False,
            fill_value=0
        )

        return interpolator(positions)

    def _interpolate_field(self, resonator: HolographicResonator, positions: np.ndarray) -> np.ndarray:
        """Interpolate complex field at arbitrary positions."""
        from scipy.interpolate import RegularGridInterpolator

        n = resonator.grid_resolution
        coords = np.linspace(-resonator.sphere_radius, resonator.sphere_radius, n)
        field_3d = resonator.get_field_3d()

        # Interpolate real and imaginary parts separately
        real_interp = RegularGridInterpolator(
            (coords, coords, coords),
            np.real(field_3d),
            bounds_error=False,
            fill_value=0
        )
        imag_interp = RegularGridInterpolator(
            (coords, coords, coords),
            np.imag(field_3d),
            bounds_error=False,
            fill_value=0
        )

        return real_interp(positions) + 1j * imag_interp(positions)

    def _confine_to_sphere(self):
        """Reflect particles that escape sphere."""
        distances = np.linalg.norm(self.positions, axis=1)
        escaped = distances > self.sphere_radius

        if np.any(escaped):
            # Reflect back
            self.positions[escaped] *= (self.sphere_radius * 0.99) / distances[escaped, None]
            # Reverse normal component of velocity
            for i in np.where(escaped)[0]:
                normal = self.positions[i] / np.linalg.norm(self.positions[i])
                v_normal = np.dot(self.velocities[i], normal)
                self.velocities[i] -= 2 * v_normal * normal


# Example usage
if __name__ == "__main__":
    print("VHL Holographic Resonance Module")
    print("=" * 50)

    # Create resonator
    resonator = HolographicResonator(
        sphere_radius=1.0,
        grid_resolution=64,
        num_sources=500,
        helical_turns=5
    )

    # Initialize sources
    resonator.initialize_sources_from_lattice(
        base_frequency=1.0,
        frequency_spread=0.1,
        polarity_phase=True
    )

    # Add a vortex
    resonator.add_vortex(center=np.array([0, 0, 0]), topological_charge=1)

    # Compute field
    print("Computing holographic field...")
    field = resonator.compute_field(time=0.0)
    intensity = resonator.compute_intensity()

    print(f"Field shape: {field.shape}")
    print(f"Max intensity: {intensity.max():.4f}")
    print(f"Mean intensity: {intensity.mean():.4f}")

    # Extract isosurfaces
    levels = [0.1, 0.5, 1.0]
    isosurfaces = resonator.extract_isosurfaces(levels)
    print(f"\nExtracted {len(isosurfaces)} isosurfaces")

    # Initialize particles
    particles = ParticleAdvector(num_particles=1000, sphere_radius=0.8)
    print(f"\nInitialized {particles.num_particles} particles")

    print("\n✓ Core module ready for visualization!")
