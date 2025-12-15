"""
Boundary Resonance Simulator
=============================

Simulates acoustic wave interference patterns on a spherical boundary,
using helical lattice point sources inspired by the iVHL geometry.

DISCLAIMER: This is a mathematical exploration of wave interference patterns,
not a claim about physical reality.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class BoundaryConfig:
    """Configuration for boundary resonance simulation."""

    # Lattice parameters
    num_nodes: int = 126
    sphere_radius: float = 10.0  # Angstroms
    helical_turns: int = 7
    radial_modulation_amplitude: float = 0.15

    # Wave parameters
    base_frequency: float = 1.0  # Hz (arbitrary units)
    wave_speed: float = 1.0  # Speed of sound (arbitrary units)
    phase_randomness: float = 0.1  # Random phase offset

    # Grid parameters
    grid_resolution: int = 64  # θ × φ grid
    timesteps: int = 100
    dt: float = 0.01  # Time step

    # Computation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class BoundaryResonanceSimulator:
    """
    Simulates wave interference on a spherical boundary with helical sources.

    The boundary field ψ(θ, φ, t) is computed as:
        ψ = Σᵢ (Aᵢ / rᵢ) * sin(k*rᵢ - ω*t + φᵢ)

    where:
    - θ, φ: Spherical coordinates on boundary
    - rᵢ: Distance from point i to evaluation point
    - Aᵢ: Amplitude (can be modulated by polarity)
    - k = 2π/λ: Wavenumber
    - ω = 2πf: Angular frequency
    - φᵢ: Phase offset
    """

    def __init__(self, config: BoundaryConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Generate helical lattice source points
        self.source_positions = self._generate_helical_lattice()

        # Initialize wave parameters
        self.frequencies = self._initialize_frequencies()
        self.amplitudes = self._initialize_amplitudes()
        self.phases = self._initialize_phases()

        # Create evaluation grid on sphere
        self.grid_theta, self.grid_phi = self._create_spherical_grid()
        self.grid_cartesian = self._spherical_to_cartesian(
            self.grid_theta,
            self.grid_phi,
            self.config.sphere_radius
        )

        # Precompute distances from sources to grid points
        self.distances = self._compute_distances()

        print(f"BoundaryResonanceSimulator initialized:")
        print(f"  - {self.config.num_nodes} helical sources")
        print(f"  - {self.config.grid_resolution}² grid points")
        print(f"  - {self.config.timesteps} timesteps")
        print(f"  - Device: {self.device}")

    def _generate_helical_lattice(self) -> torch.Tensor:
        """
        Generate helical lattice points on sphere surface.

        Returns:
            Tensor of shape (num_nodes, 3) with Cartesian coordinates
        """
        N = self.config.num_nodes
        R = self.config.sphere_radius
        turns = self.config.helical_turns

        # Parameter along helix
        s = torch.linspace(0, 1, N, device=self.device, dtype=self.config.dtype)

        # Spherical coordinates with helical pattern
        theta = torch.pi * s  # Polar angle: 0 to π
        phi = 2 * torch.pi * turns * s  # Azimuthal: multiple wraps

        # Apply hyperbolic radial modulation
        r_mod = R * (1.0 + self.config.radial_modulation_amplitude *
                     torch.sinh(4 * torch.pi * s))

        # Convert to Cartesian
        x = r_mod * torch.sin(theta) * torch.cos(phi)
        y = r_mod * torch.sin(theta) * torch.sin(phi)
        z = r_mod * torch.cos(theta)

        return torch.stack([x, y, z], dim=1)

    def _initialize_frequencies(self) -> torch.Tensor:
        """
        Initialize frequencies for each source.

        Uses base frequency with small variations based on position.
        """
        N = self.config.num_nodes
        f_base = self.config.base_frequency

        # Add small harmonic variations
        harmonics = 1.0 + 0.1 * torch.sin(
            2 * torch.pi * torch.arange(N, device=self.device) / N
        )

        return f_base * harmonics

    def _initialize_amplitudes(self) -> torch.Tensor:
        """
        Initialize amplitudes for each source.

        Could be modulated by polarity in future versions.
        """
        N = self.config.num_nodes
        return torch.ones(N, device=self.device, dtype=self.config.dtype)

    def _initialize_phases(self) -> torch.Tensor:
        """Initialize random phase offsets."""
        N = self.config.num_nodes
        return 2 * torch.pi * self.config.phase_randomness * torch.rand(
            N, device=self.device, dtype=self.config.dtype
        )

    def _create_spherical_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create θ × φ grid on sphere.

        Returns:
            (theta_grid, phi_grid): Both of shape (grid_resolution, grid_resolution)
        """
        res = self.config.grid_resolution

        theta = torch.linspace(0, torch.pi, res, device=self.device, dtype=self.config.dtype)
        phi = torch.linspace(0, 2 * torch.pi, res, device=self.device, dtype=self.config.dtype)

        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

        return theta_grid, phi_grid

    def _spherical_to_cartesian(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        r: float
    ) -> torch.Tensor:
        """
        Convert spherical to Cartesian coordinates.

        Returns:
            Tensor of shape (*theta.shape, 3)
        """
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        return torch.stack([x, y, z], dim=-1)

    def _compute_distances(self) -> torch.Tensor:
        """
        Compute distances from each source to each grid point.

        Returns:
            Tensor of shape (num_nodes, grid_resolution, grid_resolution)
        """
        # grid_cartesian: (res, res, 3)
        # source_positions: (N, 3)
        # Need to broadcast and compute distances

        grid_flat = self.grid_cartesian.reshape(-1, 3)  # (res², 3)
        sources = self.source_positions  # (N, 3)

        # Compute pairwise distances
        # distances[i, j] = ||source[i] - grid[j]||
        diff = sources.unsqueeze(1) - grid_flat.unsqueeze(0)  # (N, res², 3)
        distances = torch.norm(diff, dim=2)  # (N, res²)

        # Reshape back to grid
        res = self.config.grid_resolution
        distances = distances.reshape(self.config.num_nodes, res, res)

        # Avoid division by zero
        distances = torch.clamp(distances, min=1e-6)

        return distances

    def compute_field(self, t: float) -> torch.Tensor:
        """
        Compute boundary field ψ(θ, φ, t) via wave superposition.

        ψ = Σᵢ (Aᵢ / rᵢ) * sin(k*rᵢ - ω*t + φᵢ)

        Args:
            t: Time

        Returns:
            Field tensor of shape (grid_resolution, grid_resolution)
        """
        # Compute wavenumbers: k = 2πf / c
        k = 2 * torch.pi * self.frequencies / self.config.wave_speed

        # Compute angular frequencies: ω = 2πf
        omega = 2 * torch.pi * self.frequencies

        # Compute current phases: k*r - ω*t + φ₀
        current_phases = -omega * t + self.phases  # (N,)

        # Compute contributions from each source
        # Shape: (N, res, res)
        k_expanded = k.view(-1, 1, 1)
        contributions = (self.amplitudes.view(-1, 1, 1) / self.distances) * torch.sin(
            k_expanded * self.distances + current_phases.view(-1, 1, 1)
        )

        # Sum over all sources
        field = torch.sum(contributions, dim=0)  # (res, res)

        return field

    def run_simulation(self) -> Dict:
        """
        Run full time evolution simulation.

        Returns:
            Dictionary containing:
                - 'field_evolution': (timesteps, res, res) tensor
                - 'times': (timesteps,) tensor
                - 'spatial_fft': FFT of final field
                - 'temporal_fft': FFT of time series at sample points
        """
        print("Running boundary resonance simulation...")

        times = torch.arange(
            self.config.timesteps,
            device=self.device,
            dtype=self.config.dtype
        ) * self.config.dt

        field_evolution = torch.zeros(
            self.config.timesteps,
            self.config.grid_resolution,
            self.config.grid_resolution,
            device=self.device,
            dtype=self.config.dtype
        )

        for i, t in enumerate(times):
            field_evolution[i] = self.compute_field(t.item())

            if (i + 1) % 20 == 0:
                print(f"  Step {i+1}/{self.config.timesteps}")

        # Compute spatial FFT of final field
        final_field = field_evolution[-1]
        spatial_fft = torch.fft.fft2(final_field)
        spatial_power = torch.abs(spatial_fft) ** 2

        # Compute temporal FFT at center point
        center_idx = self.config.grid_resolution // 2
        time_series = field_evolution[:, center_idx, center_idx]
        temporal_fft = torch.fft.fft(time_series)
        temporal_power = torch.abs(temporal_fft) ** 2
        temporal_freqs = torch.fft.fftfreq(
            self.config.timesteps,
            d=self.config.dt
        ).to(self.device)

        results = {
            'field_evolution': field_evolution.cpu().numpy(),
            'times': times.cpu().numpy(),
            'spatial_power_spectrum': spatial_power.cpu().numpy(),
            'temporal_power_spectrum': temporal_power.cpu().numpy(),
            'temporal_frequencies': temporal_freqs.cpu().numpy(),
            'source_positions': self.source_positions.cpu().numpy(),
            'grid_theta': self.grid_theta.cpu().numpy(),
            'grid_phi': self.grid_phi.cpu().numpy(),
        }

        print("Simulation complete!")
        return results

    def compute_entropy(self, field: torch.Tensor) -> float:
        """
        Compute information entropy of field configuration.

        S = -Σ p * log(p), where p is normalized |ψ|²

        Args:
            field: Field tensor

        Returns:
            Entropy value (scalar)
        """
        # Normalize field to probability distribution
        prob = torch.abs(field) ** 2
        prob = prob / torch.sum(prob)

        # Avoid log(0)
        prob = torch.clamp(prob, min=1e-12)

        entropy = -torch.sum(prob * torch.log(prob))

        return entropy.item()

    def save_results(self, results: Dict, filepath: str):
        """Save simulation results to file."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            key: val.tolist() if isinstance(val, np.ndarray) else val
            for key, val in results.items()
        }

        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    config = BoundaryConfig(
        num_nodes=126,
        grid_resolution=64,
        timesteps=100,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    simulator = BoundaryResonanceSimulator(config)
    results = simulator.run_simulation()

    print(f"\nFinal field entropy: {simulator.compute_entropy(torch.tensor(results['field_evolution'][-1])):.4f}")
    print(f"Peak temporal frequency: {results['temporal_frequencies'][np.argmax(results['temporal_power_spectrum'])]:.4f} Hz")
