#!/usr/bin/env python3
"""
Helper script to create all multiscale framework module files.
This avoids heredoc quoting issues.
"""

import os

# Define all file contents as strings
GFT_FIELD_CONTENT = '''"""
Group Field Theory (GFT) Field Evolver
=======================================

Simulates evolution of a complex-valued tensor field Φ(x, t) with colored
indices, using Gross-Pitaevskii-like dynamics:

    iℏ ∂Φ/∂t = [-ℏ²/(2m) ∇² + V(x) + λ|Φ|²] Φ

Evolved via split-step Fourier method for efficiency.

DISCLAIMER: This is a mathematical model exploration, not a claim about
quantum gravity or physical group field theory.
"""

import numpy as np
import torch
import torch.fft
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import json


@dataclass
class GFTConfig:
    """Configuration for GFT field evolution."""

    # Spatial grid
    grid_size: int = 32  # Nx = Ny = Nz
    box_size: float = 20.0  # Angstroms
    num_colors: int = 4  # Tensor color dimension

    # Physical parameters (arbitrary units)
    hbar: float = 1.0
    mass: float = 1.0
    interaction_strength: float = 0.1  # λ for |Φ|² term
    potential_strength: float = 0.5  # Harmonic trap strength

    # Time evolution
    timesteps: int = 100
    dt: float = 0.01

    # Initial conditions
    initial_type: str = "gaussian_wavepacket"  # or "random", "plane_wave"
    initial_amplitude: float = 1.0
    initial_noise_level: float = 0.1

    # Computation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.complex64


class GFTFieldEvolver:
    """
    Evolves GFT field Φ(x, t, color) using split-step Fourier method.

    Split-step algorithm:
        1. Evolve kinetic term in k-space: FFT → exp(-iℏk²Δt/(2m)) → IFFT
        2. Evolve potential term in real space: exp(-i[V + λ|Φ|²]Δt/ℏ)
    """

    def __init__(self, config: GFTConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create spatial grid
        self.x, self.y, self.z = self._create_spatial_grid()

        # Create momentum grid
        self.kx, self.ky, self.kz = self._create_momentum_grid()

        # Precompute kinetic operator in k-space
        self.kinetic_k = self._compute_kinetic_operator()

        # Compute potential
        self.potential = self._compute_potential()

        # Initialize field
        self.field = self._initialize_field()

        print(f"GFTFieldEvolver initialized:")
        print(f"  - Grid: {config.grid_size}³ × {config.num_colors} colors")
        print(f"  - Box size: {config.box_size} Å")
        print(f"  - Timesteps: {config.timesteps}")
        print(f"  - Device: {self.device}")

    def _create_spatial_grid(self):
        """Create 3D spatial grid."""
        N = self.config.grid_size
        L = self.config.box_size

        x_1d = torch.linspace(-L/2, L/2, N, device=self.device, dtype=torch.float32)

        x, y, z = torch.meshgrid(x_1d, x_1d, x_1d, indexing='ij')

        return x, y, z

    def _create_momentum_grid(self):
        """Create 3D momentum grid for FFT."""
        N = self.config.grid_size
        L = self.config.box_size

        # Frequency grid
        k_1d = 2 * torch.pi * torch.fft.fftfreq(N, d=L/N, device=self.device)

        kx, ky, kz = torch.meshgrid(k_1d, k_1d, k_1d, indexing='ij')

        return kx, ky, kz

    def _compute_kinetic_operator(self):
        """Compute exp(-iℏk²Δt/(2m)) for kinetic evolution."""
        k_squared = self.kx**2 + self.ky**2 + self.kz**2

        exponent = -1j * self.config.hbar * k_squared * self.config.dt / (2 * self.config.mass)

        return torch.exp(exponent).to(self.config.dtype)

    def _compute_potential(self):
        """Compute harmonic potential V(x) = 0.5 * k * r²."""
        r_squared = self.x**2 + self.y**2 + self.z**2

        V = 0.5 * self.config.potential_strength * r_squared

        return V.to(torch.float32)

    def _initialize_field(self):
        """Initialize field Φ(x, color)."""
        N = self.config.grid_size
        C = self.config.num_colors

        if self.config.initial_type == "gaussian_wavepacket":
            r_squared = self.x**2 + self.y**2 + self.z**2
            sigma = self.config.box_size / 8

            psi = self.config.initial_amplitude * torch.exp(-r_squared / (2 * sigma**2))

            field = psi.unsqueeze(0).repeat(C, 1, 1, 1)

            for c in range(C):
                phase = 2 * torch.pi * c / C
                field[c] *= torch.exp(1j * phase * torch.tensor(1.0))

        elif self.config.initial_type == "random":
            field = torch.randn(C, N, N, N, device=self.device, dtype=self.config.dtype)
            field *= self.config.initial_amplitude

        elif self.config.initial_type == "plane_wave":
            k0 = 2 * torch.pi / self.config.box_size
            phase = k0 * self.x

            psi = self.config.initial_amplitude * torch.exp(1j * phase)
            field = psi.unsqueeze(0).repeat(C, 1, 1, 1)

        else:
            raise ValueError(f"Unknown initial_type: {self.config.initial_type}")

        noise = self.config.initial_noise_level * torch.randn(
            C, N, N, N,
            device=self.device,
            dtype=self.config.dtype
        )
        field = field + noise

        return field.to(self.config.dtype)

    def step_kinetic(self):
        """Evolve kinetic term using split-step Fourier."""
        C = self.config.num_colors

        for c in range(C):
            field_spatial = self.field[c]
            field_k = torch.fft.fftn(field_spatial, dim=(0, 1, 2))
            field_k *= self.kinetic_k
            field_spatial = torch.fft.ifftn(field_k, dim=(0, 1, 2))
            self.field[c] = field_spatial

    def step_potential(self):
        """Evolve potential term (including interaction)."""
        density = torch.sum(torch.abs(self.field)**2, dim=0)
        V_total = self.potential + self.config.interaction_strength * density
        phase = -V_total * self.config.dt / self.config.hbar
        operator = torch.exp(1j * phase).to(self.config.dtype)

        for c in range(self.config.num_colors):
            self.field[c] *= operator

    def step(self):
        """Single time step using split-step method."""
        self.step_kinetic()
        self.step_potential()

    def run_evolution(self):
        """Run full time evolution."""
        print("Running GFT field evolution...")

        snapshot_interval = max(1, self.config.timesteps // 10)
        snapshots = []
        density_evolution = []
        energy_evolution = []
        entropy_evolution = []

        for step in range(self.config.timesteps):
            if step % snapshot_interval == 0:
                snapshots.append(self.field.clone().cpu().numpy())

            density = torch.sum(torch.abs(self.field)**2, dim=0)
            density_evolution.append(density.cpu().numpy())

            energy = self.compute_energy()
            energy_evolution.append(energy)

            entropy = self.compute_entropy(density)
            entropy_evolution.append(entropy)

            self.step()

            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{self.config.timesteps}, E={energy:.4f}, S={entropy:.4f}")

        results = {
            'field_snapshots': np.array(snapshots),
            'density_evolution': np.array(density_evolution),
            'energy_evolution': np.array(energy_evolution),
            'entropy_evolution': np.array(entropy_evolution),
            'times': np.arange(self.config.timesteps) * self.config.dt,
        }

        print("Evolution complete!")
        return results

    def compute_energy(self):
        """Compute total energy E = T + V + U."""
        energy_kin = 0.0

        for c in range(self.config.num_colors):
            field_k = torch.fft.fftn(self.field[c], dim=(0, 1, 2))
            k_squared = self.kx**2 + self.ky**2 + self.kz**2

            E_kin_c = torch.sum(
                k_squared * torch.abs(field_k)**2
            ).real * self.config.hbar**2 / (2 * self.config.mass)

            E_kin_c /= self.config.grid_size**3

            energy_kin += E_kin_c.item()

        density = torch.sum(torch.abs(self.field)**2, dim=0)
        energy_pot = torch.sum(self.potential * density).item()
        energy_int = 0.5 * self.config.interaction_strength * torch.sum(density**2).item()

        total_energy = energy_kin + energy_pot + energy_int

        return total_energy

    def compute_entropy(self, density):
        """Compute information entropy S = -Σ p log(p)."""
        prob = density / torch.sum(density)
        prob = torch.clamp(prob, min=1e-12)
        entropy = -torch.sum(prob * torch.log(prob))

        return entropy.item()

    def save_results(self, results, filepath):
        """Save results to file."""
        np.savez_compressed(filepath, **results)
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    config = GFTConfig(
        grid_size=32,
        num_colors=4,
        timesteps=100,
        interaction_strength=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    evolver = GFTFieldEvolver(config)
    results = evolver.run_evolution()

    print(f"\\nFinal energy: {results['energy_evolution'][-1]:.4f}")
    print(f"Final entropy: {results['entropy_evolution'][-1]:.4f}")
'''

# Write the file
def create_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Created: {filepath}")

if __name__ == "__main__":
    create_file('ivhl/multiscale/gft_field.py', GFT_FIELD_CONTENT)
    print("GFT field module created successfully!")
