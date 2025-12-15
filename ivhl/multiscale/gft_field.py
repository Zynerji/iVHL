# -*- coding: utf-8 -*-
"""
Group Field Theory (GFT) Field Evolver
=======================================

Simulates evolution of a complex-valued tensor field with colored
indices, using Gross-Pitaevskii-like dynamics.

DISCLAIMER: This is a mathematical model exploration.
"""

import numpy as np
import torch
import torch.fft
from dataclasses import dataclass
from typing import Dict


@dataclass
class GFTConfig:
    grid_size: int = 32
    box_size: float = 20.0
    num_colors: int = 4
    hbar: float = 1.0
    mass: float = 1.0
    interaction_strength: float = 0.1
    potential_strength: float = 0.5
    timesteps: int = 100
    dt: float = 0.01
    initial_type: str = "gaussian_wavepacket"
    initial_amplitude: float = 1.0
    initial_noise_level: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.complex64


class GFTFieldEvolver:
    def __init__(self, config: GFTConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.x, self.y, self.z = self._create_spatial_grid()
        self.kx, self.ky, self.kz = self._create_momentum_grid()
        self.kinetic_k = self._compute_kinetic_operator()
        self.potential = self._compute_potential()
        self.field = self._initialize_field()
        
        print(f"GFTFieldEvolver initialized on {self.device}")
    
    def _create_spatial_grid(self):
        N = self.config.grid_size
        L = self.config.box_size
        x_1d = torch.linspace(-L/2, L/2, N, device=self.device, dtype=torch.float32)
        x, y, z = torch.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
        return x, y, z
    
    def _create_momentum_grid(self):
        N = self.config.grid_size
        L = self.config.box_size
        k_1d = 2 * torch.pi * torch.fft.fftfreq(N, d=L/N, device=self.device)
        kx, ky, kz = torch.meshgrid(k_1d, k_1d, k_1d, indexing='ij')
        return kx, ky, kz
    
    def _compute_kinetic_operator(self):
        k_squared = self.kx**2 + self.ky**2 + self.kz**2
        exponent = -1j * self.config.hbar * k_squared * self.config.dt / (2 * self.config.mass)
        return torch.exp(exponent).to(self.config.dtype)
    
    def _compute_potential(self):
        r_squared = self.x**2 + self.y**2 + self.z**2
        V = 0.5 * self.config.potential_strength * r_squared
        return V.to(torch.float32)
    
    def _initialize_field(self):
        N = self.config.grid_size
        C = self.config.num_colors
        
        r_squared = self.x**2 + self.y**2 + self.z**2
        sigma = self.config.box_size / 8
        psi = self.config.initial_amplitude * torch.exp(-r_squared / (2 * sigma**2))
        field = psi.unsqueeze(0).repeat(C, 1, 1, 1)
        
        for c in range(C):
            phase = 2 * torch.pi * c / C
            field[c] *= torch.exp(1j * phase * torch.tensor(1.0))
        
        noise = self.config.initial_noise_level * torch.randn(
            C, N, N, N, device=self.device, dtype=self.config.dtype
        )
        return (field + noise).to(self.config.dtype)
    
    def step_kinetic(self):
        for c in range(self.config.num_colors):
            field_k = torch.fft.fftn(self.field[c], dim=(0, 1, 2))
            field_k *= self.kinetic_k
            self.field[c] = torch.fft.ifftn(field_k, dim=(0, 1, 2))
    
    def step_potential(self):
        density = torch.sum(torch.abs(self.field)**2, dim=0)
        V_total = self.potential + self.config.interaction_strength * density
        phase = -V_total * self.config.dt / self.config.hbar
        operator = torch.exp(1j * phase).to(self.config.dtype)
        
        for c in range(self.config.num_colors):
            self.field[c] *= operator
    
    def step(self):
        self.step_kinetic()
        self.step_potential()
    
    def run_evolution(self):
        print("Running GFT field evolution...")
        
        snapshot_interval = max(1, self.config.timesteps // 10)
        snapshots, density_evolution = [], []
        energy_evolution, entropy_evolution = [], []
        
        for step in range(self.config.timesteps):
            if step % snapshot_interval == 0:
                snapshots.append(self.field.clone().cpu().numpy())
            
            density = torch.sum(torch.abs(self.field)**2, dim=0)
            density_evolution.append(density.cpu().numpy())
            energy_evolution.append(self.compute_energy())
            entropy_evolution.append(self.compute_entropy(density))
            
            self.step()
            
            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{self.config.timesteps}")
        
        return {
            'field_snapshots': np.array(snapshots),
            'density_evolution': np.array(density_evolution),
            'energy_evolution': np.array(energy_evolution),
            'entropy_evolution': np.array(entropy_evolution),
            'times': np.arange(self.config.timesteps) * self.config.dt,
        }
    
    def compute_energy(self):
        energy_kin = 0.0
        for c in range(self.config.num_colors):
            field_k = torch.fft.fftn(self.field[c], dim=(0, 1, 2))
            k_squared = self.kx**2 + self.ky**2 + self.kz**2
            E_kin_c = torch.sum(k_squared * torch.abs(field_k)**2).real
            E_kin_c *= self.config.hbar**2 / (2 * self.config.mass * self.config.grid_size**3)
            energy_kin += E_kin_c.item()
        
        density = torch.sum(torch.abs(self.field)**2, dim=0)
        energy_pot = torch.sum(self.potential * density).item()
        energy_int = 0.5 * self.config.interaction_strength * torch.sum(density**2).item()
        
        return energy_kin + energy_pot + energy_int
    
    def compute_entropy(self, density):
        prob = density / torch.sum(density)
        prob = torch.clamp(prob, min=1e-12)
        return -torch.sum(prob * torch.log(prob)).item()


if __name__ == "__main__":
    config = GFTConfig(grid_size=16, timesteps=50)
    evolver = GFTFieldEvolver(config)
    results = evolver.run_evolution()
    print(f"Final energy: {results['energy_evolution'][-1]:.4f}")
