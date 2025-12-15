"""
Perturbation Engine
===================

Applies GW-inspired perturbations to lattice and measures stability metrics.

DISCLAIMER: This explores mathematical lattice stability, not claims about
gravitational waves or physical spacetime.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PerturbationConfig:
    """Configuration for perturbation engine."""
    waveform_type: str = "inspiral"  # or "burst", "continuous"
    amplitude: float = 0.1
    frequency_start: float = 0.1
    frequency_end: float = 1.0
    duration: float = 10.0
    sampling_rate: float = 100.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PerturbationEngine:
    """
    Applies perturbations to lattice and measures stability.
    """
    
    def __init__(self, config: PerturbationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Generate waveform
        self.waveform = self._generate_waveform()
        
        print(f"PerturbationEngine initialized:")
        print(f"  - Waveform type: {config.waveform_type}")
        print(f"  - Amplitude: {config.amplitude}")
        print(f"  - Device: {self.device}")
    
    def _generate_waveform(self) -> torch.Tensor:
        """Generate GW-inspired waveform h(t)."""
        T = self.config.duration
        fs = self.config.sampling_rate
        t = torch.arange(0, T, 1/fs, device=self.device)
        
        if self.config.waveform_type == "inspiral":
            # Chirp: frequency increases
            f_start = self.config.frequency_start
            f_end = self.config.frequency_end
            f_t = f_start + (f_end - f_start) * (t / T)**3
            phase = 2 * torch.pi * torch.cumsum(f_t, dim=0) / fs
            h_t = self.config.amplitude * torch.sin(phase)
        
        elif self.config.waveform_type == "burst":
            # Gaussian envelope
            center = T / 2
            sigma = T / 10
            envelope = torch.exp(-(t - center)**2 / (2 * sigma**2))
            h_t = self.config.amplitude * envelope * torch.sin(
                2 * torch.pi * self.config.frequency_start * t
            )
        
        else:  # continuous
            h_t = self.config.amplitude * torch.sin(
                2 * torch.pi * self.config.frequency_start * t
            )
        
        return h_t
    
    def apply_perturbation(
        self,
        lattice: torch.Tensor,
        time_idx: int
    ) -> torch.Tensor:
        """
        Apply perturbation at given time to lattice.
        
        Args:
            lattice: Position tensor (N, 3)
            time_idx: Time index in waveform
            
        Returns:
            Perturbed lattice positions
        """
        if time_idx >= len(self.waveform):
            h = 0.0
        else:
            h = self.waveform[time_idx]
        
        # Apply strain (simplified)
        perturbed = lattice.clone()
        perturbed[:, 0] *= (1 + h)  # x-direction stretch
        perturbed[:, 1] *= (1 - h)  # y-direction compress
        
        return perturbed
    
    def compute_stability_metric(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor
    ) -> float:
        """
        Compute lattice stability via Procrustes distance.
        
        Args:
            original: Original positions (N, 3)
            perturbed: Perturbed positions (N, 3)
            
        Returns:
            Stability metric (lower = more stable)
        """
        # Center both
        orig_centered = original - original.mean(dim=0)
        pert_centered = perturbed - perturbed.mean(dim=0)
        
        # Normalize
        orig_norm = orig_centered / torch.norm(orig_centered)
        pert_norm = pert_centered / torch.norm(pert_centered)
        
        # Optimal rotation via SVD
        H = orig_norm.T @ pert_norm
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Procrustes distance
        aligned = pert_norm @ R.T
        distance = torch.norm(orig_norm - aligned)
        
        return distance.item()
    
    def run_perturbation_campaign(
        self,
        lattice: np.ndarray
    ) -> Dict:
        """
        Run full perturbation campaign on lattice.
        
        Args:
            lattice: Initial lattice positions (numpy)
            
        Returns:
            Dictionary with stability metrics
        """
        print("Running perturbation campaign...")
        
        lattice_tensor = torch.tensor(
            lattice,
            device=self.device,
            dtype=torch.float32
        )
        
        num_steps = len(self.waveform)
        stability_metrics = []
        
        for t in range(0, num_steps, 10):  # Sample every 10 steps
            perturbed = self.apply_perturbation(lattice_tensor, t)
            metric = self.compute_stability_metric(lattice_tensor, perturbed)
            stability_metrics.append(metric)
        
        results = {
            'stability_metrics': np.array(stability_metrics),
            'waveform': self.waveform.cpu().numpy(),
            'mean_stability': np.mean(stability_metrics),
            'max_perturbation': np.max(stability_metrics),
        }
        
        print(f"Campaign complete!")
        print(f"  - Mean stability: {results['mean_stability']:.6f}")
        print(f"  - Max perturbation: {results['max_perturbation']:.6f}")
        
        return results


if __name__ == "__main__":
    config = PerturbationConfig(
        waveform_type="inspiral",
        amplitude=0.1
    )
    
    engine = PerturbationEngine(config)
    
    # Test with random lattice
    lattice = np.random.randn(126, 3)
    results = engine.run_perturbation_campaign(lattice)
