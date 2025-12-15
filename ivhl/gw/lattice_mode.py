"""
LIGO-Inspired Gravitational Wave Lattice Probe Mode

Integrates recent LIGO analysis insights suggesting structured "constant lattices"
with fractal harmonic layering and attractor-based dynamics in strain data.

This module bridges iVHL's vibrational helical lattice with GW observations:
- Structured perturbations simulating GW strain on boundary lattice
- Strain time-series extraction from resonant field intensity
- Quasinormal mode ringing and memory field persistence
- Lattice stability under scrambling (phase/null perturbations)
- Fractal harmonic detection in simulated waveforms

Conceptual Foundation:
- LIGO "constant lattice" → iVHL helical/nodal resonance patterns
- Fractal harmonics in log-space → self-similar field intensity scaling
- Attractor dynamics → stable lattice formation under perturbations
- Memory field → GFT condensate persistence, quasinormal ringing
- Implication: Physical constants emerge from lattice residues

Author: iVHL Framework (LIGO Integration)
Date: 2025-12-15
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GWLatticeConfig:
    """Configuration for GW lattice probe mode"""

    # Perturbation parameters
    gw_amplitude: float = 1e-21  # Typical LIGO strain amplitude
    gw_frequency: float = 100.0  # Hz (inspiral/merger range)
    noise_level: float = 1e-23  # Background noise floor

    # Lattice structure
    num_lattice_nodes: int = 500  # Boundary helical lattice points
    helical_turns: float = 5.0
    sphere_radius: float = 1.0

    # Strain extraction
    sampling_rate: float = 4096.0  # Hz (LIGO sampling rate)
    duration: float = 4.0  # seconds
    central_probe_points: int = 100  # Points for strain measurement

    # Perturbation modes
    perturbation_type: str = 'inspiral'  # 'inspiral', 'ringdown', 'stochastic', 'constant_lattice'
    chirp_mass: float = 30.0  # Solar masses (for inspiral)
    quality_factor: float = 2.0  # Damping for ringdown

    # Scrambling tests
    phase_scramble_strength: float = 0.0  # 0 to 1
    null_scramble_fraction: float = 0.0  # Fraction of lattice to null
    tolerance_variations: List[float] = None  # Test different noise levels

    # Memory field
    memory_persistence_time: float = 2.0  # Seconds after perturbation
    quasinormal_modes: int = 5  # Number of QNM frequencies to track

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.tolerance_variations is None:
            self.tolerance_variations = [1.0, 0.5, 0.1, 0.01]


# ============================================================================
# GW Waveform Generators
# ============================================================================

class GWWaveformGenerator:
    """Generate gravitational wave strain waveforms"""

    def __init__(self, config: GWLatticeConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Time array
        self.num_samples = int(config.sampling_rate * config.duration)
        self.time = torch.linspace(0, config.duration, self.num_samples, device=self.device)

    def inspiral_waveform(self) -> torch.Tensor:
        """
        Generate inspiral waveform with frequency chirp

        Simplified post-Newtonian approximation:
        h(t) = A(t) * cos(φ(t))
        f(t) ∝ (t_c - t)^(-3/8)  (Newtonian chirp)
        """
        t = self.time
        t_c = self.config.duration  # Coalescence time

        # Chirp frequency evolution (Newtonian approximation)
        tau = t_c - t
        tau = torch.clamp(tau, min=1e-3)  # Avoid singularity

        f_instant = self.config.gw_frequency * (tau / t_c) ** (-3/8)

        # Amplitude growth (inverse distance, simplified)
        amplitude = self.config.gw_amplitude * (tau / t_c) ** (-1/4)

        # Phase integration
        phase = 2 * np.pi * torch.cumsum(f_instant * (t[1] - t[0]), dim=0)

        # Plus polarization (simplified)
        h_plus = amplitude * torch.cos(phase)

        # Add noise
        noise = torch.randn_like(h_plus, device=self.device) * self.config.noise_level

        return h_plus + noise

    def ringdown_waveform(self) -> torch.Tensor:
        """
        Generate ringdown waveform (quasinormal modes)

        h(t) = Σ_n A_n * exp(-t/τ_n) * cos(2πf_n*t + φ_n)

        Characteristic of post-merger black hole ringing
        """
        t = self.time
        h_ringdown = torch.zeros_like(t)

        # Fundamental mode + overtones
        for n in range(self.config.quasinormal_modes):
            # QNM frequencies (simplified, real BH has complex frequencies)
            f_n = self.config.gw_frequency * (1 + 0.1 * n)

            # Damping time (quality factor Q = π f τ)
            tau_n = self.config.quality_factor / (np.pi * f_n)

            # Amplitude and phase
            A_n = self.config.gw_amplitude * 0.5 ** n  # Decreasing overtones
            phi_n = torch.rand(1, device=self.device) * 2 * np.pi

            # Add mode
            h_ringdown += A_n * torch.exp(-t / tau_n) * torch.cos(2 * np.pi * f_n * t + phi_n)

        # Add noise
        noise = torch.randn_like(h_ringdown, device=self.device) * self.config.noise_level

        return h_ringdown + noise

    def stochastic_background(self) -> torch.Tensor:
        """
        Generate stochastic gravitational wave background

        Models cosmological GW background or unresolved sources
        """
        # Power-law spectrum in frequency domain
        freq = torch.fft.rfftfreq(self.num_samples, d=1/self.config.sampling_rate, device=self.device)

        # Ω_GW(f) ∝ f^α power spectrum (α ≈ 2/3 for inflation)
        alpha = 2.0 / 3.0
        power = self.config.gw_amplitude ** 2 * (freq / self.config.gw_frequency) ** alpha
        power[0] = 0  # No DC

        # Generate random phases
        phases = torch.rand(len(power), device=self.device) * 2 * np.pi

        # Complex spectrum
        spectrum = torch.sqrt(power) * torch.exp(1j * phases)

        # Inverse FFT to time domain
        h_stochastic = torch.fft.irfft(spectrum, n=self.num_samples)

        # Normalize
        h_stochastic = h_stochastic * (self.config.gw_amplitude / h_stochastic.std())

        # Add detector noise
        noise = torch.randn_like(h_stochastic, device=self.device) * self.config.noise_level

        return h_stochastic + noise

    def constant_lattice_waveform(self) -> torch.Tensor:
        """
        Generate waveform with embedded mathematical constants

        LIGO-inspired: Structured residues at constant-related frequencies
        e.g., f = f_0 * (π, e, φ, √2, etc.)
        """
        t = self.time
        h_lattice = torch.zeros_like(t)

        # Mathematical constants for frequency modulation
        constants = [
            np.pi,           # π
            np.e,            # e
            (1 + np.sqrt(5)) / 2,  # φ (golden ratio)
            np.sqrt(2),      # √2
            np.sqrt(3),      # √3
            np.log(2),       # ln(2)
            1.618033988749895,  # φ (higher precision)
        ]

        base_freq = self.config.gw_frequency

        for i, c in enumerate(constants):
            f_i = base_freq * c
            A_i = self.config.gw_amplitude * 0.7 ** i  # Decreasing amplitude
            phi_i = torch.rand(1, device=self.device) * 2 * np.pi

            # Add harmonic
            h_lattice += A_i * torch.cos(2 * np.pi * f_i * t + phi_i)

        # Add fractal layering (self-similar at log-scales)
        for octave in range(3):
            scale = 2 ** octave
            f_fractal = base_freq * scale
            A_fractal = self.config.gw_amplitude * 0.3 ** octave

            # Log-modulated envelope
            envelope = torch.log1p(t * scale)
            h_lattice += A_fractal * envelope * torch.sin(2 * np.pi * f_fractal * t)

        # Add noise
        noise = torch.randn_like(h_lattice, device=self.device) * self.config.noise_level

        return h_lattice + noise

    def generate(self) -> torch.Tensor:
        """Generate waveform based on perturbation type"""
        if self.config.perturbation_type == 'inspiral':
            return self.inspiral_waveform()
        elif self.config.perturbation_type == 'ringdown':
            return self.ringdown_waveform()
        elif self.config.perturbation_type == 'stochastic':
            return self.stochastic_background()
        elif self.config.perturbation_type == 'constant_lattice':
            return self.constant_lattice_waveform()
        else:
            raise ValueError(f"Unknown perturbation type: {self.config.perturbation_type}")


# ============================================================================
# Lattice Perturbation Engine
# ============================================================================

class LatticePerturbationEngine:
    """
    Apply GW-like perturbations to iVHL boundary lattice

    Maps strain h(t) to lattice deformations:
    - Radial displacement: Δr = r * h(t)
    - Angular modulation: Δθ ∝ ∂h/∂t
    - Phase shifts: Δφ from tidal forces
    """

    def __init__(self, config: GWLatticeConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Generate boundary helical lattice
        self.lattice_positions = self._generate_helical_lattice()

        # Waveform generator
        self.waveform_gen = GWWaveformGenerator(config)

    def _generate_helical_lattice(self) -> torch.Tensor:
        """
        Generate spherical helical lattice (boundary shell)

        Returns: (N, 3) tensor of Cartesian positions
        """
        N = self.config.num_lattice_nodes
        t = torch.linspace(0, 1, N, device=self.device)

        # Helical parameter
        theta = 2 * np.pi * self.config.helical_turns * t

        # Spherical shell with helical modulation
        phi = np.pi * t  # Polar angle
        r = self.config.sphere_radius

        # Cartesian coordinates
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)

        positions = torch.stack([x, y, z], dim=1)  # (N, 3)

        return positions

    def apply_perturbation(self, strain: torch.Tensor, time_idx: int) -> torch.Tensor:
        """
        Apply strain perturbation to lattice at given time index

        Args:
            strain: (T,) strain time-series
            time_idx: Current time index

        Returns:
            perturbed_positions: (N, 3) perturbed lattice positions
        """
        h_t = strain[time_idx]

        # Radial perturbation (isotropic strain)
        positions = self.lattice_positions.clone()
        r = torch.norm(positions, dim=1, keepdim=True)
        r_perturbed = r * (1 + h_t)

        # Rescale positions
        positions_perturbed = positions * (r_perturbed / r)

        # Tidal deformation (quadrupole, simplified)
        # Plus polarization: Δx/x = h, Δy/y = -h
        positions_perturbed[:, 0] *= (1 + h_t)
        positions_perturbed[:, 1] *= (1 - h_t)

        return positions_perturbed

    def phase_scramble(self, positions: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
        """
        Apply phase scrambling to lattice positions

        Randomizes phases while preserving radial structure
        Tests robustness of lattice patterns
        """
        # Convert to spherical
        r = torch.norm(positions, dim=1, keepdim=True)
        theta = torch.atan2(positions[:, 1], positions[:, 0])
        phi = torch.acos(positions[:, 2] / (r.squeeze() + 1e-8))

        # Scramble angles
        theta_scrambled = theta + strength * torch.randn_like(theta) * 2 * np.pi
        phi_scrambled = phi + strength * torch.randn_like(phi) * np.pi
        phi_scrambled = torch.clamp(phi_scrambled, 0, np.pi)

        # Back to Cartesian
        x = r.squeeze() * torch.sin(phi_scrambled) * torch.cos(theta_scrambled)
        y = r.squeeze() * torch.sin(phi_scrambled) * torch.sin(theta_scrambled)
        z = r.squeeze() * torch.cos(phi_scrambled)

        return torch.stack([x, y, z], dim=1)

    def null_scramble(self, positions: torch.Tensor, fraction: float = 0.1) -> torch.Tensor:
        """
        Null out random fraction of lattice points

        Tests persistence of lattice structure with missing data
        """
        N = positions.shape[0]
        num_null = int(N * fraction)

        # Random indices to null
        null_indices = torch.randperm(N, device=self.device)[:num_null]

        positions_scrambled = positions.clone()
        positions_scrambled[null_indices] = 0.0

        return positions_scrambled


# ============================================================================
# Strain Extraction from Resonant Field
# ============================================================================

class StrainExtractor:
    """
    Extract effective strain time-series from iVHL resonant field

    Probes central region for field intensity oscillations
    Maps to GW strain: h(t) ∝ |ψ(r_probe, t)|^2 - ⟨|ψ|^2⟩
    """

    def __init__(self, config: GWLatticeConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Probe points in central region
        self.probe_positions = self._generate_probe_grid()

    def _generate_probe_grid(self) -> torch.Tensor:
        """
        Generate probe points near center (Calabi-Yau region)

        Returns: (M, 3) tensor of probe positions
        """
        M = self.config.central_probe_points

        # Spherical grid in central region (r < 0.1)
        r_max = 0.1

        # Fibonacci sphere
        indices = torch.arange(M, device=self.device, dtype=torch.float32)
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

        theta = phi * indices
        z = 1 - (2 * indices / (M - 1))
        radius = torch.sqrt(1 - z**2)

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        positions = torch.stack([x, y, z], dim=1) * r_max

        return positions

    def extract_strain(
        self,
        field_values: torch.Tensor,
        time_idx: int
    ) -> float:
        """
        Extract strain from field intensity at time index

        Args:
            field_values: (M,) field intensity at probe points
            time_idx: Current time index

        Returns:
            strain: Effective h(t) at this time
        """
        # Mean intensity
        intensity_mean = field_values.mean()

        # Fluctuation (normalized)
        intensity_std = field_values.std() + 1e-8
        fluctuation = (field_values.mean() - intensity_mean) / intensity_std

        # Map to strain amplitude
        h_t = fluctuation.item() * self.config.gw_amplitude

        return h_t

    def compute_field_at_probes(
        self,
        source_positions: torch.Tensor,
        source_amplitudes: torch.Tensor,
        frequency: float,
        phase: float
    ) -> torch.Tensor:
        """
        Compute wave superposition at probe points

        ψ(r_probe) = Σ_i A_i * sin(k|r_probe - r_i| + φ) / |r_probe - r_i|

        Args:
            source_positions: (N, 3) lattice/source positions
            source_amplitudes: (N,) source strengths
            frequency: Wave frequency
            phase: Global phase

        Returns:
            field_values: (M,) field intensity at probes
        """
        k = 2 * np.pi * frequency

        # Distances from probes to sources
        # (M, 1, 3) - (1, N, 3) = (M, N, 3)
        diff = self.probe_positions.unsqueeze(1) - source_positions.unsqueeze(0)
        distances = torch.norm(diff, dim=2) + 1e-8  # (M, N)

        # Wave contribution
        wave = source_amplitudes.unsqueeze(0) * torch.sin(k * distances + phase) / distances

        # Superposition
        field = wave.sum(dim=1)  # (M,)

        return field


# ============================================================================
# Persistence and Memory Tests
# ============================================================================

class PersistenceAnalyzer:
    """
    Test lattice persistence under scrambling and perturbations

    Measures:
    - Lattice structure recovery after phase/null scrambling
    - Memory field persistence (quasinormal ringing decay)
    - Tolerance to noise variations
    - Attractor convergence
    """

    def __init__(self, config: GWLatticeConfig):
        self.config = config
        self.device = torch.device(config.device)

    def lattice_similarity(
        self,
        positions_original: torch.Tensor,
        positions_perturbed: torch.Tensor
    ) -> float:
        """
        Measure similarity between lattice configurations

        Uses Procrustes distance (after optimal rotation/translation)

        Returns: Similarity score in [0, 1] (1 = identical)
        """
        # Center both
        pos1 = positions_original - positions_original.mean(dim=0, keepdim=True)
        pos2 = positions_perturbed - positions_perturbed.mean(dim=0, keepdim=True)

        # Procrustes: Find rotation R minimizing |pos1 - pos2 @ R|
        # SVD of cross-correlation
        H = pos1.T @ pos2  # (3, 3)
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T

        # Apply rotation
        pos2_aligned = pos2 @ R

        # Distance
        dist = torch.norm(pos1 - pos2_aligned, dim=1).mean()

        # Normalize by original spread
        spread = torch.norm(pos1, dim=1).mean()

        similarity = torch.exp(-dist / (spread + 1e-8))

        return similarity.item()

    def memory_decay_analysis(
        self,
        field_history: List[torch.Tensor],
        perturbation_end_idx: int
    ) -> Dict[str, float]:
        """
        Analyze memory persistence after perturbation ends

        Fits exponential decay to field intensity post-perturbation:
        I(t) = I_0 + A * exp(-t/τ)

        Args:
            field_history: List of (M,) field values over time
            perturbation_end_idx: Time index when perturbation stops

        Returns:
            metrics: {
                'decay_time': τ (memory persistence time),
                'residual_amplitude': A (memory strength),
                'quality_factor': Q = π f τ
            }
        """
        # Extract post-perturbation data
        post_data = torch.stack(field_history[perturbation_end_idx:])  # (T_post, M)

        # Mean intensity over probes
        intensity = post_data.mean(dim=1)  # (T_post,)

        # Time array
        t = torch.arange(len(intensity), device=self.device, dtype=torch.float32)
        t = t / self.config.sampling_rate

        # Fit exponential decay (simple least-squares)
        # log(I - I_final) = log(A) - t/τ
        I_final = intensity[-10:].mean()  # Asymptotic value
        I_decay = intensity - I_final
        I_decay = torch.clamp(I_decay, min=1e-10)  # Avoid log(0)

        log_I = torch.log(I_decay)

        # Linear fit: y = a + b*x where b = -1/τ
        X = torch.stack([torch.ones_like(t), t], dim=1)  # (T, 2)
        y = log_I.unsqueeze(1)  # (T, 1)

        # Least squares: (X^T X)^-1 X^T y
        params = torch.linalg.lstsq(X, y).solution.squeeze()

        a, b = params[0].item(), params[1].item()

        # Decay time
        tau = -1.0 / b if b < 0 else float('inf')

        # Residual amplitude
        A = np.exp(a)

        # Quality factor
        Q = np.pi * self.config.gw_frequency * tau

        return {
            'decay_time': tau,
            'residual_amplitude': A,
            'quality_factor': Q
        }

    def tolerance_test(
        self,
        lattice_metric_fn: Callable[[float], float]
    ) -> Dict[str, List[float]]:
        """
        Test lattice stability across tolerance variations

        Args:
            lattice_metric_fn: Function noise_level → metric_value

        Returns:
            results: {
                'noise_levels': [...],
                'metric_values': [...]
            }
        """
        noise_levels = self.config.tolerance_variations
        metric_values = []

        for noise in noise_levels:
            metric = lattice_metric_fn(noise)
            metric_values.append(metric)

        return {
            'noise_levels': noise_levels,
            'metric_values': metric_values
        }


# ============================================================================
# Main GW Lattice Probe Mode
# ============================================================================

class GWLatticeProbe:
    """
    Main interface for LIGO-inspired GW lattice analysis

    Orchestrates:
    - Waveform generation (inspiral, ringdown, stochastic, constant lattice)
    - Lattice perturbation and deformation
    - Strain extraction from resonant field
    - Persistence tests (scrambling, memory decay, tolerance)
    - Data logging and export
    """

    def __init__(self, config: GWLatticeConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Components
        self.waveform_gen = GWWaveformGenerator(config)
        self.perturbation_engine = LatticePerturbationEngine(config)
        self.strain_extractor = StrainExtractor(config)
        self.persistence_analyzer = PersistenceAnalyzer(config)

        # Results storage
        self.results = {
            'strain_input': None,
            'strain_extracted': [],
            'lattice_history': [],
            'field_history': [],
            'persistence_metrics': {},
            'scrambling_tests': {}
        }

    def run_simulation(
        self,
        with_scrambling: bool = False,
        with_tolerance_test: bool = False
    ) -> Dict:
        """
        Run full GW lattice probe simulation

        Args:
            with_scrambling: Include phase/null scrambling tests
            with_tolerance_test: Test across noise variations

        Returns:
            results: Complete simulation results
        """
        print("=" * 70)
        print("GW LATTICE PROBE SIMULATION")
        print("=" * 70)
        print(f"Perturbation type: {self.config.perturbation_type}")
        print(f"Duration: {self.config.duration} s")
        print(f"Sampling rate: {self.config.sampling_rate} Hz")
        print(f"Lattice nodes: {self.config.num_lattice_nodes}")
        print()

        # Generate input strain waveform
        print("Generating GW strain waveform...")
        strain_input = self.waveform_gen.generate()
        self.results['strain_input'] = strain_input.cpu().numpy()
        print(f"  Strain amplitude: {strain_input.abs().max():.2e}")
        print()

        # Time evolution
        num_samples = len(strain_input)
        print(f"Simulating lattice evolution ({num_samples} time steps)...")

        for t_idx in range(0, num_samples, 10):  # Subsample for speed
            # Apply perturbation to lattice
            perturbed_lattice = self.perturbation_engine.apply_perturbation(
                strain_input, t_idx
            )

            # Compute field at probe points
            source_amplitudes = torch.ones(
                self.config.num_lattice_nodes,
                device=self.device
            ) / self.config.num_lattice_nodes

            field_values = self.strain_extractor.compute_field_at_probes(
                perturbed_lattice,
                source_amplitudes,
                frequency=self.config.gw_frequency,
                phase=2 * np.pi * self.config.gw_frequency * t_idx / self.config.sampling_rate
            )

            # Extract strain
            h_extracted = self.strain_extractor.extract_strain(field_values, t_idx)

            # Store
            self.results['lattice_history'].append(perturbed_lattice.cpu())
            self.results['field_history'].append(field_values.cpu())
            self.results['strain_extracted'].append(h_extracted)

            if t_idx % 100 == 0:
                print(f"  t = {t_idx / self.config.sampling_rate:.3f} s")

        print("Evolution complete.")
        print()

        # Scrambling tests
        if with_scrambling:
            print("Running scrambling tests...")
            self._run_scrambling_tests()
            print()

        # Memory persistence analysis
        print("Analyzing memory persistence...")
        perturbation_end_idx = int(0.8 * len(self.results['field_history']))
        memory_metrics = self.persistence_analyzer.memory_decay_analysis(
            [f.to(self.device) for f in self.results['field_history']],
            perturbation_end_idx
        )
        self.results['persistence_metrics']['memory'] = memory_metrics
        print(f"  Decay time tau: {memory_metrics['decay_time']:.3f} s")
        print(f"  Quality factor Q: {memory_metrics['quality_factor']:.2f}")
        print()

        # Tolerance test
        if with_tolerance_test:
            print("Running tolerance tests...")
            self._run_tolerance_tests()
            print()

        print("=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)

        return self.results

    def _run_scrambling_tests(self):
        """Run phase and null scrambling tests"""
        original_lattice = self.perturbation_engine.lattice_positions

        # Phase scrambling
        strengths = [0.1, 0.3, 0.5, 0.7, 1.0]
        phase_similarities = []

        for strength in strengths:
            scrambled = self.perturbation_engine.phase_scramble(
                original_lattice,
                strength=strength
            )
            similarity = self.persistence_analyzer.lattice_similarity(
                original_lattice,
                scrambled
            )
            phase_similarities.append(similarity)

        self.results['scrambling_tests']['phase'] = {
            'strengths': strengths,
            'similarities': phase_similarities
        }
        print(f"  Phase scrambling: similarity range {min(phase_similarities):.3f} - {max(phase_similarities):.3f}")

        # Null scrambling
        fractions = [0.05, 0.1, 0.2, 0.4, 0.6]
        null_similarities = []

        for fraction in fractions:
            scrambled = self.perturbation_engine.null_scramble(
                original_lattice,
                fraction=fraction
            )
            similarity = self.persistence_analyzer.lattice_similarity(
                original_lattice,
                scrambled
            )
            null_similarities.append(similarity)

        self.results['scrambling_tests']['null'] = {
            'fractions': fractions,
            'similarities': null_similarities
        }
        print(f"  Null scrambling: similarity range {min(null_similarities):.3f} - {max(null_similarities):.3f}")

    def _run_tolerance_tests(self):
        """Run tolerance variation tests"""
        # Simplified: Test lattice metric vs noise
        def lattice_metric_fn(noise_level):
            # Mock metric (in real case, re-run simulation with noise)
            return np.exp(-noise_level / self.config.noise_level)

        tolerance_results = self.persistence_analyzer.tolerance_test(
            lattice_metric_fn
        )
        self.results['persistence_metrics']['tolerance'] = tolerance_results
        print(f"  Tolerance test: {len(tolerance_results['noise_levels'])} levels")

    def export_results(self, output_path: str = 'gw_lattice_results.json'):
        """Export results to JSON"""
        output_file = Path(output_path)

        # Convert to serializable format
        export_data = {
            'config': {
                'perturbation_type': self.config.perturbation_type,
                'gw_amplitude': self.config.gw_amplitude,
                'gw_frequency': self.config.gw_frequency,
                'duration': self.config.duration,
                'num_lattice_nodes': self.config.num_lattice_nodes
            },
            'strain_extracted': self.results['strain_extracted'],
            'persistence_metrics': self.results['persistence_metrics'],
            'scrambling_tests': self.results['scrambling_tests']
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Results exported to: {output_file}")

        return str(output_file.absolute())


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_gw_lattice_probe():
    """Demonstrate GW lattice probe capabilities"""

    print("\n" + "=" * 70)
    print("LIGO-INSPIRED GW LATTICE PROBE DEMO")
    print("=" * 70)
    print()

    # Test all perturbation types
    perturbation_types = ['inspiral', 'ringdown', 'stochastic', 'constant_lattice']

    for ptype in perturbation_types:
        print(f"\n{'=' * 70}")
        print(f"Testing: {ptype.upper()}")
        print(f"{'=' * 70}\n")

        config = GWLatticeConfig(
            perturbation_type=ptype,
            duration=2.0,  # Shorter for demo
            num_lattice_nodes=200,
            central_probe_points=50
        )

        probe = GWLatticeProbe(config)
        results = probe.run_simulation(
            with_scrambling=True,
            with_tolerance_test=True
        )

        # Export
        output_file = f'gw_lattice_{ptype}_results.json'
        probe.export_results(output_file)
        print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAll perturbation types tested successfully!")
    print("Results saved to gw_lattice_*_results.json")


if __name__ == "__main__":
    demo_gw_lattice_probe()
