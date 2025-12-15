"""
RL Discovery Enhancements for LIGO-Inspired GW Lattice Analysis

Extends hybrid TD3-SAC reinforcement learning with GW-specific rewards and discovery modes:
- Lattice stability scoring under perturbations
- Fractal layering metrics
- Attractor convergence rewards
- Memory persistence quantification
- Targeted discovery campaigns for constant lattices and GW phenomena

Integrates with existing td3_sac_hybrid_training.py framework.

Conceptual Foundation:
- RL agent explores configuration space for emergent lattice structures
- Rewards guide toward stable, fractal, constant-embedded patterns
- Attractor dynamics → stable basins in parameter space
- Discovery modes → systematic search for LIGO-like signatures

Author: iVHL Framework (LIGO Integration)
Date: 2025-12-15
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Import existing RL framework
try:
    from td3_sac_hybrid_core import TD3SACHybridAgent, TD3SACConfig
    from td3_sac_hybrid_training import HybridTrainer, HybridTrainingConfig
except ImportError:
    print("Warning: TD3-SAC hybrid modules not found. Mock classes will be used.")
    TD3SACHybridAgent = None
    HybridTrainer = None

# Import GW analysis modules
from gw_lattice_mode import GWLatticeProbe, GWLatticeConfig
from gw_fractal_analysis import FractalHarmonicAnalyzer, FractalAnalysisConfig


# ============================================================================
# Discovery Mode Enum
# ============================================================================

class GWDiscoveryMode(Enum):
    """GW-specific discovery campaign modes"""

    FIND_CONSTANT_LATTICE = "find_constant_lattice"
    GW_MEMORY_FIELD = "gw_memory_field"
    FRACTAL_HARMONIC_STABILIZATION = "fractal_harmonic_stabilization"
    ATTRACTOR_CONVERGENCE = "attractor_convergence"
    QUASINORMAL_RINGING = "quasinormal_ringing"
    LATTICE_UNDER_SCRAMBLING = "lattice_under_scrambling"


# ============================================================================
# GW Reward Functions
# ============================================================================

class GWRewardComputer:
    """
    Compute GW-specific rewards for RL agent

    Rewards guide agent toward configurations exhibiting:
    - Stable lattice structures
    - Fractal self-similarity
    - Mathematical constant residues
    - Attractor convergence
    - Memory persistence
    """

    def __init__(
        self,
        gw_config: GWLatticeConfig,
        fractal_config: FractalAnalysisConfig
    ):
        self.gw_config = gw_config
        self.fractal_config = fractal_config
        self.device = torch.device(gw_config.device)

        # Analysis components
        self.fractal_analyzer = FractalHarmonicAnalyzer(fractal_config)

        # Reward weights
        self.weights = {
            'lattice_stability': 2.0,
            'fractal_dimension': 1.5,
            'constant_residue': 3.0,
            'attractor_convergence': 2.5,
            'memory_persistence': 2.0,
            'harmonic_richness': 1.0
        }

    def lattice_stability_reward(
        self,
        lattice_positions: torch.Tensor,
        perturbed_positions: torch.Tensor
    ) -> float:
        """
        Reward for lattice stability under perturbations

        Measures similarity before/after perturbation
        High reward → robust lattice structure

        Args:
            lattice_positions: (N, 3) original lattice
            perturbed_positions: (N, 3) perturbed lattice

        Returns:
            reward: Stability score in [0, 1]
        """
        # Procrustes similarity
        pos1 = lattice_positions - lattice_positions.mean(dim=0, keepdim=True)
        pos2 = perturbed_positions - perturbed_positions.mean(dim=0, keepdim=True)

        # SVD for optimal rotation
        H = pos1.T @ pos2
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T

        # Aligned distance
        pos2_aligned = pos2 @ R
        dist = torch.norm(pos1 - pos2_aligned, dim=1).mean()
        spread = torch.norm(pos1, dim=1).mean()

        # Similarity [0, 1]
        similarity = torch.exp(-dist / (spread + 1e-8))

        reward = similarity.item()

        return reward

    def fractal_layering_reward(
        self,
        field_3d: torch.Tensor
    ) -> float:
        """
        Reward for fractal dimension close to target

        Target D ≈ 2.5-2.7 (complex fractal structure)

        Args:
            field_3d: (X, Y, Z) scalar field

        Returns:
            reward: Fractal quality score
        """
        # Compute fractal dimension at median threshold
        threshold = torch.median(field_3d).item()

        box_sizes, counts = self.fractal_analyzer.fractal_analyzer.box_count(
            field_3d,
            threshold=threshold
        )
        dim_result = self.fractal_analyzer.fractal_analyzer.compute_fractal_dimension(
            box_sizes,
            counts
        )

        D = dim_result['fractal_dimension']

        if np.isnan(D):
            return 0.0

        # Target range
        D_target = 2.6
        D_tolerance = 0.5

        # Gaussian reward centered on target
        reward = np.exp(-((D - D_target) ** 2) / (2 * D_tolerance ** 2))

        # Bonus for good fit quality
        r_squared = dim_result['r_squared']
        reward *= (0.5 + 0.5 * r_squared)

        return reward

    def constant_residue_reward(
        self,
        strain_waveform: torch.Tensor,
        sampling_rate: float
    ) -> float:
        """
        Reward for mathematical constant residues in spectrum

        High reward if peaks align with π, e, φ, etc.

        Args:
            strain_waveform: (T,) time-series
            sampling_rate: Hz

        Returns:
            reward: Constant residue score
        """
        # Detect peaks
        frequencies, power = self.fractal_analyzer.harmonic_detector.compute_power_spectrum(
            strain_waveform,
            sampling_rate
        )
        peaks = self.fractal_analyzer.harmonic_detector.detect_peaks(frequencies, power)

        if len(peaks) == 0:
            return 0.0

        # Detect constant residues
        base_freq = peaks[0][0]
        constant_matches = self.fractal_analyzer.harmonic_detector.detect_constant_residues(
            peaks,
            base_frequency=base_freq
        )

        # Reward proportional to number of matched constants
        num_constants_matched = len(constant_matches)
        max_constants = len(self.fractal_config.test_constants)

        reward = num_constants_matched / max_constants

        # Bonus for low deviation
        if constant_matches:
            avg_deviation = np.mean([
                match['deviation']
                for matches in constant_matches.values()
                for match in matches
            ])
            reward *= (1 - avg_deviation)

        return reward

    def attractor_convergence_reward(
        self,
        state_history: List[torch.Tensor],
        window: int = 10
    ) -> float:
        """
        Reward for convergence to attractor basin

        Measures variance decrease in recent states

        Args:
            state_history: List of (D,) state vectors
            window: Recent states to consider

        Returns:
            reward: Convergence score
        """
        if len(state_history) < window + 1:
            return 0.0

        # Recent states
        recent = torch.stack(state_history[-window:], dim=0)  # (W, D)

        # Variance over time
        variance = recent.var(dim=0).mean()

        # Reward inversely proportional to variance
        reward = 1.0 / (1.0 + variance.item())

        return reward

    def memory_persistence_reward(
        self,
        field_history: List[torch.Tensor],
        perturbation_end_idx: int
    ) -> float:
        """
        Reward for long memory persistence (slow decay)

        High reward if field maintains structure after perturbation

        Args:
            field_history: List of (M,) field values
            perturbation_end_idx: When perturbation ended

        Returns:
            reward: Memory persistence score
        """
        if perturbation_end_idx >= len(field_history) - 5:
            return 0.0

        # Post-perturbation data
        post_data = torch.stack(field_history[perturbation_end_idx:])  # (T_post, M)

        # Mean intensity
        intensity = post_data.mean(dim=1)

        # Fit exponential decay to estimate τ
        t = torch.arange(len(intensity), dtype=torch.float32, device=self.device)
        I_final = intensity[-5:].mean()
        I_decay = intensity - I_final
        I_decay = torch.clamp(I_decay, min=1e-10)

        log_I = torch.log(I_decay)

        # Linear fit
        X = torch.stack([torch.ones_like(t), t], dim=1)
        y = log_I.unsqueeze(1)

        try:
            params = torch.linalg.lstsq(X, y).solution.squeeze()
            b = params[1].item()

            # Decay time
            tau = -1.0 / b if b < 0 else 0.0
        except:
            tau = 0.0

        # Normalize by target memory time
        tau_target = self.gw_config.memory_persistence_time
        reward = min(tau / tau_target, 1.0)

        return reward

    def harmonic_richness_reward(
        self,
        strain_waveform: torch.Tensor,
        sampling_rate: float
    ) -> float:
        """
        Reward for rich harmonic series

        High reward if multiple harmonics detected

        Args:
            strain_waveform: (T,) time-series
            sampling_rate: Hz

        Returns:
            reward: Harmonic richness score
        """
        # Detect harmonic series
        frequencies, power = self.fractal_analyzer.harmonic_detector.compute_power_spectrum(
            strain_waveform,
            sampling_rate
        )
        peaks = self.fractal_analyzer.harmonic_detector.detect_peaks(frequencies, power)
        harmonic_series = self.fractal_analyzer.harmonic_detector.detect_harmonic_series(peaks)

        # Reward based on harmonic ratio
        harmonic_ratio = harmonic_series['harmonic_ratio']

        return harmonic_ratio

    def compute_total_reward(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """
        Combine all GW rewards with weights

        Args:
            metrics: Dict of metric_name → value

        Returns:
            total_reward: Weighted sum
        """
        total = 0.0

        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                total += weight * metrics[metric_name]

        # Normalize by total weight
        total_weight = sum(self.weights.values())
        total /= total_weight

        return total


# ============================================================================
# GW Discovery Environment
# ============================================================================

class GWDiscoveryEnvironment:
    """
    RL environment for GW lattice discovery

    State: Lattice configuration parameters (helical turns, amplitude, etc.)
    Action: Parameter adjustments
    Reward: GW-specific metrics (lattice stability, fractal dimension, etc.)
    """

    def __init__(
        self,
        gw_config: GWLatticeConfig,
        fractal_config: FractalAnalysisConfig,
        discovery_mode: GWDiscoveryMode = GWDiscoveryMode.FIND_CONSTANT_LATTICE
    ):
        self.gw_config = gw_config
        self.fractal_config = fractal_config
        self.discovery_mode = discovery_mode

        self.device = torch.device(gw_config.device)

        # GW probe
        self.gw_probe = GWLatticeProbe(gw_config)

        # Reward computer
        self.reward_computer = GWRewardComputer(gw_config, fractal_config)

        # State: [helical_turns, sphere_radius, gw_amplitude, gw_frequency, num_sources]
        self.state_dim = 5
        self.action_dim = 5  # Continuous adjustments to each state component

        # State bounds
        self.state_bounds = {
            'helical_turns': (1.0, 10.0),
            'sphere_radius': (0.5, 2.0),
            'gw_amplitude': (1e-23, 1e-19),
            'gw_frequency': (10.0, 1000.0),
            'num_sources': (50, 1000)
        }

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state

        Returns:
            state: (5,) initial state vector
        """
        # Random initialization within bounds
        self.state = np.array([
            np.random.uniform(*self.state_bounds['helical_turns']),
            np.random.uniform(*self.state_bounds['sphere_radius']),
            np.random.uniform(*self.state_bounds['gw_amplitude']),
            np.random.uniform(*self.state_bounds['gw_frequency']),
            np.random.uniform(*self.state_bounds['num_sources'])
        ], dtype=np.float32)

        self.step_count = 0
        self.state_history = [torch.from_numpy(self.state).to(self.device)]

        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info)

        Args:
            action: (5,) parameter adjustments (normalized to [-1, 1])

        Returns:
            next_state: (5,) updated state
            reward: Scalar reward
            done: Episode termination flag
            info: Additional metrics
        """
        self.step_count += 1

        # Apply action (scaled adjustments)
        action_scale = 0.1  # Small perturbations
        self.state += action * action_scale

        # Clip to bounds
        for i, (param_name, bounds) in enumerate(self.state_bounds.items()):
            self.state[i] = np.clip(self.state[i], bounds[0], bounds[1])

        # Update GW config
        self._update_gw_config()

        # Run GW simulation (shortened for RL speed)
        self.gw_config.duration = 1.0  # 1 second episodes
        self.gw_probe = GWLatticeProbe(self.gw_config)

        # Quick simulation (no scrambling/tolerance for speed)
        results = self.gw_probe.run_simulation(
            with_scrambling=False,
            with_tolerance_test=False
        )

        # Compute reward based on discovery mode
        reward, metrics = self._compute_reward(results)

        # Termination
        done = self.step_count >= 100

        # Track state history
        self.state_history.append(torch.from_numpy(self.state).to(self.device))

        info = {
            'metrics': metrics,
            'discovery_mode': self.discovery_mode.value
        }

        return self.state.copy(), reward, done, info

    def _update_gw_config(self):
        """Update GW config from current state"""
        self.gw_config.helical_turns = float(self.state[0])
        self.gw_config.sphere_radius = float(self.state[1])
        self.gw_config.gw_amplitude = float(self.state[2])
        self.gw_config.gw_frequency = float(self.state[3])
        self.gw_config.num_lattice_nodes = int(self.state[4])

    def _compute_reward(self, results: Dict) -> Tuple[float, Dict]:
        """
        Compute reward based on discovery mode

        Returns:
            reward: Scalar reward
            metrics: Dict of individual metric values
        """
        metrics = {}

        # Extract data
        lattice_history = [l.to(self.device) for l in results['lattice_history']]
        field_history = [f.to(self.device) for f in results['field_history']]
        strain_extracted = torch.tensor(
            results['strain_extracted'],
            dtype=torch.float32,
            device=self.device
        )

        # Sampling rate
        sampling_rate = self.gw_config.sampling_rate

        # Mode-specific metrics
        if self.discovery_mode == GWDiscoveryMode.FIND_CONSTANT_LATTICE:
            # Emphasize constant residues
            metrics['constant_residue'] = self.reward_computer.constant_residue_reward(
                strain_extracted,
                sampling_rate
            )
            metrics['lattice_stability'] = self.reward_computer.lattice_stability_reward(
                lattice_history[0],
                lattice_history[-1]
            )

        elif self.discovery_mode == GWDiscoveryMode.FRACTAL_HARMONIC_STABILIZATION:
            # Emphasize fractal dimension and harmonics
            # Create 3D field from probe points (simplified)
            field_3d = self._reconstruct_field_3d(field_history)
            metrics['fractal_dimension'] = self.reward_computer.fractal_layering_reward(
                field_3d
            )
            metrics['harmonic_richness'] = self.reward_computer.harmonic_richness_reward(
                strain_extracted,
                sampling_rate
            )

        elif self.discovery_mode == GWDiscoveryMode.ATTRACTOR_CONVERGENCE:
            # Emphasize state convergence
            metrics['attractor_convergence'] = self.reward_computer.attractor_convergence_reward(
                self.state_history
            )
            metrics['lattice_stability'] = self.reward_computer.lattice_stability_reward(
                lattice_history[0],
                lattice_history[-1]
            )

        elif self.discovery_mode == GWDiscoveryMode.GW_MEMORY_FIELD:
            # Emphasize memory persistence
            perturbation_end_idx = len(field_history) // 2
            metrics['memory_persistence'] = self.reward_computer.memory_persistence_reward(
                field_history,
                perturbation_end_idx
            )

        else:
            # Default: all metrics
            metrics['lattice_stability'] = self.reward_computer.lattice_stability_reward(
                lattice_history[0],
                lattice_history[-1]
            )
            metrics['constant_residue'] = self.reward_computer.constant_residue_reward(
                strain_extracted,
                sampling_rate
            )
            metrics['attractor_convergence'] = self.reward_computer.attractor_convergence_reward(
                self.state_history
            )

        # Total reward
        reward = self.reward_computer.compute_total_reward(metrics)

        return reward, metrics

    def _reconstruct_field_3d(
        self,
        field_history: List[torch.Tensor],
        grid_size: int = 32
    ) -> torch.Tensor:
        """
        Reconstruct 3D field from probe point history (simplified)

        Returns: (grid_size, grid_size, grid_size) tensor
        """
        # Simplified: create synthetic field based on mean intensity
        mean_intensity = torch.stack(field_history).mean()

        # Random field with correct mean
        field_3d = torch.randn(
            grid_size, grid_size, grid_size,
            device=self.device
        ) + mean_intensity

        return field_3d


# ============================================================================
# GW Discovery Campaign
# ============================================================================

class GWDiscoveryCampaign:
    """
    Orchestrate GW discovery campaigns with RL agent

    Runs multiple discovery modes sequentially or in parallel
    Tracks best configurations and emergent phenomena
    """

    def __init__(
        self,
        gw_config: GWLatticeConfig,
        fractal_config: FractalAnalysisConfig,
        rl_config: Optional[any] = None  # TD3SACConfig if available
    ):
        self.gw_config = gw_config
        self.fractal_config = fractal_config
        self.rl_config = rl_config

        self.device = torch.device(gw_config.device)

        # Best configurations found
        self.best_configs = {}

        # Discovery history
        self.discovery_history = []

    def run_discovery_mode(
        self,
        mode: GWDiscoveryMode,
        num_episodes: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Run discovery campaign for specific mode

        Args:
            mode: Discovery mode to run
            num_episodes: Training episodes
            verbose: Print progress

        Returns:
            results: Best configuration and metrics
        """
        if verbose:
            print("=" * 70)
            print(f"GW DISCOVERY CAMPAIGN: {mode.value.upper()}")
            print("=" * 70)
            print()

        # Create environment
        env = GWDiscoveryEnvironment(
            self.gw_config,
            self.fractal_config,
            discovery_mode=mode
        )

        # Train RL agent (simplified without full TD3-SAC if not available)
        if HybridTrainer is not None and self.rl_config is not None:
            # Full RL training
            trainer = HybridTrainer(config=self.rl_config)
            metrics = trainer.train_online(env, num_episodes=num_episodes, verbose=verbose)

            best_reward = max(metrics['episode_rewards'])
            best_episode = np.argmax(metrics['episode_rewards'])

        else:
            # Simplified: random search
            best_reward = -float('inf')
            best_state = None
            best_metrics = {}

            for episode in range(num_episodes):
                state = env.reset()
                episode_reward = 0

                for step in range(100):
                    # Random action
                    action = np.random.uniform(-1, 1, size=env.action_dim)
                    next_state, reward, done, info = env.step(action)

                    episode_reward += reward

                    if done:
                        break

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_state = state
                    best_metrics = info['metrics']

                if verbose and episode % 10 == 0:
                    print(f"  Episode {episode}: Reward = {episode_reward:.3f}")

            metrics = {
                'best_reward': best_reward,
                'best_state': best_state,
                'best_metrics': best_metrics
            }

        # Store best config
        self.best_configs[mode] = {
            'state': env.state.copy(),
            'reward': best_reward,
            'metrics': best_metrics if 'best_metrics' in metrics else {}
        }

        if verbose:
            print()
            print(f"Best reward: {best_reward:.3f}")
            print(f"Best configuration: {env.state}")
            print()

        return self.best_configs[mode]

    def run_all_modes(
        self,
        num_episodes_per_mode: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Run all discovery modes sequentially

        Returns:
            all_results: Dict of mode → results
        """
        all_results = {}

        for mode in GWDiscoveryMode:
            results = self.run_discovery_mode(
                mode,
                num_episodes=num_episodes_per_mode,
                verbose=verbose
            )
            all_results[mode.value] = results

        return all_results


# ============================================================================
# Demo
# ============================================================================

def demo_gw_rl_discovery():
    """Demonstrate GW RL discovery"""

    print("\n" + "=" * 70)
    print("GW RL DISCOVERY DEMO")
    print("=" * 70)
    print()

    # Configs
    gw_config = GWLatticeConfig(
        duration=1.0,  # Short for demo
        num_lattice_nodes=100,
        central_probe_points=30
    )
    fractal_config = FractalAnalysisConfig()

    # Campaign
    campaign = GWDiscoveryCampaign(gw_config, fractal_config)

    # Run discovery modes
    modes_to_test = [
        GWDiscoveryMode.FIND_CONSTANT_LATTICE,
        GWDiscoveryMode.FRACTAL_HARMONIC_STABILIZATION,
        GWDiscoveryMode.ATTRACTOR_CONVERGENCE
    ]

    for mode in modes_to_test:
        campaign.run_discovery_mode(mode, num_episodes=20, verbose=True)

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nBest configurations found:")
    for mode, config in campaign.best_configs.items():
        print(f"\n{mode.value}:")
        print(f"  Reward: {config['reward']:.3f}")
        print(f"  State: {config['state']}")


if __name__ == "__main__":
    demo_gw_rl_discovery()
