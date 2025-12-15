"""
SAC Reward Engineering for iVHL Discovery

Implements sophisticated reward functions for reinforcement learning-driven
discovery in the holographic framework. Combines dense exploration rewards
with sparse novelty bonuses for efficient learning of rare phenomena.

Reward Structure:
- Dense rewards: Action diversity, smooth exploration, metric improvements
- Sparse novelty: High rewards for anomalies (Calabi-Yau, unification, phase transitions)
- Shaped rewards: Potential-based for long horizons, curriculum learning

Key Discoveries to Reward:
- Ray-detected new cycles (higher-dimensional traversal patterns)
- Entropy jumps (Page curve peaks, purification events)
- Moduli convergence (Calabi-Yau stabilization)
- GFT phase transitions (geometric ↔ non-geometric)
- Unification predictive hits (cross-stack consistency)
- Bit-thread bundling anomalies
- Quasinormal mode resonances
- Hawking pair correlations

Author: iVHL Framework
Date: 2025-12-15
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass


# ============================================================================
# Reward Configuration
# ============================================================================

@dataclass
class RewardConfig:
    """Configuration for reward computation"""

    # Dense reward weights
    action_diversity_weight: float = 0.1
    exploration_bonus_weight: float = 0.2
    metric_improvement_weight: float = 0.3
    smoothness_weight: float = 0.05

    # Sparse novelty rewards
    ray_cycle_reward: float = 10.0
    entropy_jump_reward: float = 15.0
    moduli_convergence_reward: float = 20.0
    phase_transition_reward: float = 25.0
    unification_hit_reward: float = 50.0
    calabi_yau_reward: float = 100.0

    # Thresholds for detection
    entropy_jump_threshold: float = 0.5
    moduli_convergence_threshold: float = 0.01
    metric_improvement_threshold: float = 0.1

    # Intrinsic motivation
    use_curiosity: bool = True
    curiosity_weight: float = 0.5

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 5

    # History for novelty detection
    history_size: int = 1000


# ============================================================================
# Novelty Detection
# ============================================================================

class NoveltyDetector:
    """
    Detects novel states and patterns in simulation metrics

    Uses state visitation counts, distance metrics, and temporal patterns
    to identify unexplored regions of state space.
    """

    def __init__(self, state_dim: int, history_size: int = 1000):
        self.state_dim = state_dim
        self.history_size = history_size

        # State history
        self.state_history = deque(maxlen=history_size)

        # Visitation counts (discretized state space)
        self.state_bins = 10  # Discretization per dimension
        self.visitation_counts = {}

        # Running statistics
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.update_count = 0

    def update(self, state: np.ndarray):
        """Update detector with new state"""
        self.state_history.append(state)

        # Update running stats
        self.update_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.update_count
        delta2 = state - self.state_mean
        self.state_std = np.sqrt(
            (self.state_std**2 * (self.update_count - 1) + delta * delta2) / self.update_count
        )

        # Update visitation counts
        state_bin = self._discretize_state(state)
        state_key = tuple(state_bin)
        self.visitation_counts[state_key] = self.visitation_counts.get(state_key, 0) + 1

    def compute_novelty(self, state: np.ndarray) -> float:
        """
        Compute novelty score for state

        Higher score = more novel (less visited)

        Returns:
            novelty: Score in [0, inf), higher = more novel
        """
        # Normalize state
        state_normalized = (state - self.state_mean) / (self.state_std + 1e-8)

        # Discretize and check visitation
        state_bin = self._discretize_state(state)
        state_key = tuple(state_bin)
        visit_count = self.visitation_counts.get(state_key, 0)

        # Novelty based on visit count (1/sqrt(count + 1))
        count_novelty = 1.0 / np.sqrt(visit_count + 1.0)

        # Distance to nearest historical state
        if len(self.state_history) > 0:
            distances = [np.linalg.norm(state - s) for s in list(self.state_history)[-100:]]
            min_distance = min(distances)
            distance_novelty = min_distance / (np.sqrt(self.state_dim) + 1e-8)
        else:
            distance_novelty = 1.0

        # Combine
        novelty = 0.5 * count_novelty + 0.5 * distance_novelty

        return novelty

    def _discretize_state(self, state: np.ndarray) -> np.ndarray:
        """Discretize continuous state for counting"""
        # Normalize and clip
        state_normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        state_clipped = np.clip(state_normalized, -3, 3)

        # Discretize to bins
        bins = np.linspace(-3, 3, self.state_bins + 1)
        state_bin = np.digitize(state_clipped, bins) - 1
        state_bin = np.clip(state_bin, 0, self.state_bins - 1)

        return state_bin


# ============================================================================
# iVHL Reward Computer
# ============================================================================

class iVHLRewardComputer:
    """
    Comprehensive reward computation for iVHL discovery

    Combines dense exploration rewards with sparse novelty bonuses
    based on holographic stack metrics.

    Args:
        config: Reward configuration
        state_dim: Dimension of augmented state
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        state_dim: int = 128
    ):
        self.config = config if config is not None else RewardConfig()
        self.state_dim = state_dim

        # Novelty detection
        self.novelty_detector = NoveltyDetector(state_dim, self.config.history_size)

        # Curriculum learning
        self.current_stage = 0
        self.episode_count = 0

        # History for temporal patterns
        self.metric_history = {
            'entropy': deque(maxlen=100),
            'moduli': deque(maxlen=100),
            'ray_cycles': deque(maxlen=100),
            'phase': deque(maxlen=100)
        }

        # Discovery log
        self.discoveries = []

    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        metrics: Dict[str, float],
        done: bool
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            metrics: Simulation metrics dict
            done: Episode termination flag

        Returns:
            reward: Total reward
            reward_components: Breakdown of reward sources
        """
        reward_components = {}

        # ====================================================================
        # Dense Rewards
        # ====================================================================

        # Action diversity (entropy of action distribution)
        action_diversity = self._compute_action_diversity(action)
        reward_components['action_diversity'] = (
            self.config.action_diversity_weight * action_diversity
        )

        # Exploration bonus (novelty)
        self.novelty_detector.update(state)
        novelty = self.novelty_detector.compute_novelty(next_state)
        reward_components['exploration'] = (
            self.config.exploration_bonus_weight * novelty
        )

        # Metric improvement
        metric_improvement = self._compute_metric_improvement(metrics)
        reward_components['metric_improvement'] = (
            self.config.metric_improvement_weight * metric_improvement
        )

        # Smoothness (penalize large action changes)
        smoothness = self._compute_smoothness_bonus(action)
        reward_components['smoothness'] = (
            self.config.smoothness_weight * smoothness
        )

        # ====================================================================
        # Sparse Novelty Rewards
        # ====================================================================

        # Ray cycle detection
        if self._detect_ray_cycle(metrics):
            reward_components['ray_cycle'] = self.config.ray_cycle_reward
            self._log_discovery('ray_cycle', metrics)

        # Entropy jump (Page curve)
        if self._detect_entropy_jump(metrics):
            reward_components['entropy_jump'] = self.config.entropy_jump_reward
            self._log_discovery('entropy_jump', metrics)

        # Moduli convergence (Calabi-Yau stabilization)
        if self._detect_moduli_convergence(metrics):
            reward_components['moduli_convergence'] = self.config.moduli_convergence_reward
            self._log_discovery('moduli_convergence', metrics)

        # GFT phase transition
        if self._detect_phase_transition(metrics):
            reward_components['phase_transition'] = self.config.phase_transition_reward
            self._log_discovery('phase_transition', metrics)

        # Unification consistency hit
        if self._detect_unification_hit(metrics):
            reward_components['unification'] = self.config.unification_hit_reward
            self._log_discovery('unification', metrics)

        # Calabi-Yau full stabilization (jackpot!)
        if self._detect_calabi_yau_stabilization(metrics):
            reward_components['calabi_yau'] = self.config.calabi_yau_reward
            self._log_discovery('calabi_yau', metrics)

        # ====================================================================
        # Intrinsic Motivation (Curiosity)
        # ====================================================================

        if self.config.use_curiosity:
            curiosity_bonus = self._compute_curiosity(state, action, next_state)
            reward_components['curiosity'] = (
                self.config.curiosity_weight * curiosity_bonus
            )

        # ====================================================================
        # Total Reward
        # ====================================================================

        total_reward = sum(reward_components.values())

        # Curriculum shaping
        if self.config.use_curriculum:
            total_reward *= self._get_curriculum_multiplier()

        return total_reward, reward_components

    # ========================================================================
    # Dense Reward Components
    # ========================================================================

    def _compute_action_diversity(self, action: np.ndarray) -> float:
        """Reward for diverse actions (avoid repetition)"""
        # Entropy of action (assuming normalized to [-1, 1])
        action_normalized = (action + 1.0) / 2.0  # Map to [0, 1]
        action_normalized = np.clip(action_normalized, 1e-8, 1 - 1e-8)

        # Negative entropy (higher = more diverse)
        entropy = -np.sum(action_normalized * np.log(action_normalized))

        return entropy / np.log(len(action))  # Normalize

    def _compute_smoothness_bonus(self, action: np.ndarray) -> float:
        """Bonus for smooth actions (penalize jerkiness)"""
        # L2 norm of action (smaller = smoother)
        action_norm = np.linalg.norm(action)

        # Inverse bonus (smaller norm = higher bonus)
        smoothness = 1.0 / (1.0 + action_norm)

        return smoothness

    def _compute_metric_improvement(self, metrics: Dict[str, float]) -> float:
        """Reward for improving key metrics"""
        improvement = 0.0

        # Check each metric for positive trend
        for key, value in metrics.items():
            if key in self.metric_history:
                history = self.metric_history[key]
                if len(history) > 0:
                    recent_mean = np.mean(list(history)[-10:])
                    if value > recent_mean + self.config.metric_improvement_threshold:
                        improvement += 1.0

                history.append(value)

        return improvement

    def _compute_curiosity(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        Curiosity bonus based on prediction error

        Reward for states that are hard to predict (intrinsic motivation)
        """
        # Simple curiosity: magnitude of state change
        state_change = np.linalg.norm(next_state - state)

        # Normalize by state dimension
        curiosity = state_change / np.sqrt(len(state))

        return curiosity

    # ========================================================================
    # Sparse Novelty Detection
    # ========================================================================

    def _detect_ray_cycle(self, metrics: Dict[str, float]) -> bool:
        """Detect new ray traversal cycle"""
        if 'ray_new_cycle' in metrics:
            return metrics['ray_new_cycle'] > 0.5

        # Alternative: Check ray cycle count increase
        if 'ray_cycle_count' in metrics:
            history = self.metric_history['ray_cycles']
            if len(history) > 0:
                return metrics['ray_cycle_count'] > max(history)

        return False

    def _detect_entropy_jump(self, metrics: Dict[str, float]) -> bool:
        """Detect significant entropy increase (Page curve peak)"""
        if 'entropy' not in metrics:
            return False

        entropy = metrics['entropy']
        history = self.metric_history['entropy']

        if len(history) < 5:
            return False

        # Check for jump above recent mean
        recent_mean = np.mean(list(history)[-10:])
        recent_std = np.std(list(history)[-10:])

        jump = (entropy - recent_mean) / (recent_std + 1e-8)

        return jump > self.config.entropy_jump_threshold

    def _detect_moduli_convergence(self, metrics: Dict[str, float]) -> bool:
        """Detect Calabi-Yau moduli convergence"""
        if 'moduli_variance' not in metrics:
            return False

        variance = metrics['moduli_variance']

        # Convergence = low variance
        return variance < self.config.moduli_convergence_threshold

    def _detect_phase_transition(self, metrics: Dict[str, float]) -> bool:
        """Detect GFT phase transition"""
        if 'gft_phase' not in metrics:
            return False

        phase = metrics['gft_phase']
        history = self.metric_history['phase']

        if len(history) < 2:
            return False

        # Detect phase flip
        prev_phase = history[-1]

        return abs(phase - prev_phase) > 0.5

    def _detect_unification_hit(self, metrics: Dict[str, float]) -> bool:
        """Detect cross-stack consistency (unification signal)"""
        if 'consistency_score' not in metrics:
            return False

        # High consistency across entropy/amplitude/RG
        return metrics['consistency_score'] > 0.9

    def _detect_calabi_yau_stabilization(self, metrics: Dict[str, float]) -> bool:
        """Detect full Calabi-Yau manifold stabilization (rare!)"""
        # Multiple conditions must be met
        conditions = [
            metrics.get('moduli_variance', 1.0) < 0.001,
            metrics.get('ray_stable_cycles', 0) > 5,
            metrics.get('entropy_plateau', 0) > 0.9,
            metrics.get('consistency_score', 0) > 0.95
        ]

        return all(conditions)

    # ========================================================================
    # Curriculum Learning
    # ========================================================================

    def _get_curriculum_multiplier(self) -> float:
        """Get reward multiplier based on curriculum stage"""
        # Progressive difficulty: easier tasks → harder tasks
        stage_multipliers = {
            0: 1.5,  # Stage 0: High rewards for basic exploration
            1: 1.2,  # Stage 1: Moderate rewards
            2: 1.0,  # Stage 2: Normal rewards
            3: 0.8,  # Stage 3: Harder (require better performance)
            4: 0.6   # Stage 4: Hardest (sparse rewards only)
        }

        return stage_multipliers.get(self.current_stage, 1.0)

    def advance_curriculum(self):
        """Advance to next curriculum stage"""
        if self.current_stage < self.config.curriculum_stages - 1:
            self.current_stage += 1
            print(f"[Curriculum] Advanced to stage {self.current_stage}")

    def reset_episode(self):
        """Reset for new episode"""
        self.episode_count += 1

        # Auto-advance curriculum every N episodes
        if self.config.use_curriculum and self.episode_count % 100 == 0:
            self.advance_curriculum()

    # ========================================================================
    # Discovery Logging
    # ========================================================================

    def _log_discovery(self, discovery_type: str, metrics: Dict[str, float]):
        """Log significant discovery"""
        discovery = {
            'type': discovery_type,
            'episode': self.episode_count,
            'metrics': metrics.copy()
        }

        self.discoveries.append(discovery)
        print(f"[Discovery] {discovery_type} detected! Metrics: {metrics}")

    def get_discoveries(self) -> List[Dict]:
        """Get all logged discoveries"""
        return self.discoveries.copy()

    def clear_discoveries(self):
        """Clear discovery log"""
        self.discoveries.clear()


# ============================================================================
# Shaped Reward Functions
# ============================================================================

class PotentialBasedShaping:
    """
    Potential-based reward shaping for long-horizon tasks

    Adds shaped rewards F(s, s') = γΦ(s') - Φ(s) that preserve
    optimal policy while providing denser feedback.
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def compute_potential(self, state: np.ndarray, metrics: Dict[str, float]) -> float:
        """
        Compute state potential

        Higher potential = closer to goal
        """
        potential = 0.0

        # Potential from metrics
        if 'entropy' in metrics:
            # Higher entropy (up to a point) = higher potential
            potential += min(metrics['entropy'] / 10.0, 1.0)

        if 'moduli_variance' in metrics:
            # Lower variance = higher potential (convergence)
            potential += 1.0 / (1.0 + metrics['moduli_variance'])

        if 'consistency_score' in metrics:
            # Higher consistency = higher potential
            potential += metrics['consistency_score']

        return potential

    def compute_shaping(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        metrics: Dict[str, float],
        next_metrics: Dict[str, float]
    ) -> float:
        """
        Compute shaped reward: F(s, s') = γΦ(s') - Φ(s)
        """
        phi = self.compute_potential(state, metrics)
        phi_next = self.compute_potential(next_state, next_metrics)

        return self.gamma * phi_next - phi


# ============================================================================
# Main - Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SAC REWARD ENGINEERING TEST")
    print("=" * 70)
    print()

    # Create reward computer
    config = RewardConfig(
        action_diversity_weight=0.1,
        exploration_bonus_weight=0.3,
        entropy_jump_reward=20.0,
        calabi_yau_reward=100.0
    )

    reward_computer = iVHLRewardComputer(config, state_dim=20)

    print("1. Testing Dense Rewards:")
    print("-" * 70)

    # Simulate trajectory
    state = np.random.randn(20).astype(np.float32)
    action = np.random.randn(3).astype(np.float32)
    next_state = state + 0.1 * np.random.randn(20).astype(np.float32)

    metrics = {
        'entropy': 5.0,
        'moduli_variance': 0.5,
        'ray_cycle_count': 3,
        'gft_phase': 0.0
    }

    reward, components = reward_computer.compute_reward(state, action, next_state, metrics, done=False)

    print(f"  State dim: {state.shape}")
    print(f"  Action: {action}")
    print(f"  Total reward: {reward:.6f}")
    print(f"  Components:")
    for key, value in components.items():
        print(f"    {key}: {value:.6f}")
    print()

    # Test novelty detection
    print("2. Testing Novelty Detection:")
    print("-" * 70)

    for i in range(50):
        test_state = np.random.randn(20).astype(np.float32)
        novelty = reward_computer.novelty_detector.compute_novelty(test_state)
        reward_computer.novelty_detector.update(test_state)

        if i % 10 == 0:
            print(f"  Step {i}: Novelty = {novelty:.6f}")

    print()

    # Test sparse rewards
    print("3. Testing Sparse Novelty Rewards:")
    print("-" * 70)

    # Entropy jump
    state2 = np.random.randn(20).astype(np.float32)
    next_state2 = state2 + 0.1 * np.random.randn(20).astype(np.float32)

    # Build history
    for i in range(20):
        metrics_hist = {'entropy': 5.0 + 0.1 * i}
        reward_computer.metric_history['entropy'].append(metrics_hist['entropy'])

    # Trigger jump
    metrics_jump = {
        'entropy': 10.0,  # Large jump!
        'moduli_variance': 0.5
    }

    reward_jump, components_jump = reward_computer.compute_reward(
        state2, action, next_state2, metrics_jump, done=False
    )

    print(f"  Entropy jump test:")
    print(f"    Previous mean: {np.mean(list(reward_computer.metric_history['entropy'])):.2f}")
    print(f"    Current: {metrics_jump['entropy']:.2f}")
    print(f"    Reward: {reward_jump:.6f}")
    if 'entropy_jump' in components_jump:
        print(f"    ✓ Entropy jump detected! Bonus: {components_jump['entropy_jump']:.2f}")
    print()

    # Moduli convergence
    metrics_moduli = {
        'moduli_variance': 0.005,  # Very low variance!
        'entropy': 6.0
    }

    reward_moduli, components_moduli = reward_computer.compute_reward(
        state2, action, next_state2, metrics_moduli, done=False
    )

    print(f"  Moduli convergence test:")
    print(f"    Variance: {metrics_moduli['moduli_variance']:.6f}")
    if 'moduli_convergence' in components_moduli:
        print(f"    ✓ Convergence detected! Bonus: {components_moduli['moduli_convergence']:.2f}")
    print()

    # Calabi-Yau jackpot
    metrics_cy = {
        'moduli_variance': 0.0005,
        'ray_stable_cycles': 6,
        'entropy_plateau': 0.95,
        'consistency_score': 0.98
    }

    reward_cy, components_cy = reward_computer.compute_reward(
        state2, action, next_state2, metrics_cy, done=False
    )

    print(f"  Calabi-Yau stabilization test:")
    if 'calabi_yau' in components_cy:
        print(f"    ✓✓✓ CALABI-YAU DETECTED! Jackpot: {components_cy['calabi_yau']:.2f}")
        print(f"    Total reward: {reward_cy:.2f}")
    print()

    # Test curriculum
    print("4. Testing Curriculum Learning:")
    print("-" * 70)

    print(f"  Current stage: {reward_computer.current_stage}")
    print(f"  Multiplier: {reward_computer._get_curriculum_multiplier():.2f}")

    reward_computer.advance_curriculum()
    print(f"  Advanced to stage: {reward_computer.current_stage}")
    print(f"  New multiplier: {reward_computer._get_curriculum_multiplier():.2f}")
    print()

    # Test potential shaping
    print("5. Testing Potential-Based Shaping:")
    print("-" * 70)

    shaper = PotentialBasedShaping(gamma=0.99)

    shaped_reward = shaper.compute_shaping(state, next_state, metrics, metrics_jump)
    print(f"  Shaped reward: {shaped_reward:.6f}")
    print()

    # Discoveries
    print("6. Discovery Log:")
    print("-" * 70)
    discoveries = reward_computer.get_discoveries()
    print(f"  Total discoveries: {len(discoveries)}")
    for disc in discoveries:
        print(f"    - {disc['type']} (episode {disc['episode']})")
    print()

    print("[OK] SAC reward engineering ready!")
