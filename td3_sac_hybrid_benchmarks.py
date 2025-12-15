"""
TD3-SAC Hybrid Benchmarks

Comprehensive comparison of TD3-SAC hybrid against pure TD3 and pure SAC on:
- Sample efficiency
- Discovery rate (sparse rewards)
- Stability (gradient norms, loss variance)
- Convergence speed
- Final performance

Validates the hybrid's advantages for iVHL discovery tasks.

Usage:
    python td3_sac_hybrid_benchmarks.py --suite all
    python td3_sac_hybrid_benchmarks.py --suite discovery --trials 5

Author: iVHL Framework
Date: 2025-12-15
"""

import numpy as np
import torch
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# Import algorithms
from td3_sac_hybrid_training import HybridTrainer, HybridTrainingConfig
from td3_sac_hybrid_core import TD3SACConfig


# ============================================================================
# Benchmark Environment
# ============================================================================

class DiscoveryEnv:
    """
    Synthetic environment designed for benchmarking discovery algorithms

    Features:
    - Continuous state/action spaces
    - Sparse rewards (discoveries)
    - Multiple local optima (multi-modal)
    - Stochastic dynamics
    """

    def __init__(self, state_dim=128, action_dim=16, difficulty='medium'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.difficulty = difficulty

        # Difficulty parameters
        if difficulty == 'easy':
            self.noise_scale = 0.1
            self.discovery_threshold = 0.5
            self.num_modes = 3
        elif difficulty == 'medium':
            self.noise_scale = 0.3
            self.discovery_threshold = 0.7
            self.num_modes = 5
        else:  # hard
            self.noise_scale = 0.5
            self.discovery_threshold = 0.9
            self.num_modes = 8

        # Goal states (multiple modes for multi-modal discovery)
        self.goal_states = []
        for _ in range(self.num_modes):
            goal = np.random.randn(state_dim).astype(np.float32)
            goal /= np.linalg.norm(goal)
            self.goal_states.append(goal)

        self.reset()

    def reset(self):
        """Reset environment"""
        self.state = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        self.step_count = 0
        self.discoveries = 0
        self.discovered_modes = set()
        return self.state.copy()

    def step(self, action):
        """Execute action"""
        self.step_count += 1

        # Dynamics
        action_effect = np.mean(action) * 0.1
        noise = np.random.randn(self.state_dim).astype(np.float32) * self.noise_scale

        # Find nearest goal
        best_alignment = -1
        best_goal_idx = 0
        for i, goal in enumerate(self.goal_states):
            alignment = np.dot(
                self.state / (np.linalg.norm(self.state) + 1e-8),
                goal
            )
            if alignment > best_alignment:
                best_alignment = alignment
                best_goal_idx = i

        # Move towards best goal with action influence
        self.state = (0.9 * self.state +
                     0.1 * self.goal_states[best_goal_idx] * action_effect +
                     noise)
        self.state = np.clip(self.state, -5, 5)

        # Check for discovery
        discovery = best_alignment > self.discovery_threshold
        if discovery and best_goal_idx not in self.discovered_modes:
            self.discoveries += 1
            self.discovered_modes.add(best_goal_idx)

        # Metrics
        metrics = {
            'entropy': np.abs(np.mean(self.state)) * 2.0,
            'moduli_variance': np.var(self.state[:10]),
            'ray_cycles': len(self.discovered_modes),
            'phase': np.arctan2(self.state[0], self.state[1]),
            'unification_consistency': best_alignment,
            'alignment': best_alignment,
            'discovered_modes': len(self.discovered_modes)
        }

        done = self.step_count >= 200 or len(self.discovered_modes) >= self.num_modes
        info = {
            'discoveries': self.discoveries,
            'modes_found': len(self.discovered_modes)
        }

        return self.state.copy(), metrics, done, info


# ============================================================================
# Benchmark Tests
# ============================================================================

@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    algorithm: str
    rewards: List[float]
    discoveries: List[int]
    convergence_episode: int
    final_performance: float
    stability_score: float


class HybridBenchmarkSuite:
    """Comprehensive benchmark suite comparing algorithms"""

    def __init__(self, num_trials=3, num_episodes=200, state_dim=128, action_dim=16):
        self.num_trials = num_trials
        self.num_episodes = num_episodes
        self.state_dim = state_dim
        self.action_dim = action_dim

    def run_algorithm(
        self,
        algorithm_name: str,
        mode: str,
        td3_sac_config: TD3SACConfig,
        env_difficulty: str = 'medium',
        verbose: bool = False
    ) -> BenchmarkResult:
        """Run a single algorithm and collect results"""

        all_rewards = []
        all_discoveries = []
        convergence_episodes = []
        stability_scores = []

        for trial in range(self.num_trials):
            if verbose:
                print(f"  Trial {trial + 1}/{self.num_trials}")

            # Create environment
            env = DiscoveryEnv(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                difficulty=env_difficulty
            )

            # Create trainer
            config = HybridTrainingConfig(
                mode=mode,
                rnn_hidden_dim=self.state_dim,
                action_dim=self.action_dim,
                td3_sac_config=td3_sac_config,
                warmup_steps=100,
                batch_size=64
            )

            trainer = HybridTrainer(config=config)

            # Train
            metrics = trainer.train_online(env, num_episodes=self.num_episodes, verbose=False)

            # Collect results
            all_rewards.append(metrics['episode_rewards'])
            discoveries = [info.get('discoveries', 0) for info in metrics.get('discoveries', [])]
            if not discoveries:
                discoveries = [0] * len(metrics['episode_rewards'])
            all_discoveries.append(discoveries)

            # Compute convergence episode
            conv_ep = self._detect_convergence(metrics['episode_rewards'])
            convergence_episodes.append(conv_ep)

            # Compute stability
            if metrics['critic_losses']:
                stability = 1.0 / (np.var(metrics['critic_losses']) + 1.0)
            else:
                stability = 0.0
            stability_scores.append(stability)

        # Aggregate results
        avg_rewards = np.mean(all_rewards, axis=0)
        avg_discoveries = np.mean(all_discoveries, axis=0) if all_discoveries[0] else [0]
        final_perf = np.mean([r[-10:].mean() for r in all_rewards])

        return BenchmarkResult(
            algorithm=algorithm_name,
            rewards=avg_rewards.tolist(),
            discoveries=avg_discoveries.tolist() if hasattr(avg_discoveries, 'tolist') else avg_discoveries,
            convergence_episode=int(np.mean(convergence_episodes)),
            final_performance=float(final_perf),
            stability_score=float(np.mean(stability_scores))
        )

    def _detect_convergence(self, rewards: List[float], window=20) -> int:
        """Detect convergence episode"""
        if len(rewards) < window:
            return len(rewards)

        for i in range(window, len(rewards)):
            recent = rewards[i-window:i]
            if np.var(recent) < 50.0:
                return i

        return len(rewards)

    def benchmark_discovery_rate(self, verbose=True):
        """Benchmark discovery rate across algorithms"""

        if verbose:
            print("=" * 70)
            print("DISCOVERY RATE BENCHMARK")
            print("=" * 70)
            print()

        results = {}

        # Hybrid
        if verbose:
            print("Testing Hybrid TD3-SAC...")
        results['Hybrid'] = self.run_algorithm(
            'Hybrid',
            mode='hybrid',
            td3_sac_config=TD3SACConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                mode='hybrid',
                auto_tune_alpha=True
            ),
            verbose=verbose
        )

        # Pure SAC
        if verbose:
            print("Testing Pure SAC...")
        results['SAC'] = self.run_algorithm(
            'SAC',
            mode='sac',
            td3_sac_config=TD3SACConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                mode='sac',
                auto_tune_alpha=True,
                policy_delay=1  # No delay for pure SAC
            ),
            verbose=verbose
        )

        # Pure TD3
        if verbose:
            print("Testing Pure TD3...")
        results['TD3'] = self.run_algorithm(
            'TD3',
            mode='td3',
            td3_sac_config=TD3SACConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                mode='td3',
                auto_tune_alpha=False,
                alpha=0.0  # No entropy for pure TD3
            ),
            verbose=verbose
        )

        # Print results
        if verbose:
            print()
            print("=" * 70)
            print("RESULTS")
            print("=" * 70)
            print()

            for name, result in results.items():
                print(f"{name}:")
                print(f"  Final Performance: {result.final_performance:.2f}")
                print(f"  Convergence Episode: {result.convergence_episode}")
                print(f"  Stability Score: {result.stability_score:.4f}")
                print()

            # Winner
            best_perf = max(r.final_performance for r in results.values())
            winner = [name for name, r in results.items() if r.final_performance == best_perf][0]
            print(f"WINNER (Final Performance): {winner}")
            print()

        return results

    def benchmark_sample_efficiency(self, verbose=True):
        """Benchmark sample efficiency (convergence speed)"""

        if verbose:
            print("=" * 70)
            print("SAMPLE EFFICIENCY BENCHMARK")
            print("=" * 70)
            print()

        results = self.benchmark_discovery_rate(verbose=False)

        if verbose:
            print("Convergence Speed:")
            print("-" * 70)
            for name, result in results.items():
                print(f"{name}: {result.convergence_episode} episodes")
            print()

            # Winner
            best_conv = min(r.convergence_episode for r in results.values())
            winner = [name for name, r in results.items() if r.convergence_episode == best_conv][0]
            print(f"WINNER (Fastest Convergence): {winner}")
            print()

        return results

    def benchmark_stability(self, verbose=True):
        """Benchmark training stability"""

        if verbose:
            print("=" * 70)
            print("STABILITY BENCHMARK")
            print("=" * 70)
            print()

        results = self.benchmark_discovery_rate(verbose=False)

        if verbose:
            print("Stability Scores:")
            print("-" * 70)
            for name, result in results.items():
                print(f"{name}: {result.stability_score:.4f}")
            print()

            # Winner
            best_stab = max(r.stability_score for r in results.values())
            winner = [name for name, r in results.items() if r.stability_score == best_stab][0]
            print(f"WINNER (Most Stable): {winner}")
            print()

        return results

    def run_all_benchmarks(self, verbose=True):
        """Run complete benchmark suite"""

        print()
        print("=" * 70)
        print("TD3-SAC HYBRID COMPREHENSIVE BENCHMARKS")
        print("=" * 70)
        print(f"Trials: {self.num_trials}")
        print(f"Episodes: {self.num_episodes}")
        print()

        all_results = {}

        # Discovery rate
        print("1. Discovery Rate Test")
        print("-" * 70)
        all_results['discovery'] = self.benchmark_discovery_rate(verbose=True)

        # Sample efficiency
        print("2. Sample Efficiency Test")
        print("-" * 70)
        all_results['efficiency'] = self.benchmark_sample_efficiency(verbose=True)

        # Stability
        print("3. Stability Test")
        print("-" * 70)
        all_results['stability'] = self.benchmark_stability(verbose=True)

        # Overall summary
        print("=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print()

        algorithms = ['Hybrid', 'SAC', 'TD3']
        for alg in algorithms:
            discovery_res = all_results['discovery'][alg]
            efficiency_res = all_results['efficiency'][alg]
            stability_res = all_results['stability'][alg]

            print(f"{alg}:")
            print(f"  Performance: {discovery_res.final_performance:.2f}")
            print(f"  Convergence: {efficiency_res.convergence_episode} episodes")
            print(f"  Stability: {stability_res.stability_score:.4f}")
            print()

        # Save results
        self._save_results(all_results)

        return all_results

    def _save_results(self, results):
        """Save benchmark results to JSON"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = Path('benchmark_results') / f"hybrid_benchmark_{timestamp}.json"
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {}
        for category, alg_results in results.items():
            data[category] = {}
            for alg_name, result in alg_results.items():
                data[category][alg_name] = {
                    'algorithm': result.algorithm,
                    'final_performance': result.final_performance,
                    'convergence_episode': result.convergence_episode,
                    'stability_score': result.stability_score,
                    'rewards': result.rewards[:50]  # First 50 episodes
                }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filename}")
        print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3-SAC Hybrid Benchmarks')
    parser.add_argument('--suite', choices=['all', 'discovery', 'efficiency', 'stability'],
                       default='all', help='Benchmark suite to run')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per trial')

    args = parser.parse_args()

    # Create benchmark suite
    suite = HybridBenchmarkSuite(
        num_trials=args.trials,
        num_episodes=args.episodes
    )

    # Run benchmarks
    if args.suite == 'all':
        suite.run_all_benchmarks()
    elif args.suite == 'discovery':
        suite.benchmark_discovery_rate()
    elif args.suite == 'efficiency':
        suite.benchmark_sample_efficiency()
    elif args.suite == 'stability':
        suite.benchmark_stability()

    print("=" * 70)
    print("BENCHMARKS COMPLETE")
    print("=" * 70)
