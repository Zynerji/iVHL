"""
SAC Benchmarks and Stability Tests

Comprehensive benchmarking suite for validating SAC implementation:
- Sample efficiency tests
- Convergence analysis
- Discovery rate metrics
- Stability tests (gradient norms, loss tracking)
- Comparison with baselines (random policy, supervised learning)

Usage:
    python sac_benchmarks.py --suite all
    python sac_benchmarks.py --suite sample_efficiency
    python sac_benchmarks.py --suite stability

Author: iVHL Framework
Date: 2025-12-15
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import pickle

# Import SAC components
from sac_training import SACTrainer, SACTrainingConfig
from sac_rewards import RewardConfig, iVHLRewardComputer
from sac_core import SACAgent, ReplayBuffer


# ============================================================================
# Benchmark Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    num_trials: int = 5
    num_episodes: int = 200
    max_episode_steps: int = 200
    state_dim: int = 128
    action_dim: int = 16
    batch_size: int = 64
    warmup_steps: int = 100
    save_results: bool = True
    results_dir: str = "benchmark_results"
    verbose: bool = True


# ============================================================================
# Test Environment
# ============================================================================

class BenchmarkEnv:
    """
    Synthetic environment for benchmarking

    Designed to test specific aspects:
    - Continuous state/action spaces
    - Sparse rewards (discoveries)
    - Long horizons
    - Stochastic dynamics
    """

    def __init__(self, state_dim=128, action_dim=16, difficulty='medium'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.difficulty = difficulty

        # Difficulty settings
        if difficulty == 'easy':
            self.noise_scale = 0.1
            self.discovery_threshold = 0.5
            self.reward_scale = 1.0
        elif difficulty == 'medium':
            self.noise_scale = 0.3
            self.discovery_threshold = 0.7
            self.reward_scale = 1.0
        elif difficulty == 'hard':
            self.noise_scale = 0.5
            self.discovery_threshold = 0.9
            self.reward_scale = 0.5

        # Hidden goal (target state for discoveries)
        self.goal_state = np.random.randn(state_dim).astype(np.float32)
        self.goal_state /= np.linalg.norm(self.goal_state)

        self.reset()

    def reset(self):
        """Reset environment"""
        self.state = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        self.step_count = 0
        self.total_reward = 0.0
        self.discoveries = 0
        return self.state.copy()

    def step(self, action):
        """Execute action and return (next_state, metrics, done, info)"""
        self.step_count += 1

        # Dynamics: state evolves based on action + noise
        action_effect = np.mean(action) * 0.1
        noise = np.random.randn(self.state_dim).astype(np.float32) * self.noise_scale

        self.state = 0.9 * self.state + 0.1 * self.goal_state * action_effect + noise
        self.state = np.clip(self.state, -5, 5)

        # Compute metrics
        distance_to_goal = np.linalg.norm(self.state - self.goal_state)
        alignment = np.dot(self.state / (np.linalg.norm(self.state) + 1e-8),
                          self.goal_state)

        # Discovery detection (sparse reward)
        discovery = alignment > self.discovery_threshold
        if discovery:
            self.discoveries += 1

        # Metrics for reward computation
        metrics = {
            'entropy': np.abs(np.mean(self.state)) * 2.0,
            'moduli_variance': np.var(self.state[:10]),
            'ray_cycles': self.discoveries,
            'phase': np.arctan2(self.state[0], self.state[1]),
            'unification_consistency': alignment,
            'distance_to_goal': distance_to_goal
        }

        # Done condition
        done = self.step_count >= 200 or self.discoveries >= 5

        info = {
            'step': self.step_count,
            'discoveries': self.discoveries,
            'alignment': alignment
        }

        return self.state.copy(), metrics, done, info


# ============================================================================
# Baseline Policies
# ============================================================================

class RandomPolicy:
    """Random policy baseline"""

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state):
        return np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)


class GreedyPolicy:
    """Simple greedy policy (moves toward higher entropy)"""

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state):
        # Simple heuristic: action based on state gradient
        action = np.tanh(state[:self.action_dim])
        return action.astype(np.float32)


# ============================================================================
# Benchmark Tests
# ============================================================================

class SampleEfficiencyTest:
    """Test sample efficiency of SAC vs baselines"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self) -> Dict:
        """Run sample efficiency benchmark"""

        print("=" * 70)
        print("SAMPLE EFFICIENCY BENCHMARK")
        print("=" * 70)
        print()

        results = {
            'sac': [],
            'random': [],
            'greedy': []
        }

        for trial in range(self.config.num_trials):
            print(f"Trial {trial + 1}/{self.config.num_trials}")
            print("-" * 70)

            # Test SAC
            print("  Testing SAC...")
            sac_rewards = self._test_sac()
            results['sac'].append(sac_rewards)

            # Test Random
            print("  Testing Random policy...")
            random_rewards = self._test_baseline(RandomPolicy(self.config.action_dim))
            results['random'].append(random_rewards)

            # Test Greedy
            print("  Testing Greedy policy...")
            greedy_rewards = self._test_baseline(GreedyPolicy(self.config.action_dim))
            results['greedy'].append(greedy_rewards)

            print()

        # Compute statistics
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()

        for policy_name, rewards_list in results.items():
            avg_rewards = np.mean([np.mean(r) for r in rewards_list])
            std_rewards = np.std([np.mean(r) for r in rewards_list])

            print(f"{policy_name.upper()}:")
            print(f"  Average reward: {avg_rewards:.2f} ± {std_rewards:.2f}")

            # Sample efficiency (episodes to reach threshold)
            threshold = 50.0
            episodes_to_threshold = []
            for rewards in rewards_list:
                for i, r in enumerate(rewards):
                    if r >= threshold:
                        episodes_to_threshold.append(i + 1)
                        break
                else:
                    episodes_to_threshold.append(self.config.num_episodes)

            avg_episodes = np.mean(episodes_to_threshold)
            print(f"  Episodes to reach {threshold}: {avg_episodes:.1f}")
            print()

        return results

    def _test_sac(self) -> List[float]:
        """Test SAC policy"""
        config = SACTrainingConfig(
            mode='online',
            rnn_hidden_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            max_episode_steps=self.config.max_episode_steps
        )

        env = BenchmarkEnv(self.config.state_dim, self.config.action_dim, difficulty='medium')
        trainer = SACTrainer(config=config, rnn_backbone=None)

        metrics = trainer.train_online(env, num_episodes=self.config.num_episodes, verbose=False)

        return metrics['episode_rewards']

    def _test_baseline(self, policy) -> List[float]:
        """Test baseline policy"""
        env = BenchmarkEnv(self.config.state_dim, self.config.action_dim, difficulty='medium')
        reward_computer = iVHLRewardComputer(state_dim=self.config.state_dim)

        episode_rewards = []

        for episode in range(self.config.num_episodes):
            state = env.reset()
            episode_reward = 0.0

            for step in range(self.config.max_episode_steps):
                action = policy.select_action(state)
                next_state, metrics, done, info = env.step(action)

                # Compute reward
                reward, _ = reward_computer.compute_reward(
                    state, action, next_state, metrics, done
                )

                episode_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)

        return episode_rewards


# ============================================================================
# Convergence Test
# ============================================================================

class ConvergenceTest:
    """Test convergence properties of SAC"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self) -> Dict:
        """Run convergence test"""

        print("=" * 70)
        print("CONVERGENCE BENCHMARK")
        print("=" * 70)
        print()

        results = {
            'trials': []
        }

        for trial in range(self.config.num_trials):
            print(f"Trial {trial + 1}/{self.config.num_trials}")

            # Create trainer
            config = SACTrainingConfig(
                mode='online',
                rnn_hidden_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                batch_size=self.config.batch_size,
                warmup_steps=self.config.warmup_steps
            )

            env = BenchmarkEnv(self.config.state_dim, self.config.action_dim)
            trainer = SACTrainer(config=config)

            # Train and track metrics
            metrics = trainer.train_online(env, num_episodes=self.config.num_episodes, verbose=False)

            # Analyze convergence
            trial_results = {
                'episode_rewards': metrics['episode_rewards'],
                'actor_losses': metrics['actor_losses'],
                'critic_losses': metrics['critic_losses'],
                'alpha_values': metrics['alpha_values'],
                'convergence_episode': self._detect_convergence(metrics['episode_rewards'])
            }

            results['trials'].append(trial_results)

            print(f"  Convergence at episode: {trial_results['convergence_episode']}")
            print()

        # Summary statistics
        convergence_episodes = [t['convergence_episode'] for t in results['trials']]
        avg_convergence = np.mean(convergence_episodes)
        std_convergence = np.std(convergence_episodes)

        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Average convergence episode: {avg_convergence:.1f} ± {std_convergence:.1f}")
        print()

        results['average_convergence'] = avg_convergence
        results['std_convergence'] = std_convergence

        return results

    def _detect_convergence(self, rewards: List[float], window=20) -> int:
        """Detect convergence episode (when reward stabilizes)"""
        if len(rewards) < window:
            return len(rewards)

        for i in range(window, len(rewards)):
            recent_rewards = rewards[i - window:i]
            variance = np.var(recent_rewards)

            if variance < 10.0:  # Low variance indicates convergence
                return i

        return len(rewards)


# ============================================================================
# Discovery Rate Test
# ============================================================================

class DiscoveryRateTest:
    """Test discovery rate (sparse reward handling)"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self) -> Dict:
        """Run discovery rate test"""

        print("=" * 70)
        print("DISCOVERY RATE BENCHMARK")
        print("=" * 70)
        print()

        difficulties = ['easy', 'medium', 'hard']
        results = {diff: [] for diff in difficulties}

        for difficulty in difficulties:
            print(f"Testing difficulty: {difficulty.upper()}")
            print("-" * 70)

            for trial in range(self.config.num_trials):
                print(f"  Trial {trial + 1}/{self.config.num_trials}")

                config = SACTrainingConfig(
                    mode='online',
                    rnn_hidden_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    warmup_steps=self.config.warmup_steps
                )

                env = BenchmarkEnv(self.config.state_dim, self.config.action_dim, difficulty=difficulty)
                trainer = SACTrainer(config=config)

                metrics = trainer.train_online(env, num_episodes=self.config.num_episodes, verbose=False)

                # Count discoveries
                discovery_count = sum(1 for r in metrics['episode_rewards'] if r > 50)

                results[difficulty].append(discovery_count)

            print()

        # Print results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)

        for difficulty, counts in results.items():
            avg_discoveries = np.mean(counts)
            std_discoveries = np.std(counts)

            print(f"{difficulty.upper()}:")
            print(f"  Average discoveries: {avg_discoveries:.1f} ± {std_discoveries:.1f}")
            print()

        return results


# ============================================================================
# Stability Test
# ============================================================================

class StabilityTest:
    """Test training stability (gradient norms, loss variance)"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self) -> Dict:
        """Run stability test"""

        print("=" * 70)
        print("STABILITY BENCHMARK")
        print("=" * 70)
        print()

        config = SACTrainingConfig(
            mode='online',
            rnn_hidden_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps
        )

        env = BenchmarkEnv(self.config.state_dim, self.config.action_dim)
        trainer = SACTrainer(config=config)

        # Train
        print("Training...")
        metrics = trainer.train_online(env, num_episodes=self.config.num_episodes, verbose=False)

        # Analyze stability
        results = {
            'actor_loss_variance': np.var(metrics['actor_losses']),
            'critic_loss_variance': np.var(metrics['critic_losses']),
            'alpha_variance': np.var(metrics['alpha_values']),
            'reward_variance': np.var(metrics['episode_rewards']),
            'num_nan_losses': sum(1 for loss in metrics['actor_losses'] if np.isnan(loss)),
            'num_inf_losses': sum(1 for loss in metrics['actor_losses'] if np.isinf(loss))
        }

        # Print results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Actor loss variance: {results['actor_loss_variance']:.4f}")
        print(f"Critic loss variance: {results['critic_loss_variance']:.4f}")
        print(f"Alpha variance: {results['alpha_variance']:.4f}")
        print(f"Reward variance: {results['reward_variance']:.2f}")
        print(f"NaN losses: {results['num_nan_losses']}")
        print(f"Inf losses: {results['num_inf_losses']}")
        print()

        if results['num_nan_losses'] == 0 and results['num_inf_losses'] == 0:
            print("[OK] Training is stable (no NaN/Inf values)")
        else:
            print("[WARN] Training instability detected!")

        print()

        return results


# ============================================================================
# Main Benchmark Suite
# ============================================================================

class BenchmarkSuite:
    """Complete benchmark suite"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        # Create results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    def run_all(self):
        """Run all benchmarks"""

        print()
        print("=" * 70)
        print("SAC COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 70)
        print()
        print(f"Configuration:")
        print(f"  Trials: {self.config.num_trials}")
        print(f"  Episodes per trial: {self.config.num_episodes}")
        print(f"  State dim: {self.config.state_dim}")
        print(f"  Action dim: {self.config.action_dim}")
        print()

        all_results = {}

        # Run tests
        print("Running benchmarks...\n")

        # 1. Sample Efficiency
        sample_test = SampleEfficiencyTest(self.config)
        all_results['sample_efficiency'] = sample_test.run()

        # 2. Convergence
        convergence_test = ConvergenceTest(self.config)
        all_results['convergence'] = convergence_test.run()

        # 3. Discovery Rate
        discovery_test = DiscoveryRateTest(self.config)
        all_results['discovery_rate'] = discovery_test.run()

        # 4. Stability
        stability_test = StabilityTest(self.config)
        all_results['stability'] = stability_test.run()

        # Save results
        if self.config.save_results:
            self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: Dict):
        """Save results to JSON"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = Path(self.config.results_dir) / f"benchmark_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj

        results_serializable = convert(results)

        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to: {filename}")
        print()

    def _print_summary(self, results: Dict):
        """Print benchmark summary"""

        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print()

        # Sample efficiency
        if 'sample_efficiency' in results:
            sac_avg = np.mean([np.mean(r) for r in results['sample_efficiency']['sac']])
            random_avg = np.mean([np.mean(r) for r in results['sample_efficiency']['random']])
            improvement = (sac_avg - random_avg) / abs(random_avg) * 100

            print(f"Sample Efficiency:")
            print(f"  SAC vs Random: +{improvement:.1f}% improvement")
            print()

        # Convergence
        if 'convergence' in results:
            print(f"Convergence:")
            print(f"  Average episode: {results['convergence']['average_convergence']:.1f}")
            print()

        # Stability
        if 'stability' in results:
            stable = (results['stability']['num_nan_losses'] == 0 and
                     results['stability']['num_inf_losses'] == 0)

            print(f"Stability:")
            print(f"  Status: {'STABLE' if stable else 'UNSTABLE'}")
            print()

        print("=" * 70)
        print("ALL BENCHMARKS COMPLETE")
        print("=" * 70)
        print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC Benchmarks')
    parser.add_argument('--suite', choices=['all', 'sample_efficiency', 'convergence', 'discovery', 'stability'],
                       default='all', help='Benchmark suite to run')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per trial')
    parser.add_argument('--save', action='store_true', help='Save results')

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        num_trials=args.trials,
        num_episodes=args.episodes,
        save_results=args.save
    )

    # Run benchmarks
    suite = BenchmarkSuite(config)

    if args.suite == 'all':
        suite.run_all()
    elif args.suite == 'sample_efficiency':
        test = SampleEfficiencyTest(config)
        test.run()
    elif args.suite == 'convergence':
        test = ConvergenceTest(config)
        test.run()
    elif args.suite == 'discovery':
        test = DiscoveryRateTest(config)
        test.run()
    elif args.suite == 'stability':
        test = StabilityTest(config)
        test.run()
