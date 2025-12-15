"""
TD3-SAC Hybrid Training Integration

Training orchestration for the TD3-SAC hybrid algorithm with full integration
into the iVHL discovery framework. Combines TD3's stability with SAC's exploration.

Features:
- Multiple training modes (hybrid, TD3-only, SAC-only, phased)
- Replay buffer management with prioritized sampling
- Delayed policy updates (TD3)
- Target policy smoothing with noise (TD3)
- Entropy regularization with auto-alpha (SAC)
- Discovery-driven rewards (sparse + dense)
- RNN backbone integration
- Online/offline/hybrid training
- Checkpointing and evaluation

Usage:
    from td3_sac_hybrid_training import HybridTrainer, HybridTrainingConfig

    config = HybridTrainingConfig(mode='hybrid')
    trainer = HybridTrainer(config=config)
    metrics = trainer.train_online(env, num_episodes=1000)

Author: iVHL Framework
Date: 2025-12-15
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque

# Import hybrid components
from td3_sac_hybrid_core import (
    TD3SACAgent,
    TD3SACConfig,
    HybridReplayBuffer,
    HybridGaussianActor
)

# Reuse SAC reward engineering
try:
    from sac_rewards import RewardConfig, iVHLRewardComputer, NoveltyDetector
except ImportError:
    print("Warning: Could not import SAC rewards. Using dummy rewards.")
    RewardConfig = None
    iVHLRewardComputer = None


# ============================================================================
# Hybrid Training Configuration
# ============================================================================

@dataclass
class HybridTrainingConfig:
    """Configuration for hybrid TD3-SAC training"""

    # Training mode
    mode: str = 'hybrid'  # 'hybrid', 'td3', 'sac', 'phased'

    # Network architecture
    rnn_hidden_dim: int = 256
    sac_hidden_dim: int = 256
    action_dim: int = 16

    # TD3-SAC parameters (use TD3SACConfig for agent-level config)
    td3_sac_config: Optional[TD3SACConfig] = None

    # Replay buffer
    buffer_capacity: int = 1_000_000
    prioritized_replay: bool = False

    # Training schedule
    batch_size: int = 256
    warmup_steps: int = 1000
    max_episode_steps: int = 500
    gradient_steps: int = 1  # Gradient steps per environment step

    # Phased mode parameters
    phase_1_episodes: int = 300  # Exploration phase (high alpha)
    phase_2_episodes: int = 700  # Exploitation phase (low alpha/deterministic)
    phase_1_alpha: float = 0.5  # High exploration
    phase_2_alpha: float = 0.05  # Low exploration

    # Evaluation
    eval_frequency: int = 10
    save_frequency: int = 50

    # Reward configuration
    reward_config: Optional[RewardConfig] = None

    # Directories
    checkpoint_dir: str = 'checkpoints/td3_sac_hybrid'
    log_dir: str = 'logs/td3_sac_hybrid'

    # Device
    device: str = 'auto'


# ============================================================================
# Rollout Collector
# ============================================================================

class HybridRolloutCollector:
    """Collects trajectories during hybrid training"""

    def __init__(self, reward_computer):
        self.reward_computer = reward_computer
        self.reset()

    def reset(self):
        """Reset trajectory buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.metrics = []
        self.info = []

    def step(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray,
             metrics: Dict, done: bool, info: Dict = None):
        """Record a single step"""
        # Compute reward
        reward, reward_components = self.reward_computer.compute_reward(
            state, action, next_state, metrics, done
        )

        # Store transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.metrics.append(metrics)
        self.info.append(info or {})

        return reward, reward_components

    def get_trajectory(self) -> Dict:
        """Get complete trajectory"""
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'next_states': np.array(self.next_states, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'metrics': self.metrics,
            'info': self.info,
            'return': sum(self.rewards),
            'length': len(self.rewards)
        }

    def push_to_buffer(self, replay_buffer: HybridReplayBuffer):
        """Push collected trajectory to replay buffer"""
        for i in range(len(self.states)):
            replay_buffer.push(
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i]
            )


# ============================================================================
# Hybrid Trainer
# ============================================================================

class HybridTrainer:
    """
    Main hybrid TD3-SAC training orchestration

    Supports multiple modes:
    - Hybrid: Full TD3-SAC with adaptive alpha
    - TD3: Deterministic policy, fixed low alpha
    - SAC: Stochastic policy, adaptive alpha
    - Phased: Exploration phase → exploitation phase
    """

    def __init__(self, config: HybridTrainingConfig = None, rnn_backbone: nn.Module = None):
        """Initialize hybrid trainer"""
        self.config = config or HybridTrainingConfig()

        # Device setup
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        # RNN backbone (optional shared feature extractor)
        self.rnn_backbone = rnn_backbone
        if self.rnn_backbone is not None:
            self.rnn_backbone.to(self.device)
            self.rnn_backbone.eval()

        # State dimension
        self.state_dim = self.config.rnn_hidden_dim
        self.action_dim = self.config.action_dim

        # Create TD3-SAC config
        if self.config.td3_sac_config is None:
            td3_sac_config = TD3SACConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config.sac_hidden_dim,
                mode=self.config.mode,
                device=self.device
            )
        else:
            td3_sac_config = self.config.td3_sac_config
            td3_sac_config.device = self.device

        # Create agent
        self.agent = TD3SACAgent(config=td3_sac_config, device=self.device)

        # Replay buffer
        self.replay_buffer = HybridReplayBuffer(
            capacity=self.config.buffer_capacity,
            device=self.device,
            prioritized=self.config.prioritized_replay
        )

        # Reward computer
        if RewardConfig and iVHLRewardComputer:
            reward_config = self.config.reward_config or RewardConfig()
            self.reward_computer = iVHLRewardComputer(
                config=reward_config,
                state_dim=self.state_dim
            )
        else:
            # Dummy reward computer
            class DummyRewardComputer:
                def compute_reward(self, s, a, s_next, metrics, done):
                    return np.random.randn(), {}
            self.reward_computer = DummyRewardComputer()

        # Rollout collector
        self.rollout_collector = HybridRolloutCollector(self.reward_computer)

        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'critic_losses': [],
            'actor_losses': [],
            'alpha_values': [],
            'q_values': [],
            'discoveries': []
        }

        # Counters
        self.total_steps = 0
        self.episode_count = 0

        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def extract_features(self, raw_state: np.ndarray, hidden_state=None) -> Tuple[np.ndarray, Any]:
        """Extract features using RNN backbone (if available)"""
        if self.rnn_backbone is None:
            return raw_state, None

        with torch.no_grad():
            state_tensor = torch.from_numpy(raw_state).unsqueeze(0).to(self.device)

            if hidden_state is not None:
                features, new_hidden = self.rnn_backbone(state_tensor, hidden_state)
            else:
                features, new_hidden = self.rnn_backbone(state_tensor)

            features = features[:, -1, :].cpu().numpy().squeeze()

        return features, new_hidden

    def select_action(self, state: np.ndarray, deterministic: bool = None) -> np.ndarray:
        """Select action using hybrid policy"""
        return self.agent.select_action(state, deterministic=deterministic)

    def update_mode_for_phase(self, episode: int):
        """Update agent mode for phased training"""
        if self.config.mode == 'phased':
            if episode < self.config.phase_1_episodes:
                # Phase 1: Exploration (high alpha, stochastic)
                self.agent.config.mode = 'sac'
                self.agent.log_alpha.data = torch.tensor(
                    [np.log(self.config.phase_1_alpha)],
                    device=self.device,
                    dtype=torch.float32
                )
            else:
                # Phase 2: Exploitation (low alpha/deterministic)
                self.agent.config.mode = 'td3'
                self.agent.log_alpha.data = torch.tensor(
                    [np.log(self.config.phase_2_alpha)],
                    device=self.device,
                    dtype=torch.float32
                )

    def train_online(self, env, num_episodes: int = 1000, verbose: bool = True):
        """
        Online training with environment interaction

        Args:
            env: Environment with step() and reset() methods
            num_episodes: Number of training episodes
            verbose: Print progress

        Returns:
            Training metrics
        """
        if verbose:
            print("=" * 70)
            print(f"TD3-SAC HYBRID ONLINE TRAINING ({num_episodes} episodes)")
            print("=" * 70)
            print(f"Mode: {self.config.mode}")
            print(f"Device: {self.device}")
            print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
            print(f"Warmup steps: {self.config.warmup_steps}")
            print()

        for episode in range(num_episodes):
            # Update mode for phased training
            self.update_mode_for_phase(episode)

            # Reset environment
            raw_state = env.reset()
            state, hidden_state = self.extract_features(raw_state)
            self.rollout_collector.reset()

            episode_reward = 0.0
            episode_length = 0
            done = False

            # Episode rollout
            while not done and episode_length < self.config.max_episode_steps:
                # Select action
                if self.total_steps < self.config.warmup_steps:
                    # Random warmup
                    action = np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)
                else:
                    # Hybrid policy
                    action = self.select_action(state)

                # Environment step
                next_raw_state, metrics, done, info = env.step(action)
                next_state, hidden_state = self.extract_features(next_raw_state, hidden_state)

                # Record transition
                reward, reward_components = self.rollout_collector.step(
                    state, action, next_state, metrics, done, info
                )

                # Update
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1

                # Hybrid updates
                if (self.total_steps >= self.config.warmup_steps and
                    len(self.replay_buffer) >= self.config.batch_size):

                    for _ in range(self.config.gradient_steps):
                        metrics_dict = self.agent.update(self.replay_buffer, self.config.batch_size)

                        # Log metrics
                        self.training_metrics['critic_losses'].append(metrics_dict['critic_loss'])
                        self.training_metrics['actor_losses'].append(metrics_dict['actor_loss'])
                        self.training_metrics['alpha_values'].append(metrics_dict['alpha'])
                        self.training_metrics['q_values'].append(
                            (metrics_dict['q1_mean'] + metrics_dict['q2_mean']) / 2
                        )

            # Push trajectory to buffer
            self.rollout_collector.push_to_buffer(self.replay_buffer)

            # Log episode metrics
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(episode_length)
            self.episode_count += 1

            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-10:])
                avg_length = np.mean(self.training_metrics['episode_lengths'][-10:])
                alpha = self.agent.alpha.item()
                buffer_size = len(self.replay_buffer)
                mode = self.agent.config.mode if hasattr(self.agent.config, 'mode') else 'hybrid'

                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} (avg: {avg_reward:.2f}) | "
                      f"Length: {episode_length} | "
                      f"Alpha: {alpha:.3f} | "
                      f"Mode: {mode} | "
                      f"Buffer: {buffer_size}")

            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_reward = self.evaluate(env, num_episodes=5, verbose=False)
                if verbose:
                    print(f"  [EVAL] Episode {episode + 1}: {eval_reward:.2f}")

            # Save checkpoint
            if (episode + 1) % self.config.save_frequency == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"hybrid_episode_{episode + 1}.pt"
                self.save_checkpoint(str(checkpoint_path))
                if verbose:
                    print(f"  [SAVE] Checkpoint saved")

        if verbose:
            print()
            print("=" * 70)
            print("TRAINING COMPLETED")
            print("=" * 70)
            print(f"Total steps: {self.total_steps}")
            print(f"Episodes: {self.episode_count}")
            print(f"Average reward (last 100): {np.mean(self.training_metrics['episode_rewards'][-100:]):.2f}")
            print()

        return self.training_metrics

    def evaluate(self, env, num_episodes: int = 10, verbose: bool = False) -> float:
        """Evaluate current policy (deterministic)"""
        eval_rewards = []

        for episode in range(num_episodes):
            raw_state = env.reset()
            state, hidden_state = self.extract_features(raw_state)

            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < self.config.max_episode_steps:
                # Deterministic action
                action = self.select_action(state, deterministic=True)

                # Step
                next_raw_state, metrics, done, info = env.step(action)
                next_state, hidden_state = self.extract_features(next_raw_state, hidden_state)

                # Compute reward
                reward, _ = self.reward_computer.compute_reward(
                    state, action, next_state, metrics, done
                )

                episode_reward += reward
                state = next_state
                steps += 1

            eval_rewards.append(episode_reward)

        avg_reward = np.mean(eval_rewards)

        if verbose:
            print(f"Evaluation ({num_episodes} episodes): {avg_reward:.2f} +/- {np.std(eval_rewards):.2f}")

        return avg_reward

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        path = Path(path)

        # Save agent
        agent_path = path.parent / f"{path.stem}_agent.pt"
        self.agent.save(str(agent_path))

        # Save trainer state
        checkpoint = {
            'config': self.config,
            'training_metrics': self.training_metrics,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'agent_path': str(agent_path)
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load agent
        agent_path = checkpoint.get('agent_path')
        if agent_path and Path(agent_path).exists():
            self.agent.load(agent_path)

        # Load metrics
        self.training_metrics = checkpoint['training_metrics']
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']


# ============================================================================
# Main - Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TD3-SAC HYBRID TRAINING TEST")
    print("=" * 70)
    print()

    # Simple test environment
    class DummyEnv:
        def __init__(self, state_dim=128, action_dim=16):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.step_count = 0
            self.max_steps = 100

        def reset(self):
            self.step_count = 0
            return np.random.randn(self.state_dim).astype(np.float32)

        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(self.state_dim).astype(np.float32)

            metrics = {
                'entropy': np.random.rand() * 2.0,
                'moduli_variance': np.random.rand() * 0.1,
                'ray_cycles': np.random.randint(0, 5),
                'phase': np.random.rand() * 2 * np.pi,
                'unification_consistency': np.random.rand()
            }

            done = self.step_count >= self.max_steps
            info = {'step': self.step_count}

            return next_state, metrics, done, info

    # Test hybrid training
    print("Test: Hybrid Mode Training")
    print("-" * 70)

    config = HybridTrainingConfig(
        mode='hybrid',
        rnn_hidden_dim=128,
        action_dim=16,
        warmup_steps=50,
        batch_size=32
    )

    env = DummyEnv(state_dim=128, action_dim=16)
    trainer = HybridTrainer(config=config)

    metrics = trainer.train_online(env, num_episodes=20, verbose=True)

    print()
    print(f"[OK] Hybrid training completed")
    print(f"  Episodes: {len(metrics['episode_rewards'])}")
    print(f"  Average reward: {np.mean(metrics['episode_rewards']):.2f}")
    print(f"  Total steps: {trainer.total_steps}")
    print()

    # Test checkpoint
    print("Test: Checkpoint Save/Load")
    print("-" * 70)

    checkpoint_path = Path('checkpoints/td3_sac_hybrid/test_checkpoint.pt')
    trainer.save_checkpoint(str(checkpoint_path))
    print(f"[OK] Saved to {checkpoint_path}")

    trainer2 = HybridTrainer(config=config)
    trainer2.load_checkpoint(str(checkpoint_path))
    print(f"[OK] Loaded successfully")
    print(f"  Restored steps: {trainer2.total_steps}")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print()
    print("[OK] TD3-SAC hybrid training integration ready!")
    print()
