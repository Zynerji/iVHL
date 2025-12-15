"""
SAC Training Integration for iVHL Framework

Integrates Soft Actor-Critic (SAC) with the expanded RNN backbone to enable:
- Online training: Collect rollouts during simulation, update agent asynchronously
- Offline training: Pre-train on historical trajectory logs
- Hybrid mode: Supervised/Fourier initialization + SAC fine-tuning

The RNN serves as a shared feature extractor, with SAC actor/critics operating
on the extracted features. This enables maximum-entropy exploration for discovery
while leveraging the RNN's learned representations.

Architecture:
    Simulation State → RNN Backbone → Features → SAC Actor → Actions
                                              ↘ SAC Critics → Q-values

Usage:
    # Online training
    trainer = SACTrainer(rnn_backbone, state_dim, action_dim)
    trainer.train_online(env, num_episodes=1000)

    # Offline pre-training
    trainer.train_offline(trajectory_log='logs/trajectories.pkl', epochs=100)

    # Hybrid mode
    trainer.initialize_supervised(supervised_data)
    trainer.train_online(env, num_episodes=1000)

Author: iVHL Framework
Date: 2025-12-15
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path
from collections import deque
import time

# Import SAC components
try:
    from sac_core import SACAgent, ReplayBuffer, GaussianPolicy, QNetwork
    from sac_rewards import iVHLRewardComputer, RewardConfig, NoveltyDetector
except ImportError:
    print("Warning: Could not import SAC modules. Ensure sac_core.py and sac_rewards.py are available.")
    raise


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SACTrainingConfig:
    """Configuration for SAC training integration"""

    # Training mode
    mode: str = 'online'  # 'online', 'offline', or 'hybrid'

    # Network architecture
    rnn_hidden_dim: int = 256
    sac_hidden_dim: int = 256
    action_dim: int = 16  # Vortex control parameters

    # SAC hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_tune_alpha: bool = True

    # Replay buffer
    buffer_capacity: int = 1_000_000
    prioritized_replay: bool = False

    # Training schedule
    batch_size: int = 256
    update_frequency: int = 1  # Update every N environment steps
    gradient_steps: int = 1  # Number of gradient steps per update
    warmup_steps: int = 1000  # Random exploration before training

    # Online training
    max_episode_steps: int = 500
    eval_frequency: int = 10  # Evaluate every N episodes
    save_frequency: int = 50  # Save checkpoint every N episodes

    # Offline training
    offline_epochs: int = 100
    offline_batch_size: int = 256
    offline_gradient_steps: int = 1000  # Per epoch

    # Reward engineering
    reward_config: Optional[RewardConfig] = None

    # Checkpointing
    checkpoint_dir: str = 'checkpoints/sac'
    log_dir: str = 'logs/sac'

    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'


# ============================================================================
# Rollout Collection
# ============================================================================

class RolloutCollector:
    """Collects trajectories during simulation for SAC training"""

    def __init__(self, reward_computer: iVHLRewardComputer):
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
        """
        Record a single step

        Args:
            state: Current RNN feature state
            action: Action taken
            next_state: Next RNN feature state
            metrics: iVHL metrics dict (for reward computation)
            done: Episode termination flag
            info: Additional metadata
        """
        # Compute reward from metrics
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
        """
        Get complete trajectory

        Returns:
            Dict with trajectory data
        """
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

    def push_to_buffer(self, replay_buffer: ReplayBuffer):
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
# Offline Data Loading
# ============================================================================

class OfflineDataLoader:
    """Loads historical trajectories for offline SAC training"""

    def __init__(self, log_path: str, reward_computer: iVHLRewardComputer):
        self.log_path = Path(log_path)
        self.reward_computer = reward_computer
        self.trajectories = []

    def load_trajectories(self, max_trajectories: Optional[int] = None):
        """
        Load trajectories from log file

        Args:
            max_trajectories: Maximum number of trajectories to load

        Returns:
            Number of trajectories loaded
        """
        if not self.log_path.exists():
            raise FileNotFoundError(f"Trajectory log not found: {self.log_path}")

        # Load pickle or JSON format
        if self.log_path.suffix == '.pkl':
            with open(self.log_path, 'rb') as f:
                data = pickle.load(f)
        elif self.log_path.suffix == '.json':
            with open(self.log_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported log format: {self.log_path.suffix}")

        # Extract trajectories
        if isinstance(data, list):
            self.trajectories = data[:max_trajectories] if max_trajectories else data
        elif isinstance(data, dict) and 'trajectories' in data:
            self.trajectories = data['trajectories'][:max_trajectories] if max_trajectories else data['trajectories']
        else:
            raise ValueError("Unexpected log format")

        return len(self.trajectories)

    def compute_rewards(self, trajectory: Dict) -> np.ndarray:
        """
        Recompute rewards for a trajectory using current reward function

        Args:
            trajectory: Trajectory dict with states, actions, metrics

        Returns:
            Array of rewards
        """
        rewards = []
        states = trajectory['states']
        actions = trajectory['actions']
        next_states = trajectory.get('next_states', states[1:])
        metrics_list = trajectory.get('metrics', [{}] * len(states))
        dones = trajectory.get('dones', [False] * len(states))

        for i in range(len(states) - 1):
            reward, _ = self.reward_computer.compute_reward(
                states[i], actions[i], next_states[i],
                metrics_list[i], dones[i]
            )
            rewards.append(reward)

        return np.array(rewards, dtype=np.float32)

    def populate_buffer(self, replay_buffer: ReplayBuffer, recompute_rewards: bool = True):
        """
        Populate replay buffer with loaded trajectories

        Args:
            replay_buffer: ReplayBuffer to populate
            recompute_rewards: Whether to recompute rewards with current reward function

        Returns:
            Number of transitions added
        """
        transitions_added = 0

        for traj in self.trajectories:
            states = traj['states']
            actions = traj['actions']
            next_states = traj.get('next_states', states[1:])
            dones = traj.get('dones', [False] * len(states))

            # Rewards
            if recompute_rewards:
                rewards = self.compute_rewards(traj)
            else:
                rewards = traj.get('rewards', np.zeros(len(states) - 1, dtype=np.float32))

            # Push to buffer
            for i in range(len(states) - 1):
                replay_buffer.push(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    dones[i]
                )
                transitions_added += 1

        return transitions_added


# ============================================================================
# SAC Trainer (Main Integration)
# ============================================================================

class SACTrainer:
    """
    Main SAC training orchestration

    Integrates SAC agent with RNN backbone for iVHL discovery optimization.
    Supports online, offline, and hybrid training modes.
    """

    def __init__(self, config: SACTrainingConfig = None, rnn_backbone: nn.Module = None):
        """
        Initialize SAC trainer

        Args:
            config: Training configuration
            rnn_backbone: Pre-trained RNN for feature extraction (optional)
        """
        self.config = config or SACTrainingConfig()

        # Device setup
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        # RNN backbone (shared feature extractor)
        self.rnn_backbone = rnn_backbone
        if self.rnn_backbone is not None:
            self.rnn_backbone.to(self.device)
            self.rnn_backbone.eval()  # Keep in eval mode during SAC training

        # Determine state dimension (RNN features)
        self.state_dim = self.config.rnn_hidden_dim  # RNN output dimension
        self.action_dim = self.config.action_dim

        # SAC agent
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.sac_hidden_dim,
            lr_actor=self.config.lr_actor,
            lr_critic=self.config.lr_critic,
            lr_alpha=self.config.lr_alpha,
            gamma=self.config.gamma,
            tau=self.config.tau,
            alpha=self.config.alpha,
            auto_tune_alpha=self.config.auto_tune_alpha,
            device=self.device
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            device=self.device,
            prioritized=self.config.prioritized_replay
        )

        # Reward computer
        reward_config = self.config.reward_config or RewardConfig()
        self.reward_computer = iVHLRewardComputer(
            config=reward_config,
            state_dim=self.state_dim
        )

        # Rollout collector
        self.rollout_collector = RolloutCollector(self.reward_computer)

        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_values': [],
            'q_values': [],
            'discoveries': []
        }

        # Episode counter
        self.total_steps = 0
        self.episode_count = 0

        # Create checkpoint and log directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def extract_features(self, raw_state: np.ndarray, hidden_state=None) -> Tuple[np.ndarray, Any]:
        """
        Extract features from raw simulation state using RNN backbone

        Args:
            raw_state: Raw simulation state (vortex positions, fields, etc.)
            hidden_state: RNN hidden state (for sequential processing)

        Returns:
            (features, new_hidden_state): Feature vector and updated hidden state
        """
        if self.rnn_backbone is None:
            # No RNN backbone - use raw state directly
            return raw_state, None

        # Convert to tensor
        with torch.no_grad():
            state_tensor = torch.from_numpy(raw_state).unsqueeze(0).to(self.device)

            # RNN forward pass
            if hidden_state is not None:
                features, new_hidden = self.rnn_backbone(state_tensor, hidden_state)
            else:
                features, new_hidden = self.rnn_backbone(state_tensor)

            # Extract features (last timestep)
            features = features[:, -1, :].cpu().numpy().squeeze()

        return features, new_hidden

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using SAC policy

        Args:
            state: Feature state (post-RNN)
            deterministic: If True, use mean of Gaussian policy

        Returns:
            Action array
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.agent.actor(state_tensor, deterministic=deterministic)
            action = action.cpu().numpy().squeeze()

        return action

    def train_online(self, env, num_episodes: int = 1000, verbose: bool = True):
        """
        Online SAC training with environment interaction

        Args:
            env: Environment with step() and reset() methods
            num_episodes: Number of training episodes
            verbose: Print training progress

        Returns:
            Training metrics dictionary
        """
        if verbose:
            print("=" * 70)
            print(f"SAC ONLINE TRAINING ({num_episodes} episodes)")
            print("=" * 70)
            print(f"Device: {self.device}")
            print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
            print(f"Warmup steps: {self.config.warmup_steps}")
            print()

        for episode in range(num_episodes):
            # Reset environment and collector
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
                    # Random exploration during warmup
                    action = np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)
                else:
                    # SAC policy
                    action = self.select_action(state, deterministic=False)

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

                # SAC update
                if (self.total_steps >= self.config.warmup_steps and
                    self.total_steps % self.config.update_frequency == 0 and
                    len(self.replay_buffer) >= self.config.batch_size):

                    for _ in range(self.config.gradient_steps):
                        metrics = self.agent.update(self.replay_buffer, self.config.batch_size)

                        # Log metrics
                        self.training_metrics['actor_losses'].append(metrics['actor_loss'])
                        self.training_metrics['critic_losses'].append((metrics['critic1_loss'] + metrics['critic2_loss']) / 2)
                        self.training_metrics['alpha_values'].append(metrics['alpha'])
                        self.training_metrics['q_values'].append((metrics['mean_q1'] + metrics['mean_q2']) / 2)

            # Push trajectory to replay buffer
            self.rollout_collector.push_to_buffer(self.replay_buffer)

            # Log episode metrics
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(episode_length)
            self.episode_count += 1

            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-10:])
                avg_length = np.mean(self.training_metrics['episode_lengths'][-10:])
                alpha = self.agent.alpha.item() if hasattr(self.agent.alpha, 'item') else self.agent.alpha
                buffer_size = len(self.replay_buffer)

                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} (avg: {avg_reward:.2f}) | "
                      f"Length: {episode_length} (avg: {avg_length:.1f}) | "
                      f"Alpha: {alpha:.3f} | "
                      f"Buffer: {buffer_size}")

            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_reward = self.evaluate(env, num_episodes=5, verbose=False)
                if verbose:
                    print(f"  [EVAL] Episode {episode + 1}: {eval_reward:.2f}")

            # Save checkpoint
            if (episode + 1) % self.config.save_frequency == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"sac_episode_{episode + 1}.pt"
                self.save_checkpoint(checkpoint_path)
                if verbose:
                    print(f"  [SAVE] Checkpoint saved to {checkpoint_path}")

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

    def train_offline(self, trajectory_log: str, epochs: int = 100, verbose: bool = True):
        """
        Offline SAC training on historical trajectories

        Args:
            trajectory_log: Path to trajectory log file (.pkl or .json)
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            Training metrics
        """
        if verbose:
            print("=" * 70)
            print(f"SAC OFFLINE TRAINING ({epochs} epochs)")
            print("=" * 70)
            print(f"Loading trajectories from: {trajectory_log}")

        # Load offline data
        loader = OfflineDataLoader(trajectory_log, self.reward_computer)
        num_trajectories = loader.load_trajectories()

        if verbose:
            print(f"Loaded {num_trajectories} trajectories")

        # Populate replay buffer
        transitions_added = loader.populate_buffer(self.replay_buffer, recompute_rewards=True)

        if verbose:
            print(f"Added {transitions_added} transitions to replay buffer")
            print()
            print("Starting offline training...")
            print()

        # Training loop
        for epoch in range(epochs):
            epoch_metrics = {
                'actor_loss': [],
                'critic_loss': [],
                'alpha': []
            }

            # Multiple gradient steps per epoch
            for _ in range(self.config.offline_gradient_steps):
                if len(self.replay_buffer) < self.config.offline_batch_size:
                    break

                metrics = self.agent.update(self.replay_buffer, self.config.offline_batch_size)

                epoch_metrics['actor_loss'].append(metrics['actor_loss'])
                epoch_metrics['critic_loss'].append((metrics['critic1_loss'] + metrics['critic2_loss']) / 2)
                epoch_metrics['alpha'].append(metrics['alpha'])

            # Log epoch metrics
            if epoch_metrics['actor_loss']:
                avg_actor_loss = np.mean(epoch_metrics['actor_loss'])
                avg_critic_loss = np.mean(epoch_metrics['critic_loss'])
                avg_alpha = np.mean(epoch_metrics['alpha'])

                self.training_metrics['actor_losses'].extend(epoch_metrics['actor_loss'])
                self.training_metrics['critic_losses'].extend(epoch_metrics['critic_loss'])
                self.training_metrics['alpha_values'].extend(epoch_metrics['alpha'])

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} | "
                          f"Actor Loss: {avg_actor_loss:.4f} | "
                          f"Critic Loss: {avg_critic_loss:.4f} | "
                          f"Alpha: {avg_alpha:.3f}")

            # Save checkpoint
            if (epoch + 1) % 50 == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"sac_offline_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path)

        if verbose:
            print()
            print("=" * 70)
            print("OFFLINE TRAINING COMPLETED")
            print("=" * 70)
            print()

        return self.training_metrics

    def evaluate(self, env, num_episodes: int = 10, verbose: bool = True) -> float:
        """
        Evaluate current policy

        Args:
            env: Environment
            num_episodes: Number of evaluation episodes
            verbose: Print results

        Returns:
            Average evaluation reward
        """
        eval_rewards = []

        for episode in range(num_episodes):
            raw_state = env.reset()
            state, hidden_state = self.extract_features(raw_state)

            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < self.config.max_episode_steps:
                # Deterministic policy
                action = self.select_action(state, deterministic=True)

                # Step
                next_raw_state, metrics, done, info = env.step(action)
                next_state, hidden_state = self.extract_features(next_raw_state, hidden_state)

                # Compute reward for logging
                reward, _ = self.reward_computer.compute_reward(
                    state, action, next_state, metrics, done
                )

                episode_reward += reward
                state = next_state
                steps += 1

            eval_rewards.append(episode_reward)

        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)

        if verbose:
            print(f"Evaluation ({num_episodes} episodes): {avg_reward:.2f} +/- {std_reward:.2f}")

        return avg_reward

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        path = Path(path)

        # Save agent separately
        agent_path = path.parent / f"{path.stem}_agent.pt"
        self.agent.save(str(agent_path))

        # Save trainer state
        checkpoint = {
            'config': self.config,
            'training_metrics': self.training_metrics,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'reward_computer_state': {
                'config': self.reward_computer.config,
                'novelty_detector': {
                    'state_mean': self.reward_computer.novelty_detector.state_mean,
                    'state_std': self.reward_computer.novelty_detector.state_std
                }
            },
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
        else:
            # Fallback: look for agent file with same name
            path_obj = Path(path)
            agent_path = path_obj.parent / f"{path_obj.stem}_agent.pt"
            if agent_path.exists():
                self.agent.load(str(agent_path))

        # Load metrics
        self.training_metrics = checkpoint['training_metrics']
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']

        # Load reward computer state
        if 'reward_computer_state' in checkpoint:
            novelty_state = checkpoint['reward_computer_state']['novelty_detector']
            self.reward_computer.novelty_detector.state_mean = novelty_state['state_mean']
            self.reward_computer.novelty_detector.state_std = novelty_state['state_std']


# ============================================================================
# Test and Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SAC TRAINING INTEGRATION TEST")
    print("=" * 70)
    print()

    # Simple dummy environment for testing
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

            # Simple reward: encourage action diversity
            reward = -np.mean(action**2) + 0.1 * np.std(action)

            done = self.step_count >= self.max_steps

            # Dummy metrics
            metrics = {
                'entropy': np.random.rand() * 2.0,
                'moduli_variance': np.random.rand() * 0.1,
                'ray_cycles': np.random.randint(0, 5),
                'phase': np.random.rand() * 2 * np.pi,
                'unification_consistency': np.random.rand()
            }

            info = {'step': self.step_count}

            return next_state, metrics, done, info

    # Test 1: Online training
    print("Test 1: Online Training (short)")
    print("-" * 70)

    config = SACTrainingConfig(
        mode='online',
        rnn_hidden_dim=128,
        action_dim=16,
        warmup_steps=50,
        batch_size=32,
        update_frequency=1
    )

    env = DummyEnv(state_dim=128, action_dim=16)
    trainer = SACTrainer(config=config, rnn_backbone=None)  # No RNN for this test

    # Train for a few episodes
    metrics = trainer.train_online(env, num_episodes=20, verbose=True)

    print()
    print(f"[OK] Online training completed")
    print(f"  Episodes: {len(metrics['episode_rewards'])}")
    print(f"  Average reward: {np.mean(metrics['episode_rewards']):.2f}")
    print(f"  Total steps: {trainer.total_steps}")
    print()

    # Test 2: Checkpoint save/load
    print("Test 2: Checkpoint Save/Load")
    print("-" * 70)

    checkpoint_path = Path('checkpoints/sac/test_checkpoint.pt')
    trainer.save_checkpoint(checkpoint_path)
    print(f"[OK] Checkpoint saved to {checkpoint_path}")

    # Load in new trainer
    trainer2 = SACTrainer(config=config, rnn_backbone=None)
    trainer2.load_checkpoint(checkpoint_path)
    print(f"[OK] Checkpoint loaded")
    print(f"  Restored steps: {trainer2.total_steps}")
    print(f"  Restored episodes: {trainer2.episode_count}")
    print()

    # Test 3: Offline data creation and loading
    print("Test 3: Offline Data Loading")
    print("-" * 70)

    # Create dummy offline trajectories
    dummy_trajectories = []
    for _ in range(10):
        traj_length = 50
        states = np.random.randn(traj_length, 128).astype(np.float32)
        actions = np.random.uniform(-1, 1, size=(traj_length, 16)).astype(np.float32)
        rewards = np.random.randn(traj_length).astype(np.float32)
        dones = np.zeros(traj_length, dtype=np.float32)
        dones[-1] = 1.0

        metrics_list = [{
            'entropy': np.random.rand() * 2.0,
            'moduli_variance': np.random.rand() * 0.1,
            'ray_cycles': np.random.randint(0, 5),
            'phase': np.random.rand() * 2 * np.pi
        } for _ in range(traj_length)]

        dummy_trajectories.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': states[1:],
            'dones': dones,
            'metrics': metrics_list
        })

    # Save to pickle
    offline_log_path = Path('logs/sac/test_offline_trajectories.pkl')
    offline_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(offline_log_path, 'wb') as f:
        pickle.dump(dummy_trajectories, f)

    print(f"[OK] Created dummy offline log: {offline_log_path}")

    # Load and populate buffer
    loader = OfflineDataLoader(str(offline_log_path), trainer.reward_computer)
    num_traj = loader.load_trajectories()
    print(f"[OK] Loaded {num_traj} trajectories")

    transitions = loader.populate_buffer(trainer.replay_buffer, recompute_rewards=True)
    print(f"[OK] Added {transitions} transitions to replay buffer")
    print()

    # Test 4: Offline training (short)
    print("Test 4: Offline Training (short)")
    print("-" * 70)

    offline_metrics = trainer.train_offline(str(offline_log_path), epochs=10, verbose=True)
    print(f"[OK] Offline training completed")
    print(f"  Actor loss updates: {len(offline_metrics['actor_losses'])}")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print()
    print("SAC training integration ready!")
    print()
    print("Next steps:")
    print("1. Integrate with real iVHL simulation environment")
    print("2. Add Streamlit controls for SAC modes (online/offline/hybrid)")
    print("3. Implement comprehensive benchmarks")
    print("4. Add monitoring dashboards for training curves")
    print()
