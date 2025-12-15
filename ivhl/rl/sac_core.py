"""
Soft Actor-Critic (SAC) Reinforcement Learning Core

Implements state-of-the-art off-policy actor-critic with maximum entropy regularization
for sample-efficient adaptive discovery in the iVHL holographic framework.

SAC combines:
- Stochastic policy with entropy maximization (exploration)
- Dual Q-critics with double-Q trick (overestimation bias reduction)
- Automatic temperature tuning (adaptive exploration/exploitation balance)
- Off-policy learning with replay buffer (sample efficiency)

Perfect for:
- Sparse/delayed rewards (rare discoveries like Calabi-Yau stabilizations)
- Continuous action spaces (vortex trajectories)
- Long-horizon tasks (multi-step discovery campaigns)
- High-dimensional state spaces (augmented simulation metrics)

Key References:
- Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications"
- Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"

Author: iVHL Framework
Date: 2025-12-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque
import random

# Import utilities
try:
    from utils.device import get_device, get_compiled
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    def get_compiled(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Gaussian Policy (Actor)
# ============================================================================

class GaussianPolicy(nn.Module):
    """
    Stochastic Gaussian policy with reparameterization trick

    Outputs continuous actions sampled from diagonal Gaussian distribution.
    Uses tanh squashing to bound actions to [-1, 1] range.

    Args:
        input_dim: Feature dimension from RNN backbone
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
        log_std_min: Minimum log std (stability)
        log_std_max: Maximum log std (prevents collapse)
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate heads for mean and log_std
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Initialize weights (small last layer for stability)
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.bias.data.uniform_(-3e-3, 3e-3)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            state: Input state (batch, input_dim)
            deterministic: If True, return mean (no sampling)
            with_logprob: If True, return log probability

        Returns:
            action: Sampled action (batch, action_dim)
            log_prob: Log probability (batch, 1) if with_logprob else None
        """
        # Forward through shared layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Get mean and log_std
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            # Deterministic policy (for evaluation)
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Stochastic policy with reparameterization trick
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick

            # Apply tanh squashing
            action = torch.tanh(x_t)

            if with_logprob:
                # Compute log probability with change of variables
                # log π(a|s) = log π(x|s) - Σ log(1 - tanh²(x))
                log_prob = normal.log_prob(x_t)
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                log_prob = None

        return action, log_prob

    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action given state

        Used for updating alpha (temperature)

        Args:
            state: State (batch, state_dim)
            action: Action (batch, action_dim)

        Returns:
            log_prob: (batch, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # Inverse tanh to get pre-squashed action
        action_clamped = torch.clamp(action, -0.999, 0.999)
        x_t = torch.atanh(action_clamped)

        # Log probability
        normal = Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob


# ============================================================================
# Q-Network (Critic)
# ============================================================================

class QNetwork(nn.Module):
    """
    Q-function (critic) network

    Estimates Q(s, a) - expected return from state-action pair.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer size
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        # State-action concatenation network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Initialize
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: State (batch, state_dim)
            action: Action (batch, action_dim)

        Returns:
            Q-value: (batch, 1)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        return q


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning

    Stores transitions (s, a, r, s', done) and samples mini-batches.

    Args:
        capacity: Maximum buffer size
        device: Torch device
        prioritized: Use prioritized experience replay (optional)
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        device: Optional[torch.device] = None,
        prioritized: bool = False
    ):
        self.capacity = capacity
        self.device = device if device is not None else get_device()
        self.prioritized = prioritized

        # Storage
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity) if prioritized else None

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ):
        """
        Add transition to buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            priority: Priority for prioritized replay (optional)
        """
        self.buffer.append((state, action, reward, next_state, done))

        if self.prioritized:
            if priority is None:
                priority = 1.0  # Default priority
            self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample mini-batch

        Args:
            batch_size: Batch size

        Returns:
            states, actions, rewards, next_states, dones (all torch.Tensor)
        """
        if self.prioritized:
            # Prioritized sampling
            probs = np.array(self.priorities, dtype=np.float32)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            batch = [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            batch = random.sample(self.buffer, batch_size)

        # Unpack
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors (ensure float32)
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay"""
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        if self.prioritized:
            self.priorities.clear()


# ============================================================================
# SAC Agent
# ============================================================================

class SACAgent:
    """
    Soft Actor-Critic Agent

    Combines actor, dual critics, automatic temperature tuning,
    and target networks for stable off-policy learning.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        hidden_dim: Hidden layer size for networks
        lr_actor: Actor learning rate
        lr_critic: Critic learning rate
        lr_alpha: Alpha learning rate
        gamma: Discount factor
        tau: Polyak averaging coefficient for target networks
        alpha: Initial temperature (if auto_tune=False)
        auto_tune_alpha: Automatic temperature tuning
        target_entropy: Target entropy (default: -action_dim)
        device: Torch device
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        target_entropy: Optional[float] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device if device is not None else get_device()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_tune_alpha = auto_tune_alpha

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)

        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Target networks (for stability)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Copy parameters to targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Freeze target networks
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # Temperature (alpha)
        if auto_tune_alpha:
            # Learnable log_alpha (ensure float32)
            self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=self.device, dtype=torch.float32)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        else:
            self.log_alpha = torch.tensor([np.log(alpha)], device=self.device, dtype=torch.float32)
            self.alpha_optimizer = None
            self.target_entropy = None

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature"""
        return self.log_alpha.exp()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action from policy

        Args:
            state: Current state
            deterministic: Use mean (no sampling)

        Returns:
            action: Action in [-1, 1]^action_dim
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor(state_tensor, deterministic=deterministic, with_logprob=False)

        return action.cpu().numpy()[0]

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Update actor, critics, and alpha

        Performs one gradient step for each network.

        Args:
            replay_buffer: Replay buffer to sample from
            batch_size: Mini-batch size

        Returns:
            Dictionary of losses and metrics
        """
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # ====================================================================
        # Update Critics
        # ====================================================================

        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs = self.actor(next_states, deterministic=False, with_logprob=True)

            # Target Q-values (double-Q trick)
            q1_target_next = self.critic1_target(next_states, next_actions)
            q2_target_next = self.critic2_target(next_states, next_actions)
            q_target_next = torch.min(q1_target_next, q2_target_next)

            # Soft Bellman backup
            # Q_target = r + γ (1 - done) * (Q_next - α * log π)
            q_target = rewards + (1 - dones) * self.gamma * (q_target_next - self.alpha * next_log_probs)

        # Current Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        # Critic losses (MSE)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ====================================================================
        # Update Actor
        # ====================================================================

        # Sample new actions from current policy
        new_actions, log_probs = self.actor(states, deterministic=False, with_logprob=True)

        # Q-values for new actions
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss: maximize Q - α * log π (minimize negative)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ====================================================================
        # Update Alpha (Temperature)
        # ====================================================================

        alpha_loss = torch.tensor(0.0)
        if self.auto_tune_alpha:
            # Alpha loss: match entropy to target
            # α_loss = -α * (log π + target_entropy)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ====================================================================
        # Update Target Networks (Polyak averaging)
        # ====================================================================

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        # ====================================================================
        # Return metrics
        # ====================================================================

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_tune_alpha else 0.0,
            'alpha': self.alpha.item(),
            'mean_q1': q1.mean().item(),
            'mean_q2': q2.mean().item(),
            'mean_log_prob': log_probs.mean().item(),
            'mean_reward': rewards.mean().item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Polyak averaging: θ_target ← τ θ + (1-τ) θ_target

        Args:
            source: Source network
            target: Target network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.alpha_optimizer else None
        }, filepath)

    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        if self.alpha_optimizer and checkpoint['alpha_optimizer']:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


# ============================================================================
# Main - Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SAC CORE MODULE TEST")
    print("=" * 70)
    print()

    device = get_device()
    print(f"Device: {device}")
    print()

    # Test configuration
    state_dim = 20
    action_dim = 3
    batch_size = 64

    print("1. Testing Gaussian Policy:")
    print("-" * 70)

    policy = GaussianPolicy(state_dim, action_dim).to(device)
    test_state = torch.randn(batch_size, state_dim, device=device)

    # Stochastic action
    action, log_prob = policy(test_state, deterministic=False, with_logprob=True)
    print(f"  State shape: {test_state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Mean log prob: {log_prob.mean():.3f}")
    print()

    # Deterministic action
    det_action, _ = policy(test_state, deterministic=True, with_logprob=False)
    print(f"  Deterministic action range: [{det_action.min():.3f}, {det_action.max():.3f}]")
    print()

    print("2. Testing Q-Network:")
    print("-" * 70)

    critic = QNetwork(state_dim, action_dim).to(device)
    q_value = critic(test_state, action)
    print(f"  Q-value shape: {q_value.shape}")
    print(f"  Mean Q: {q_value.mean():.3f}")
    print()

    print("3. Testing Replay Buffer:")
    print("-" * 70)

    buffer = ReplayBuffer(capacity=1000, device=device)

    # Add transitions
    for _ in range(100):
        state = np.random.randn(state_dim).astype(np.float32)
        action_np = np.random.randn(action_dim).astype(np.float32)
        reward = float(np.random.randn())
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = float(np.random.rand() < 0.1)

        buffer.push(state, action_np, reward, next_state, done)

    print(f"  Buffer size: {len(buffer)}")

    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"  Sampled batch:")
    print(f"    States: {states.shape}")
    print(f"    Actions: {actions.shape}")
    print(f"    Rewards: {rewards.shape}")
    print(f"    Next states: {next_states.shape}")
    print(f"    Dones: {dones.shape}")
    print()

    print("4. Testing SAC Agent:")
    print("-" * 70)

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        auto_tune_alpha=True,
        device=device
    )

    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Initial alpha: {agent.alpha.item():.3f}")
    print()

    # Test action selection
    test_state_np = np.random.randn(state_dim).astype(np.float32)
    action_np = agent.select_action(test_state_np, deterministic=False)
    print(f"  Selected action: {action_np}")
    print(f"  Action shape: {action_np.shape}")
    print()

    # Test update
    print("  Performing SAC update...")
    metrics = agent.update(buffer, batch_size=32)

    print(f"  Update metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}")
    print()

    # Test save/load
    print("5. Testing Save/Load:")
    print("-" * 70)

    save_path = 'test_sac_agent.pt'
    agent.save(save_path)
    print(f"  Saved agent to {save_path}")

    # Create new agent with same architecture and load
    agent2 = SACAgent(state_dim, action_dim, hidden_dim=128, device=device)
    agent2.load(save_path)
    print(f"  Loaded agent from {save_path}")

    # Verify same action
    action_np2 = agent2.select_action(test_state_np, deterministic=True)
    print(f"  Action difference: {np.abs(action_np - action_np2).max():.9f}")
    print()

    import os
    os.remove(save_path)

    print("[OK] SAC core module ready!")
