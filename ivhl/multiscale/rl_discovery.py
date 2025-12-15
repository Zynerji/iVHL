"""
RL Discovery Agent
==================

Uses reinforcement learning to discover stable lattice configurations.

DISCLAIMER: This is a computational optimization tool, not a claim about
discovering physical laws or principles.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class RLDiscoveryConfig:
    """Configuration for RL discovery agent."""
    state_dim: int = 10
    action_dim: int = 5
    hidden_dim: int = 128
    learning_rate: float = 0.001
    episodes: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    """Actor network for policy."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class RLDiscoveryAgent:
    """
    TD3-SAC hybrid agent for discovering stable configurations.
    """
    
    def __init__(self, config: RLDiscoveryConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = Actor(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        ).to(self.device)
        
        self.critic = Critic(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        
        print(f"RLDiscoveryAgent initialized:")
        print(f"  - State dim: {config.state_dim}")
        print(f"  - Action dim: {config.action_dim}")
        print(f"  - Device: {self.device}")
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.tensor(
            state,
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        action = action.cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = action + noise
            action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update networks using TD3-style update."""
        # Convert to tensors
        state_t = torch.tensor(state, device=self.device, dtype=torch.float32)
        action_t = torch.tensor(action, device=self.device, dtype=torch.float32)
        reward_t = torch.tensor(reward, device=self.device, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        
        # Critic update
        with torch.no_grad():
            next_action = self.actor(next_state_t)
            target_q = reward_t + 0.99 * self.critic(next_state_t, next_action) * (1 - done)
        
        current_q = self.critic(state_t, action_t)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        new_action = self.actor(state_t)
        actor_loss = -self.critic(state_t, new_action).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def run_discovery(self, environment) -> Dict:
        """
        Run discovery campaign.
        
        Args:
            environment: Environment with step() method
            
        Returns:
            Dictionary with discovered configurations
        """
        print("Running RL discovery campaign...")
        
        rewards_history = []
        best_reward = -np.inf
        best_config = None
        
        for episode in range(self.config.episodes):
            state = environment.reset()
            episode_reward = 0
            
            for step in range(100):  # Max steps per episode
                action = self.select_action(state, explore=True)
                next_state, reward, done = environment.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards_history.append(episode_reward)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_config = state.copy()
            
            if (episode + 1) % 20 == 0:
                print(f"  Episode {episode+1}/{self.config.episodes}, Reward: {episode_reward:.4f}")
        
        results = {
            'rewards_history': np.array(rewards_history),
            'best_reward': best_reward,
            'best_configuration': best_config,
        }
        
        print(f"Discovery complete!")
        print(f"  - Best reward: {best_reward:.4f}")
        
        return results


if __name__ == "__main__":
    config = RLDiscoveryConfig(
        state_dim=10,
        action_dim=5,
        episodes=100
    )
    
    agent = RLDiscoveryAgent(config)
    print("Agent ready for discovery!")
