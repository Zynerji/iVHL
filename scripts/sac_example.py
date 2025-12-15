"""
SAC Training Usage Examples

Demonstrates how to use the SAC training integration with iVHL simulations.

Examples:
1. Online training with dummy environment
2. Offline training from historical logs
3. Hybrid training (offline pre-training + online fine-tuning)
4. Integration with RNN backbone
5. Custom reward engineering

Author: iVHL Framework
Date: 2025-12-15
"""

import numpy as np
import torch
from pathlib import Path
import pickle

from sac_training import SACTrainer, SACTrainingConfig, RolloutCollector, OfflineDataLoader
from sac_rewards import RewardConfig, iVHLRewardComputer
from sac_core import SACAgent


# ============================================================================
# Example 1: Simple Online Training
# ============================================================================

def example_online_training():
    """Example: Online SAC training with a simple environment"""

    print("=" * 70)
    print("EXAMPLE 1: Online SAC Training")
    print("=" * 70)
    print()

    # Define a simple dummy environment for demonstration
    class SimpleEnv:
        def __init__(self, state_dim=128, action_dim=16):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.step_count = 0
            self.max_steps = 200

        def reset(self):
            self.step_count = 0
            return np.random.randn(self.state_dim).astype(np.float32)

        def step(self, action):
            self.step_count += 1

            # Simple dynamics: next state depends on action
            next_state = np.random.randn(self.state_dim).astype(np.float32) + 0.1 * action.mean()

            # Dummy metrics for reward computation
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

    # Create configuration
    config = SACTrainingConfig(
        mode='online',
        rnn_hidden_dim=128,
        sac_hidden_dim=128,
        action_dim=16,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        warmup_steps=100,
        batch_size=64,
        max_episode_steps=200
    )

    # Create environment
    env = SimpleEnv(state_dim=128, action_dim=16)

    # Create trainer
    trainer = SACTrainer(config=config, rnn_backbone=None)

    # Train for a few episodes
    print("Training for 50 episodes...")
    metrics = trainer.train_online(env, num_episodes=50, verbose=True)

    # Print results
    print()
    print("Training complete!")
    print(f"Average reward (last 10): {np.mean(metrics['episode_rewards'][-10:]):.2f}")
    print()

    # Save checkpoint
    checkpoint_path = Path('checkpoints/sac/example1_checkpoint.pt')
    trainer.save_checkpoint(str(checkpoint_path))
    print(f"Checkpoint saved to: {checkpoint_path}")
    print()


# ============================================================================
# Example 2: Offline Training
# ============================================================================

def example_offline_training():
    """Example: Offline SAC training from historical trajectories"""

    print("=" * 70)
    print("EXAMPLE 2: Offline SAC Training")
    print("=" * 70)
    print()

    # Create dummy offline trajectories
    print("Creating dummy trajectory log...")

    dummy_trajectories = []
    for traj_id in range(20):
        traj_length = 100
        states = np.random.randn(traj_length, 128).astype(np.float32)
        actions = np.random.uniform(-1, 1, size=(traj_length, 16)).astype(np.float32)
        rewards = np.random.randn(traj_length).astype(np.float32)
        dones = np.zeros(traj_length, dtype=np.float32)
        dones[-1] = 1.0

        metrics_list = [{
            'entropy': np.random.rand() * 2.0,
            'moduli_variance': np.random.rand() * 0.1,
            'ray_cycles': np.random.randint(0, 5),
            'phase': np.random.rand() * 2 * np.pi,
            'unification_consistency': np.random.rand()
        } for _ in range(traj_length)]

        dummy_trajectories.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': states[1:],
            'dones': dones,
            'metrics': metrics_list
        })

    # Save trajectories
    offline_log_path = Path('logs/sac/example_offline_trajectories.pkl')
    offline_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(offline_log_path, 'wb') as f:
        pickle.dump(dummy_trajectories, f)

    print(f"Created {len(dummy_trajectories)} dummy trajectories")
    print(f"Saved to: {offline_log_path}")
    print()

    # Create configuration for offline training
    config = SACTrainingConfig(
        mode='offline',
        rnn_hidden_dim=128,
        sac_hidden_dim=128,
        action_dim=16,
        offline_epochs=20,
        offline_batch_size=64,
        offline_gradient_steps=100
    )

    # Create trainer
    trainer = SACTrainer(config=config, rnn_backbone=None)

    # Train offline
    print("Training offline for 20 epochs...")
    metrics = trainer.train_offline(str(offline_log_path), epochs=20, verbose=True)

    print()
    print("Offline training complete!")
    print(f"Actor loss: {metrics['actor_losses'][-1]:.4f}")
    print(f"Critic loss: {metrics['critic_losses'][-1]:.4f}")
    print()


# ============================================================================
# Example 3: Hybrid Training
# ============================================================================

def example_hybrid_training():
    """Example: Hybrid training (offline pre-training + online fine-tuning)"""

    print("=" * 70)
    print("EXAMPLE 3: Hybrid Training")
    print("=" * 70)
    print()

    # Simple environment
    class SimpleEnv:
        def __init__(self, state_dim=128, action_dim=16):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.step_count = 0
            self.max_steps = 150

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
            info = {}

            return next_state, metrics, done, info

    # Step 1: Offline pre-training
    print("Step 1: Offline pre-training...")

    # Check if offline log exists from Example 2
    offline_log_path = Path('logs/sac/example_offline_trajectories.pkl')

    if not offline_log_path.exists():
        print("  No offline log found. Run example 2 first, or create dummy data.")
        return

    config = SACTrainingConfig(
        mode='offline',
        rnn_hidden_dim=128,
        sac_hidden_dim=128,
        action_dim=16,
        offline_epochs=10,
        offline_batch_size=64
    )

    trainer = SACTrainer(config=config, rnn_backbone=None)
    trainer.train_offline(str(offline_log_path), epochs=10, verbose=False)

    print("  Offline pre-training complete!")
    print()

    # Step 2: Online fine-tuning
    print("Step 2: Online fine-tuning...")

    # Update config for online mode
    config.mode = 'online'
    config.warmup_steps = 50
    config.max_episode_steps = 150

    env = SimpleEnv()

    # Continue training online (trainer already has pre-trained weights)
    metrics = trainer.train_online(env, num_episodes=30, verbose=False)

    print("  Online fine-tuning complete!")
    print()
    print(f"Final average reward: {np.mean(metrics['episode_rewards'][-10:]):.2f}")
    print()


# ============================================================================
# Example 4: Custom Reward Configuration
# ============================================================================

def example_custom_rewards():
    """Example: Custom reward configuration for specific discovery goals"""

    print("=" * 70)
    print("EXAMPLE 4: Custom Reward Configuration")
    print("=" * 70)
    print()

    # Create custom reward config emphasizing Calabi-Yau discoveries
    custom_reward_config = RewardConfig(
        # Reduce dense reward weights
        action_diversity_weight=0.05,
        exploration_bonus_weight=0.1,
        metric_improvement_weight=0.2,
        smoothness_weight=0.02,

        # Emphasize sparse discovery rewards
        ray_cycle_reward=5.0,
        entropy_jump_reward=10.0,
        moduli_convergence_reward=15.0,
        phase_transition_reward=30.0,
        unification_hit_reward=50.0,
        calabi_yau_reward=200.0,  # High reward for Calabi-Yau!

        # Tight thresholds for quality discoveries
        entropy_jump_threshold=0.8,
        moduli_convergence_threshold=0.005
    )

    print("Custom reward configuration:")
    print(f"  Calabi-Yau reward: {custom_reward_config.calabi_yau_reward}")
    print(f"  Phase transition reward: {custom_reward_config.phase_transition_reward}")
    print(f"  Moduli convergence threshold: {custom_reward_config.moduli_convergence_threshold}")
    print()

    # Create SAC config with custom rewards
    config = SACTrainingConfig(
        mode='online',
        rnn_hidden_dim=128,
        action_dim=16,
        reward_config=custom_reward_config,
        warmup_steps=50
    )

    trainer = SACTrainer(config=config, rnn_backbone=None)

    print("Trainer initialized with custom reward configuration!")
    print()


# ============================================================================
# Example 5: Integration with RNN Backbone
# ============================================================================

def example_rnn_integration():
    """Example: SAC with RNN backbone for feature extraction"""

    print("=" * 70)
    print("EXAMPLE 5: RNN Backbone Integration")
    print("=" * 70)
    print()

    # Create a simple RNN backbone
    class SimpleRNN(torch.nn.Module):
        def __init__(self, input_dim=64, hidden_dim=128):
            super().__init__()
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)

        def forward(self, x, hidden=None):
            # x: (batch, seq_len, input_dim)
            output, hidden = self.rnn(x, hidden)
            return output, hidden

    # Initialize RNN
    rnn_backbone = SimpleRNN(input_dim=64, hidden_dim=128)

    print("Created RNN backbone:")
    print(f"  Input dim: 64")
    print(f"  Hidden dim: 128")
    print()

    # Create SAC trainer with RNN
    config = SACTrainingConfig(
        mode='online',
        rnn_hidden_dim=128,  # Must match RNN output
        action_dim=16
    )

    trainer = SACTrainer(config=config, rnn_backbone=rnn_backbone)

    print("SAC trainer initialized with RNN backbone!")
    print("  RNN extracts features from raw simulation state")
    print("  SAC actor/critics operate on RNN features")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("SAC TRAINING USAGE EXAMPLES")
    print("=" * 70)
    print()

    print("Available examples:")
    print("  1. Online training")
    print("  2. Offline training")
    print("  3. Hybrid training")
    print("  4. Custom reward configuration")
    print("  5. RNN backbone integration")
    print()

    choice = input("Select example (1-5, or 'all' to run all): ").strip()

    if choice == '1':
        example_online_training()
    elif choice == '2':
        example_offline_training()
    elif choice == '3':
        example_hybrid_training()
    elif choice == '4':
        example_custom_rewards()
    elif choice == '5':
        example_rnn_integration()
    elif choice.lower() == 'all':
        example_online_training()
        print("\n" + "=" * 70 + "\n")
        example_offline_training()
        print("\n" + "=" * 70 + "\n")
        example_hybrid_training()
        print("\n" + "=" * 70 + "\n")
        example_custom_rewards()
        print("\n" + "=" * 70 + "\n")
        example_rnn_integration()
    else:
        print("Invalid choice. Please run again and select 1-5 or 'all'.")

    print()
    print("=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print()
