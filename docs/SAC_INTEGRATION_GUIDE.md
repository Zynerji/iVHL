# SAC Integration Guide for iVHL Framework

## Overview

This guide documents the complete **Soft Actor-Critic (SAC)** reinforcement learning integration for the iVHL (Vibrational Helix Lattice) framework. SAC enables adaptive, maximum-entropy exploration to discover rare phenomena like Calabi-Yau stabilizations, phase transitions, and unification hits.

**Author**: iVHL Framework
**Date**: 2025-12-15
**Version**: 1.0

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Modules](#modules)
4. [Training Modes](#training-modes)
5. [Reward Engineering](#reward-engineering)
6. [Streamlit Dashboard](#streamlit-dashboard)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)
9. [Performance Tips](#performance-tips)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Ensure you have PyTorch and required dependencies:

```bash
pip install torch numpy
pip install streamlit plotly pandas  # For dashboard
```

### 30-Second Start

```python
from sac_training import SACTrainer, SACTrainingConfig

# Create configuration
config = SACTrainingConfig(
    mode='online',
    action_dim=16,
    warmup_steps=100
)

# Create trainer
trainer = SACTrainer(config=config)

# Train on your environment
metrics = trainer.train_online(env, num_episodes=1000)
```

### Launch Dashboard

```bash
streamlit run sac_streamlit.py
```

---

## Architecture Overview

### High-Level Flow

```
Simulation State → RNN Backbone (optional) → SAC Actor → Actions
                                           ↘ SAC Critics → Q-values

Rewards ← iVHLRewardComputer ← Metrics (entropy, moduli, phase, etc.)
       ↓
Replay Buffer → SAC Updates (actor, critics, alpha)
```

### Key Components

1. **SAC Agent** (`sac_core.py`)
   - Gaussian policy with reparameterization trick
   - Dual Q-critics with double-Q trick
   - Automatic temperature (alpha) tuning
   - Target networks with Polyak averaging

2. **Reward Computer** (`sac_rewards.py`)
   - Dense rewards: exploration, action diversity, smoothness
   - Sparse rewards: entropy jumps, moduli convergence, Calabi-Yau
   - Novelty detection with state visitation tracking
   - Curriculum learning (5 stages)

3. **Training Orchestration** (`sac_training.py`)
   - Online, offline, and hybrid training modes
   - RNN backbone integration
   - Rollout collection and replay buffer management
   - Checkpointing and evaluation

4. **Streamlit Dashboard** (`sac_streamlit.py`)
   - Live training curves
   - Hyperparameter controls
   - Discovery logging
   - Real-time metrics

---

## Modules

### `sac_core.py` (730 lines)

Core SAC implementation with state-of-the-art components.

**Classes:**
- `GaussianPolicy`: Actor network with tanh-squashed Gaussian policy
- `QNetwork`: Critic network (state-action → Q-value)
- `ReplayBuffer`: Experience storage for off-policy learning
- `SACAgent`: Main agent with actor, dual critics, automatic alpha tuning

**Key Features:**
- Reparameterization trick for backpropagation through stochastic policy
- Double-Q trick to reduce overestimation bias
- Automatic entropy coefficient tuning
- Polyak-averaged target networks (τ = 0.005)

**Example:**
```python
from sac_core import SACAgent

agent = SACAgent(
    state_dim=128,
    action_dim=16,
    hidden_dim=256,
    lr_actor=3e-4,
    lr_critic=3e-4,
    gamma=0.99,
    auto_tune_alpha=True
)

# Update from replay buffer
metrics = agent.update(replay_buffer, batch_size=256)
```

---

### `sac_rewards.py` (590 lines)

Comprehensive reward engineering for iVHL discoveries.

**Classes:**
- `RewardConfig`: Dataclass with all reward weights and thresholds
- `NoveltyDetector`: State visitation tracking and novelty scoring
- `iVHLRewardComputer`: Main reward computation with dense + sparse components
- `PotentialBasedShaping`: Reward shaping for long-horizon tasks

**Reward Types:**

**Dense Rewards (continuous feedback):**
- Action diversity: Encourages varied exploration
- Exploration bonus: Novelty-based intrinsic motivation
- Metric improvement: Rewards for improving entropy, moduli, etc.
- Smoothness: Penalizes jerky actions

**Sparse Rewards (discoveries):**
- Ray cycle detection: +10.0
- Entropy jump: +15.0
- Moduli convergence: +20.0
- Phase transition: +25.0
- Unification hit: +50.0
- **Calabi-Yau stabilization: +100.0** 🎉

**Example:**
```python
from sac_rewards import RewardConfig, iVHLRewardComputer

# Custom reward config
reward_config = RewardConfig(
    calabi_yau_reward=200.0,  # Emphasize Calabi-Yau!
    entropy_jump_threshold=0.8
)

reward_computer = iVHLRewardComputer(config=reward_config)

# Compute reward from metrics
reward, components = reward_computer.compute_reward(
    state, action, next_state, metrics, done
)
```

---

### `sac_training.py` (850 lines)

Training orchestration with online/offline/hybrid modes.

**Classes:**
- `SACTrainingConfig`: Comprehensive configuration dataclass
- `RolloutCollector`: Trajectory collection during simulation
- `OfflineDataLoader`: Historical trajectory loading
- `SACTrainer`: Main training loop with RNN integration

**Training Modes:**

1. **Online**: Real-time interaction with simulation
2. **Offline**: Pre-train on historical logs
3. **Hybrid**: Offline initialization + online fine-tuning

**Example:**
```python
from sac_training import SACTrainer, SACTrainingConfig

config = SACTrainingConfig(
    mode='online',
    rnn_hidden_dim=256,
    action_dim=16,
    warmup_steps=1000,
    max_episode_steps=500
)

trainer = SACTrainer(config=config, rnn_backbone=my_rnn)

# Online training
metrics = trainer.train_online(env, num_episodes=1000)

# Offline training
metrics = trainer.train_offline('logs/trajectories.pkl', epochs=100)
```

---

### `sac_streamlit.py` (850+ lines)

Interactive web dashboard for monitoring and control.

**Features:**
- Mode selection (off/online/offline/hybrid)
- Real-time hyperparameter adjustment
- Live training curves (rewards, losses, Q-values, alpha)
- Discovery alerts and logging
- Episode statistics
- Checkpoint management

**Launch:**
```bash
streamlit run sac_streamlit.py
```

---

## Training Modes

### Online Training

Real-time interaction with simulation, asynchronous updates.

**When to use:**
- Active exploration for new discoveries
- Live adaptation to simulation changes
- Maximum sample efficiency with off-policy learning

**Configuration:**
```python
config = SACTrainingConfig(
    mode='online',
    warmup_steps=1000,        # Random exploration first
    max_episode_steps=500,    # Episode length
    update_frequency=1,       # Update every N steps
    batch_size=256
)
```

**Workflow:**
1. Initialize environment
2. Warmup with random actions (fill replay buffer)
3. SAC policy takes over
4. Asynchronous updates every N steps
5. Periodic evaluation and checkpointing

---

### Offline Training

Pre-train on historical trajectory logs.

**When to use:**
- Leverage existing simulation data
- Pre-training before online fine-tuning
- Safe exploration from historical successes

**Configuration:**
```python
config = SACTrainingConfig(
    mode='offline',
    offline_epochs=100,
    offline_batch_size=256,
    offline_gradient_steps=1000  # Per epoch
)
```

**Workflow:**
1. Load historical trajectories (pickle/JSON)
2. Recompute rewards with current reward function
3. Populate replay buffer
4. Multiple gradient steps per epoch
5. No environment interaction

**Data Format:**
```python
# Trajectory log format (pickle)
trajectories = [
    {
        'states': np.array(...),       # (T, state_dim)
        'actions': np.array(...),      # (T, action_dim)
        'rewards': np.array(...),      # (T,)
        'next_states': np.array(...),  # (T, state_dim)
        'dones': np.array(...),        # (T,)
        'metrics': [dict, dict, ...]   # Per-step metrics
    },
    ...
]
```

---

### Hybrid Training

Combines offline pre-training with online fine-tuning.

**When to use:**
- Best of both worlds: safe initialization + active exploration
- Transfer learning from historical data
- Bootstrapping from supervised/Fourier seeds

**Workflow:**
```python
# Step 1: Offline pre-training
config.mode = 'offline'
trainer = SACTrainer(config=config)
trainer.train_offline('logs/trajectories.pkl', epochs=100)

# Step 2: Online fine-tuning
config.mode = 'online'
trainer.train_online(env, num_episodes=1000)
```

---

## Reward Engineering

### Dense Rewards (Continuous Feedback)

**Action Diversity:**
```python
reward = std(action) * weight  # Encourages exploration
```

**Exploration Bonus:**
```python
novelty = 1 / sqrt(visitation_count + 1)
reward = novelty * weight
```

**Metric Improvement:**
```python
reward = (metric_new - metric_old) * weight
```

### Sparse Rewards (Discoveries)

**Entropy Jump Detection:**
```python
if entropy > mean(recent_entropy) + threshold:
    reward = entropy_jump_reward  # +15.0
```

**Moduli Convergence:**
```python
if variance(moduli) < threshold:
    reward = moduli_convergence_reward  # +20.0
```

**Calabi-Yau Stabilization (Jackpot!):**
```python
if (moduli_variance < 0.001 and
    stable_cycles > 5 and
    entropy_plateau > 0.9 and
    unification_consistency > 0.95):
    reward = calabi_yau_reward  # +100.0 🎉
```

### Curriculum Learning

Five progressive stages:

1. **Exploration**: Random actions, learn dynamics
2. **Basic Control**: Smooth trajectories
3. **Metric Optimization**: Improve entropy/moduli
4. **Discovery Search**: Hunt for rare events
5. **Mastery**: Consistent high-quality discoveries

---

## Streamlit Dashboard

### Launch

```bash
streamlit run sac_streamlit.py
```

### Features

**Sidebar Controls:**
- Training mode selection
- Architecture parameters (RNN/SAC hidden dims)
- Hyperparameters (learning rates, gamma, tau, alpha)
- Reward weights (dense + sparse)
- Checkpoint management

**Main Dashboard:**
- **Status Panel**: Current episode, total steps, best reward
- **Training Curves**: Live plots of rewards, losses, Q-values, alpha
- **Discovery Log**: Recent discoveries with color-coded alerts
- **Statistics**: Average rewards, episode lengths, total discoveries

**Control Buttons:**
- ▶️ Start Training
- ⏸️ Pause
- ▶️ Resume
- ⏹️ Stop

---

## Usage Examples

### Example 1: Basic Online Training

```python
from sac_training import SACTrainer, SACTrainingConfig

config = SACTrainingConfig(
    mode='online',
    action_dim=16,
    warmup_steps=500
)

trainer = SACTrainer(config=config)
metrics = trainer.train_online(env, num_episodes=1000)

print(f"Average reward: {np.mean(metrics['episode_rewards']):.2f}")
```

### Example 2: Custom Rewards for Calabi-Yau Discovery

```python
from sac_rewards import RewardConfig

# Emphasize Calabi-Yau discoveries
custom_rewards = RewardConfig(
    calabi_yau_reward=200.0,
    phase_transition_reward=30.0,
    moduli_convergence_threshold=0.005
)

config = SACTrainingConfig(
    mode='online',
    reward_config=custom_rewards
)

trainer = SACTrainer(config=config)
```

### Example 3: Hybrid Training

```python
# Step 1: Offline pre-training
config = SACTrainingConfig(mode='offline')
trainer = SACTrainer(config=config)
trainer.train_offline('logs/historical.pkl', epochs=50)

# Step 2: Online fine-tuning
config.mode = 'online'
trainer.train_online(env, num_episodes=500)
```

### Example 4: RNN Backbone Integration

```python
import torch.nn as nn

# Define RNN backbone
class VortexRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(64, 256, batch_first=True)

    def forward(self, x, hidden=None):
        return self.gru(x, hidden)

rnn = VortexRNN()

# Create trainer with RNN
trainer = SACTrainer(config=config, rnn_backbone=rnn)
```

---

## API Reference

### SACAgent

```python
SACAgent(
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
    device: torch.device = None
)
```

**Methods:**
- `update(replay_buffer, batch_size)`: Perform one gradient step
- `save(filepath)`: Save agent state
- `load(filepath)`: Load agent state

---

### SACTrainer

```python
SACTrainer(
    config: SACTrainingConfig,
    rnn_backbone: nn.Module = None
)
```

**Methods:**
- `train_online(env, num_episodes, verbose=True)`: Online training
- `train_offline(trajectory_log, epochs, verbose=True)`: Offline training
- `evaluate(env, num_episodes=10)`: Evaluate policy
- `save_checkpoint(path)`: Save complete training state
- `load_checkpoint(path)`: Load training state

---

### iVHLRewardComputer

```python
iVHLRewardComputer(
    config: RewardConfig = None,
    state_dim: int = 128
)
```

**Methods:**
- `compute_reward(state, action, next_state, metrics, done)`: Compute total reward
- `_detect_calabi_yau_stabilization(metrics)`: Check for Calabi-Yau
- `_detect_phase_transition(metrics)`: Check for phase transition
- `_detect_entropy_jump(metrics)`: Check for entropy jump

---

## Performance Tips

### Sample Efficiency

**Use off-policy learning:**
- Replay buffer capacity: 1M transitions
- Batch size: 256 (larger is better with sufficient data)
- Update frequency: 1 (update every step for fastest learning)

**Leverage historical data:**
- Offline pre-training on successful trajectories
- Recompute rewards with improved reward functions

### Exploration vs Exploitation

**Auto-tune alpha:**
```python
config.auto_tune_alpha = True  # Recommended
```

**Manual alpha tuning:**
- High alpha (0.5-1.0): More exploration, diverse behaviors
- Low alpha (0.05-0.2): More exploitation, refined policies

### Computational Efficiency

**Use GPU:**
```python
config.device = 'cuda'  # Auto-detected if available
```

**Batch gradient updates:**
```python
config.gradient_steps = 10  # Multiple updates per environment step
```

**Reduce episode length during exploration:**
```python
config.max_episode_steps = 200  # Shorter episodes for faster iteration
```

### Stability

**Target network update rate:**
```python
config.tau = 0.005  # Slow updates for stability (recommended)
```

**Learning rates:**
```python
config.lr_actor = 3e-4
config.lr_critic = 3e-4  # Match actor LR
config.lr_alpha = 3e-4
```

---

## Troubleshooting

### Issue: Training is slow

**Solutions:**
- Increase `batch_size` (more samples per update)
- Increase `gradient_steps` (multiple updates per step)
- Reduce `max_episode_steps` (faster episodes)
- Use GPU (`config.device = 'cuda'`)

### Issue: Agent doesn't explore

**Solutions:**
- Increase `alpha` (more entropy)
- Enable `auto_tune_alpha = True`
- Increase `exploration_bonus_weight` in reward config
- Check `warmup_steps` is sufficient (>= 1000)

### Issue: No discoveries detected

**Solutions:**
- Lower sparse reward thresholds (e.g., `entropy_jump_threshold`)
- Increase sparse reward weights (make discoveries more valuable)
- Check metric values are in expected ranges
- Verify reward computer is receiving correct metrics

### Issue: Unstable training (diverging losses)

**Solutions:**
- Decrease learning rates (`lr_actor`, `lr_critic`)
- Increase `warmup_steps` (more initial exploration)
- Decrease `tau` (slower target network updates)
- Check for NaN values in rewards/states

### Issue: Checkpoint won't load

**Solutions:**
- Ensure same architecture parameters (hidden dims, action dim)
- Use `weights_only=False` in `torch.load()`
- Check file paths are correct
- Verify agent file exists (`*_agent.pt`)

---

## Advanced Topics

### Custom Reward Functions

Extend `iVHLRewardComputer` for domain-specific rewards:

```python
class CustomRewardComputer(iVHLRewardComputer):
    def compute_reward(self, state, action, next_state, metrics, done):
        base_reward, components = super().compute_reward(
            state, action, next_state, metrics, done
        )

        # Add custom reward component
        if metrics.get('custom_metric') > threshold:
            components['custom'] = 50.0

        total_reward = sum(components.values())
        return total_reward, components
```

### Multi-Agent SAC

Extend for multiple vortex agents:

```python
agents = [SACAgent(...) for _ in range(num_vortices)]

for agent in agents:
    metrics = agent.update(replay_buffer, batch_size)
```

### Prioritized Experience Replay

Enable prioritized sampling:

```python
config.prioritized_replay = True
replay_buffer = ReplayBuffer(capacity=1_000_000, prioritized=True)
```

---

## Files Summary

| File | Lines | Description |
|------|-------|-------------|
| `sac_core.py` | 730 | Core SAC algorithm (actor, critics, replay buffer) |
| `sac_rewards.py` | 590 | Reward engineering (dense + sparse, novelty detection) |
| `sac_training.py` | 850 | Training orchestration (online/offline/hybrid) |
| `sac_streamlit.py` | 850+ | Interactive dashboard with live monitoring |
| `sac_usage_example.py` | 450 | Usage examples (5 scenarios) |

**Total: ~3,500 lines of production-ready SAC integration**

---

## Citation

If you use this SAC integration in your research, please cite:

```
iVHL Framework - Soft Actor-Critic Integration (2025)
Vibrational Helix Lattice with Maximum-Entropy Reinforcement Learning
https://github.com/your-repo/ivhl
```

---

## Support

For questions, issues, or contributions:
- GitHub Issues: [your-repo/issues]
- Documentation: This guide + inline code comments
- Examples: `sac_usage_example.py`

---

**Happy discovering! 🎉**
