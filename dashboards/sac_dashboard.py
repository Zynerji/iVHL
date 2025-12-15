"""
SAC Training Dashboard - Streamlit Interface

Interactive web dashboard for monitoring and controlling Soft Actor-Critic (SAC)
reinforcement learning training in the iVHL framework.

Features:
- Training mode selection (off/online/offline/hybrid)
- Real-time hyperparameter adjustment
- Live training curves (rewards, losses, Q-values, alpha)
- Discovery logging and alerts
- Episode metrics and statistics
- Checkpoint management
- Integration with iVHL simulation

Usage:
    streamlit run sac_streamlit.py

Author: iVHL Framework
Date: 2025-12-15
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import time
import json
from typing import Dict, List, Optional
from collections import deque
import threading

# Import SAC components
try:
    from sac_training import SACTrainer, SACTrainingConfig
    from sac_rewards import RewardConfig, iVHLRewardComputer
    from sac_core import SACAgent
    SAC_AVAILABLE = True
except ImportError as e:
    st.error(f"SAC modules not available: {e}")
    SAC_AVAILABLE = False


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="SAC Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff00;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #00aaff;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #00ff00;
        margin-bottom: 1rem;
    }
    .discovery-alert {
        background-color: #2a1a00;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ffaa00;
        margin-bottom: 0.5rem;
    }
    .status-running {
        color: #00ff00;
        font-weight: bold;
    }
    .status-stopped {
        color: #ff0000;
        font-weight: bold;
    }
    .status-paused {
        color: #ffaa00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state variables"""

    if 'sac_trainer' not in st.session_state:
        st.session_state.sac_trainer = None

    if 'training_active' not in st.session_state:
        st.session_state.training_active = False

    if 'training_paused' not in st.session_state:
        st.session_state.training_paused = False

    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None

    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_values': [],
            'q_values': [],
            'timestamps': [],
            'discoveries': []
        }

    if 'current_episode' not in st.session_state:
        st.session_state.current_episode = 0

    if 'total_steps' not in st.session_state:
        st.session_state.total_steps = 0

    if 'best_reward' not in st.session_state:
        st.session_state.best_reward = -float('inf')

    if 'recent_discoveries' not in st.session_state:
        st.session_state.recent_discoveries = deque(maxlen=10)


# ============================================================================
# Sidebar - Configuration Controls
# ============================================================================

def render_sidebar():
    """Render sidebar with SAC configuration controls"""

    st.sidebar.markdown('<p class="main-header">🧠 SAC Training</p>', unsafe_allow_html=True)

    # ========================================================================
    # Training Mode Selection
    # ========================================================================

    st.sidebar.markdown('<p class="sub-header">Training Mode</p>', unsafe_allow_html=True)

    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Off", "Online", "Offline", "Hybrid"],
        index=0,
        help="Training mode: Online (live interaction), Offline (historical data), Hybrid (both)"
    )

    # ========================================================================
    # Network Architecture
    # ========================================================================

    st.sidebar.markdown('<p class="sub-header">Architecture</p>', unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        rnn_hidden_dim = st.number_input(
            "RNN Hidden Dim",
            min_value=64,
            max_value=1024,
            value=256,
            step=64,
            help="RNN backbone hidden dimension"
        )

    with col2:
        sac_hidden_dim = st.number_input(
            "SAC Hidden Dim",
            min_value=64,
            max_value=1024,
            value=256,
            step=64,
            help="SAC actor/critic hidden dimension"
        )

    action_dim = st.sidebar.number_input(
        "Action Dimension",
        min_value=4,
        max_value=64,
        value=16,
        step=4,
        help="Number of vortex control parameters"
    )

    # ========================================================================
    # SAC Hyperparameters
    # ========================================================================

    st.sidebar.markdown('<p class="sub-header">SAC Hyperparameters</p>', unsafe_allow_html=True)

    with st.sidebar.expander("Learning Rates", expanded=False):
        lr_actor = st.number_input(
            "Actor LR",
            min_value=1e-5,
            max_value=1e-2,
            value=3e-4,
            format="%.5f",
            help="Actor learning rate"
        )

        lr_critic = st.number_input(
            "Critic LR",
            min_value=1e-5,
            max_value=1e-2,
            value=3e-4,
            format="%.5f",
            help="Critic learning rate"
        )

        lr_alpha = st.number_input(
            "Alpha LR",
            min_value=1e-5,
            max_value=1e-2,
            value=3e-4,
            format="%.5f",
            help="Temperature (alpha) learning rate"
        )

    with st.sidebar.expander("RL Parameters", expanded=False):
        gamma = st.slider(
            "Gamma (γ)",
            min_value=0.9,
            max_value=0.999,
            value=0.99,
            step=0.001,
            help="Discount factor"
        )

        tau = st.slider(
            "Tau (τ)",
            min_value=0.001,
            max_value=0.05,
            value=0.005,
            step=0.001,
            help="Target network update rate (Polyak averaging)"
        )

        alpha = st.slider(
            "Alpha (α)",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Initial temperature (entropy regularization)"
        )

        auto_tune_alpha = st.checkbox(
            "Auto-tune Alpha",
            value=True,
            help="Automatically adjust alpha to match target entropy"
        )

    # ========================================================================
    # Training Schedule
    # ========================================================================

    st.sidebar.markdown('<p class="sub-header">Training Schedule</p>', unsafe_allow_html=True)

    if mode == "Online":
        num_episodes = st.sidebar.number_input(
            "Number of Episodes",
            min_value=10,
            max_value=10000,
            value=1000,
            step=10,
            help="Total training episodes"
        )

        max_episode_steps = st.sidebar.number_input(
            "Max Episode Steps",
            min_value=100,
            max_value=5000,
            value=500,
            step=50,
            help="Maximum steps per episode"
        )

        warmup_steps = st.sidebar.number_input(
            "Warmup Steps",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Random exploration steps before training"
        )

    elif mode == "Offline":
        offline_epochs = st.sidebar.number_input(
            "Offline Epochs",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of training epochs on historical data"
        )

        offline_log_path = st.sidebar.text_input(
            "Trajectory Log Path",
            value="logs/trajectories.pkl",
            help="Path to historical trajectory log"
        )

    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=32,
        max_value=1024,
        value=256,
        step=32,
        help="Mini-batch size for gradient updates"
    )

    update_frequency = st.sidebar.number_input(
        "Update Frequency",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Update agent every N steps"
    )

    # ========================================================================
    # Reward Configuration
    # ========================================================================

    st.sidebar.markdown('<p class="sub-header">Reward Weights</p>', unsafe_allow_html=True)

    with st.sidebar.expander("Dense Rewards", expanded=False):
        action_diversity_weight = st.slider(
            "Action Diversity",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        )

        exploration_bonus_weight = st.slider(
            "Exploration Bonus",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01
        )

        metric_improvement_weight = st.slider(
            "Metric Improvement",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01
        )

    with st.sidebar.expander("Sparse Discovery Rewards", expanded=False):
        entropy_jump_reward = st.number_input(
            "Entropy Jump",
            min_value=1.0,
            max_value=100.0,
            value=15.0,
            step=1.0
        )

        moduli_convergence_reward = st.number_input(
            "Moduli Convergence",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0
        )

        phase_transition_reward = st.number_input(
            "Phase Transition",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

        calabi_yau_reward = st.number_input(
            "Calabi-Yau Stabilization",
            min_value=10.0,
            max_value=1000.0,
            value=100.0,
            step=10.0
        )

    # ========================================================================
    # Checkpoint Management
    # ========================================================================

    st.sidebar.markdown('<p class="sub-header">Checkpointing</p>', unsafe_allow_html=True)

    save_frequency = st.sidebar.number_input(
        "Save Every N Episodes",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Checkpoint save frequency"
    )

    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint Directory",
        value="checkpoints/sac",
        help="Directory for saving checkpoints"
    )

    # Load checkpoint
    checkpoint_files = list(Path(checkpoint_dir).glob("*.pt")) if Path(checkpoint_dir).exists() else []

    if checkpoint_files:
        selected_checkpoint = st.sidebar.selectbox(
            "Load Checkpoint",
            ["None"] + [str(f.name) for f in checkpoint_files],
            help="Load a saved checkpoint"
        )

        if selected_checkpoint != "None" and st.sidebar.button("Load"):
            load_checkpoint(Path(checkpoint_dir) / selected_checkpoint)

    # ========================================================================
    # Return Configuration
    # ========================================================================

    config_dict = {
        'mode': mode.lower(),
        'rnn_hidden_dim': rnn_hidden_dim,
        'sac_hidden_dim': sac_hidden_dim,
        'action_dim': action_dim,
        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'lr_alpha': lr_alpha,
        'gamma': gamma,
        'tau': tau,
        'alpha': alpha,
        'auto_tune_alpha': auto_tune_alpha,
        'batch_size': batch_size,
        'update_frequency': update_frequency,
        'save_frequency': save_frequency,
        'checkpoint_dir': checkpoint_dir,
        'reward_config': {
            'action_diversity_weight': action_diversity_weight,
            'exploration_bonus_weight': exploration_bonus_weight,
            'metric_improvement_weight': metric_improvement_weight,
            'entropy_jump_reward': entropy_jump_reward,
            'moduli_convergence_reward': moduli_convergence_reward,
            'phase_transition_reward': phase_transition_reward,
            'calabi_yau_reward': calabi_yau_reward
        }
    }

    if mode == "Online":
        config_dict['num_episodes'] = num_episodes
        config_dict['max_episode_steps'] = max_episode_steps
        config_dict['warmup_steps'] = warmup_steps
    elif mode == "Offline":
        config_dict['offline_epochs'] = offline_epochs
        config_dict['offline_log_path'] = offline_log_path

    return config_dict


# ============================================================================
# Main Dashboard
# ============================================================================

def render_status_panel():
    """Render current training status"""

    st.markdown('<p class="sub-header">Training Status</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.session_state.training_active:
            if st.session_state.training_paused:
                status_text = '<span class="status-paused">PAUSED</span>'
            else:
                status_text = '<span class="status-running">RUNNING</span>'
        else:
            status_text = '<span class="status-stopped">STOPPED</span>'

        st.markdown(f'<div class="metric-box">Status: {status_text}</div>', unsafe_allow_html=True)

    with col2:
        st.metric("Episode", st.session_state.current_episode)

    with col3:
        st.metric("Total Steps", st.session_state.total_steps)

    with col4:
        st.metric("Best Reward", f"{st.session_state.best_reward:.2f}")


def render_training_curves():
    """Render live training curves"""

    st.markdown('<p class="sub-header">Training Curves</p>', unsafe_allow_html=True)

    metrics = st.session_state.metrics_history

    if not metrics['episode_rewards']:
        st.info("No training data yet. Start training to see live curves.")
        return

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Episode Rewards", "Actor/Critic Losses", "Q-Values", "Alpha (Temperature)"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Episode rewards
    if metrics['episode_rewards']:
        episodes = list(range(1, len(metrics['episode_rewards']) + 1))
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=metrics['episode_rewards'],
                mode='lines+markers',
                name='Episode Reward',
                line=dict(color='#00ff00', width=2)
            ),
            row=1, col=1
        )

        # Moving average
        if len(metrics['episode_rewards']) >= 10:
            window = 10
            ma = pd.Series(metrics['episode_rewards']).rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=ma,
                    mode='lines',
                    name=f'{window}-Episode MA',
                    line=dict(color='#ffff00', width=2, dash='dash')
                ),
                row=1, col=1
            )

    # Losses
    if metrics['actor_losses']:
        steps = list(range(1, len(metrics['actor_losses']) + 1))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=metrics['actor_losses'],
                mode='lines',
                name='Actor Loss',
                line=dict(color='#ff00ff', width=1)
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=metrics['critic_losses'],
                mode='lines',
                name='Critic Loss',
                line=dict(color='#00ffff', width=1)
            ),
            row=1, col=2
        )

    # Q-values
    if metrics['q_values']:
        steps = list(range(1, len(metrics['q_values']) + 1))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=metrics['q_values'],
                mode='lines',
                name='Mean Q-Value',
                line=dict(color='#00aaff', width=2)
            ),
            row=2, col=1
        )

    # Alpha
    if metrics['alpha_values']:
        steps = list(range(1, len(metrics['alpha_values']) + 1))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=metrics['alpha_values'],
                mode='lines',
                name='Alpha',
                line=dict(color='#ff6600', width=2)
            ),
            row=2, col=2
        )

    # Layout
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Update Step", row=1, col=2)
    fig.update_xaxes(title_text="Update Step", row=2, col=1)
    fig.update_xaxes(title_text="Update Step", row=2, col=2)

    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Q-Value", row=2, col=1)
    fig.update_yaxes(title_text="Temperature", row=2, col=2)

    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_dark',
        title_text="Live Training Metrics"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_discovery_log():
    """Render recent discoveries"""

    st.markdown('<p class="sub-header">Recent Discoveries</p>', unsafe_allow_html=True)

    if not st.session_state.recent_discoveries:
        st.info("No discoveries yet. Discoveries will appear here when detected.")
        return

    for discovery in reversed(list(st.session_state.recent_discoveries)):
        discovery_type = discovery.get('type', 'unknown')
        metrics = discovery.get('metrics', {})
        episode = discovery.get('episode', 0)

        # Color code by discovery type
        if discovery_type == 'calabi_yau':
            border_color = '#ff0000'
            emoji = '🎉'
        elif discovery_type == 'phase_transition':
            border_color = '#ff6600'
            emoji = '🌀'
        elif discovery_type == 'entropy_jump':
            border_color = '#ffaa00'
            emoji = '⚡'
        elif discovery_type == 'moduli_convergence':
            border_color = '#00ff00'
            emoji = '✓'
        else:
            border_color = '#666666'
            emoji = '📊'

        st.markdown(
            f'<div style="background-color: #1a1a1a; padding: 0.5rem; border-radius: 0.5rem; '
            f'border: 2px solid {border_color}; margin-bottom: 0.5rem;">'
            f'{emoji} <b>Episode {episode}:</b> {discovery_type.replace("_", " ").title()}'
            f'</div>',
            unsafe_allow_html=True
        )


def render_statistics():
    """Render training statistics"""

    st.markdown('<p class="sub-header">Statistics</p>', unsafe_allow_html=True)

    metrics = st.session_state.metrics_history

    if not metrics['episode_rewards']:
        st.info("No statistics available yet.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        recent_rewards = metrics['episode_rewards'][-100:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        st.metric("Avg Reward (last 100)", f"{avg_reward:.2f}")

    with col2:
        recent_lengths = metrics['episode_lengths'][-100:]
        avg_length = np.mean(recent_lengths) if recent_lengths else 0
        st.metric("Avg Episode Length", f"{avg_length:.1f}")

    with col3:
        total_discoveries = len(st.session_state.recent_discoveries)
        st.metric("Total Discoveries", total_discoveries)

    with col4:
        if metrics['alpha_values']:
            current_alpha = metrics['alpha_values'][-1]
            st.metric("Current Alpha", f"{current_alpha:.3f}")


def render_control_buttons(config):
    """Render training control buttons"""

    st.markdown('<p class="sub-header">Controls</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶️ Start Training", disabled=st.session_state.training_active):
            start_training(config)

    with col2:
        if st.button("⏸️ Pause", disabled=not st.session_state.training_active or st.session_state.training_paused):
            st.session_state.training_paused = True

    with col3:
        if st.button("▶️ Resume", disabled=not st.session_state.training_paused):
            st.session_state.training_paused = False

    with col4:
        if st.button("⏹️ Stop", disabled=not st.session_state.training_active):
            stop_training()


# ============================================================================
# Training Control Functions
# ============================================================================

def start_training(config):
    """Initialize and start SAC training"""

    if not SAC_AVAILABLE:
        st.error("SAC modules not available. Cannot start training.")
        return

    try:
        # Create reward config
        reward_config = RewardConfig(
            action_diversity_weight=config['reward_config']['action_diversity_weight'],
            exploration_bonus_weight=config['reward_config']['exploration_bonus_weight'],
            metric_improvement_weight=config['reward_config']['metric_improvement_weight'],
            entropy_jump_reward=config['reward_config']['entropy_jump_reward'],
            moduli_convergence_reward=config['reward_config']['moduli_convergence_reward'],
            phase_transition_reward=config['reward_config']['phase_transition_reward'],
            calabi_yau_reward=config['reward_config']['calabi_yau_reward']
        )

        # Create SAC config
        sac_config = SACTrainingConfig(
            mode=config['mode'],
            rnn_hidden_dim=config['rnn_hidden_dim'],
            sac_hidden_dim=config['sac_hidden_dim'],
            action_dim=config['action_dim'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            lr_alpha=config['lr_alpha'],
            gamma=config['gamma'],
            tau=config['tau'],
            alpha=config['alpha'],
            auto_tune_alpha=config['auto_tune_alpha'],
            batch_size=config['batch_size'],
            update_frequency=config['update_frequency'],
            reward_config=reward_config,
            checkpoint_dir=config['checkpoint_dir']
        )

        if config['mode'] == 'online':
            sac_config.max_episode_steps = config.get('max_episode_steps', 500)
            sac_config.warmup_steps = config.get('warmup_steps', 1000)

        # Initialize trainer
        st.session_state.sac_trainer = SACTrainer(config=sac_config, rnn_backbone=None)
        st.session_state.training_active = True
        st.session_state.training_paused = False

        st.success("Training started!")

    except Exception as e:
        st.error(f"Error starting training: {e}")


def stop_training():
    """Stop SAC training"""
    st.session_state.training_active = False
    st.session_state.training_paused = False
    st.info("Training stopped.")


def load_checkpoint(checkpoint_path):
    """Load a training checkpoint"""
    try:
        if st.session_state.sac_trainer is None:
            st.warning("No trainer initialized. Initialize trainer first.")
            return

        st.session_state.sac_trainer.load_checkpoint(str(checkpoint_path))

        # Update metrics
        metrics = st.session_state.sac_trainer.training_metrics
        st.session_state.metrics_history = metrics
        st.session_state.current_episode = st.session_state.sac_trainer.episode_count
        st.session_state.total_steps = st.session_state.sac_trainer.total_steps

        if metrics['episode_rewards']:
            st.session_state.best_reward = max(metrics['episode_rewards'])

        st.success(f"Checkpoint loaded: {checkpoint_path.name}")

    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application"""

    # Initialize session state
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🧠 SAC Training Dashboard</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #888; margin-bottom: 2rem;">
    Soft Actor-Critic Reinforcement Learning for iVHL Discovery Optimization
    </div>
    """, unsafe_allow_html=True)

    # Render sidebar configuration
    config = render_sidebar()

    # Main content area
    if config['mode'] == 'off':
        st.info("SAC training is OFF. Select a training mode (Online/Offline/Hybrid) in the sidebar to begin.")

        st.markdown("### About SAC Training")
        st.markdown("""
        **Soft Actor-Critic (SAC)** is a state-of-the-art off-policy reinforcement learning algorithm
        for continuous control. In the iVHL framework, SAC enables:

        - 🎯 **Maximum-entropy exploration**: Diverse vortex trajectory discovery
        - 🔄 **Off-policy learning**: Sample-efficient learning from historical data
        - 🎁 **Sparse reward handling**: Optimized for rare discoveries (Calabi-Yau, phase transitions)
        - 📊 **Automatic tuning**: Self-adjusting exploration/exploitation balance

        **Training Modes:**
        - **Online**: Real-time interaction with simulation, asynchronous updates
        - **Offline**: Pre-train on historical trajectory logs
        - **Hybrid**: Initialize with supervised learning, fine-tune with online SAC
        """)

    else:
        # Render main dashboard
        render_status_panel()
        render_control_buttons(config)

        # Training curves
        render_training_curves()

        # Two columns for discoveries and statistics
        col1, col2 = st.columns([2, 1])

        with col1:
            render_discovery_log()

        with col2:
            render_statistics()

        # Auto-refresh
        if st.session_state.training_active and not st.session_state.training_paused:
            time.sleep(1)
            st.rerun()


if __name__ == "__main__":
    if not SAC_AVAILABLE:
        st.error("SAC modules are not available. Please ensure sac_training.py, sac_core.py, and sac_rewards.py are installed.")
    else:
        main()
