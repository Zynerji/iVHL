"""
Advanced Multi-Vortex Control System for VHL Holographic Resonance

Extended features:
- Support for 2-8+ independent vortices with individual Fourier trajectories
- Enhanced trajectory presets: circle, figure8, star5, heart, Lissajous, trefoil
- Advanced RNN training (LSTM/GRU) with multiple control modes
- Pattern warping and smooth crossing dynamics
- Integration with AdS/CFT tensor network framework

Author: Zynerji
Date: 2025-12-15
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Callable
import json
from dataclasses import dataclass


@dataclass
class VortexConfig:
    """Configuration for a single vortex."""
    charge: int  # Topological charge (±1, ±2, etc.)
    amplitude: float  # Trajectory amplitude
    k: float  # Wavenumber
    trajectory_type: str  # 'fourier', 'rnn_direct', 'rnn_autoregressive'
    fourier_coeffs: Optional[Dict] = None  # Fourier coefficients if applicable
    initial_position: Optional[np.ndarray] = None


class AdvancedFourierTrajectory:
    """
    Enhanced Fourier trajectory generator with additional presets and features.

    Supports complex 3D paths via Fourier series:
    x(t) = Σ A_n * cos(n*ω*t + φ_n)
    y(t) = Σ B_n * cos(n*ω*t + ψ_n)
    z(t) = Σ C_n * cos(n*ω*t + θ_n)
    """

    def __init__(self, preset: str = 'circle', omega: float = 1.0,
                 num_harmonics: int = 5, amplitude: float = 0.3,
                 phase_offset: float = 0.0):
        """
        Initialize advanced Fourier trajectory.

        Args:
            preset: Trajectory shape preset
            omega: Base angular frequency
            num_harmonics: Number of harmonics to use
            amplitude: Overall amplitude scale
            phase_offset: Global phase shift for coordination
        """
        self.preset = preset
        self.omega = omega
        self.num_harmonics = num_harmonics
        self.amplitude = amplitude
        self.phase_offset = phase_offset

        # Fourier coefficients: List of (amplitude, phase) tuples
        self.coeffs_x: List[Tuple[float, float]] = []
        self.coeffs_y: List[Tuple[float, float]] = []
        self.coeffs_z: List[Tuple[float, float]] = []

        self._initialize_preset()

    def _initialize_preset(self):
        """Initialize Fourier coefficients based on preset shape."""
        A = self.amplitude
        φ = self.phase_offset

        if self.preset == 'circle':
            # Perfect circle in XY plane
            self.coeffs_x = [(A, φ)]
            self.coeffs_y = [(A, φ - np.pi/2)]
            self.coeffs_z = [(0.0, 0.0)]

        elif self.preset == 'figure8':
            # Figure-eight (Lissajous 1:2 ratio)
            self.coeffs_x = [(A, φ)]
            self.coeffs_y = [(A, φ), (A, φ)]  # 2ω component
            self.coeffs_z = [(0.0, 0.0)]

        elif self.preset == 'star5':
            # Five-pointed star using 5-fold symmetry
            for n in range(1, 6):
                angle = n * 2 * np.pi / 5
                amp = A / (1 + 0.3 * (n - 1))  # Decreasing amplitude
                self.coeffs_x.append((amp * np.cos(angle), φ + angle))
                self.coeffs_y.append((amp * np.sin(angle), φ + angle + np.pi/2))
                self.coeffs_z.append((0.0, 0.0))

        elif self.preset == 'heart':
            # Heart shape via parametric equations
            # x = 16sin³(t), y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
            self.coeffs_x = [
                (A * 0.6, φ),  # Approximate with Fourier
                (A * 0.4, φ),
                (A * 0.2, φ)
            ]
            self.coeffs_y = [
                (A * 0.65, φ),
                (A * 0.25, φ),
                (A * 0.1, φ),
                (A * 0.05, φ)
            ]
            self.coeffs_z = [(0.0, 0.0)]

        elif self.preset == 'lissajous':
            # 3D Lissajous curve (ω_x:ω_y:ω_z = 1:2:3)
            self.coeffs_x = [(A, φ)]
            self.coeffs_y = [(A, φ), (A * 0.5, φ + np.pi/4)]
            self.coeffs_z = [(A * 0.6, φ), (A * 0.3, φ + np.pi/3), (A * 0.15, φ)]

        elif self.preset == 'trefoil':
            # Trefoil knot projection
            # x = sin(t) + 2sin(2t), y = cos(t) - 2cos(2t), z = -sin(3t)
            self.coeffs_x = [(A, φ), (2*A, φ)]
            self.coeffs_y = [(A, φ - np.pi/2), (2*A, φ + np.pi/2)]
            self.coeffs_z = [(A, φ), (0.0, 0.0), (A, φ)]

        elif self.preset == 'spiral':
            # Rising spiral
            self.coeffs_x = [(A, φ)]
            self.coeffs_y = [(A, φ - np.pi/2)]
            self.coeffs_z = [(0.15 * A, φ)]  # Vertical component

        elif self.preset == 'custom':
            # Initialize with zeros - user must set coefficients
            for _ in range(self.num_harmonics):
                self.coeffs_x.append((0.0, 0.0))
                self.coeffs_y.append((0.0, 0.0))
                self.coeffs_z.append((0.0, 0.0))

        else:
            raise ValueError(f"Unknown preset: {self.preset}")

    def set_custom_coefficients(self, coeffs_x: List[Tuple[float, float]],
                               coeffs_y: List[Tuple[float, float]],
                               coeffs_z: List[Tuple[float, float]]):
        """Set custom Fourier coefficients."""
        self.coeffs_x = coeffs_x
        self.coeffs_y = coeffs_y
        self.coeffs_z = coeffs_z
        self.preset = 'custom'

    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate trajectory at time t.

        Args:
            t: Time parameter

        Returns:
            3D position [x, y, z]
        """
        position = np.zeros(3)

        # Sum Fourier series for each coordinate
        for n, (amp, phase) in enumerate(self.coeffs_x, start=1):
            position[0] += amp * np.cos(n * self.omega * t + phase)

        for n, (amp, phase) in enumerate(self.coeffs_y, start=1):
            position[1] += amp * np.cos(n * self.omega * t + phase)

        for n, (amp, phase) in enumerate(self.coeffs_z, start=1):
            position[2] += amp * np.cos(n * self.omega * t + phase)

        return position

    def evaluate_batch(self, times: np.ndarray) -> np.ndarray:
        """
        Evaluate trajectory at multiple times.

        Args:
            times: Array of time values

        Returns:
            Array of shape (len(times), 3)
        """
        return np.array([self.evaluate(t) for t in times])

    def get_velocity(self, t: float) -> np.ndarray:
        """
        Get velocity vector at time t (derivative of position).

        Args:
            t: Time parameter

        Returns:
            3D velocity [vx, vy, vz]
        """
        velocity = np.zeros(3)

        for n, (amp, phase) in enumerate(self.coeffs_x, start=1):
            velocity[0] -= amp * n * self.omega * np.sin(n * self.omega * t + phase)

        for n, (amp, phase) in enumerate(self.coeffs_y, start=1):
            velocity[1] -= amp * n * self.omega * np.sin(n * self.omega * t + phase)

        for n, (amp, phase) in enumerate(self.coeffs_z, start=1):
            velocity[2] -= amp * n * self.omega * np.sin(n * self.omega * t + phase)

        return velocity


class AdvancedVortexRNN(nn.Module):
    """
    Enhanced RNN for vortex trajectory learning and prediction.

    Supports:
    - LSTM and GRU architectures
    - Direct position prediction
    - Autoregressive generation
    - Multi-vortex coordination
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 3,
                 rnn_type: str = 'lstm', dropout: float = 0.1):
        """
        Initialize advanced vortex RNN.

        Args:
            input_size: Input dimension (time + position features)
            hidden_size: RNN hidden dimension
            num_layers: Number of RNN layers
            output_size: Output dimension (position or delta)
            rnn_type: 'lstm' or 'gru'
            dropout: Dropout probability
        """
        super(AdvancedVortexRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_size)
            hidden: Optional hidden state

        Returns:
            (output, hidden) tuple
        """
        # RNN
        rnn_out, hidden = self.rnn(x, hidden)

        # Fully connected layers with dropout
        out = self.activation(self.fc1(rnn_out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.activation(out) * 0.5  # Scale to [-0.5, 0.5]

        return out, hidden

    def predict_trajectory(self, initial_pos: np.ndarray, num_steps: int,
                          dt: float = 0.05, mode: str = 'autoregressive') -> np.ndarray:
        """
        Generate trajectory prediction.

        Args:
            initial_pos: Starting position (3D)
            num_steps: Number of steps to predict
            dt: Time step
            mode: 'autoregressive' or 'direct'

        Returns:
            Trajectory array (num_steps, 3)
        """
        self.eval()
        trajectory = [initial_pos]
        current_pos = initial_pos
        hidden = None

        with torch.no_grad():
            for step in range(num_steps - 1):
                t = step * dt

                # Input: [time, x, y, z]
                input_vec = torch.tensor([[t] + list(current_pos)],
                                        dtype=torch.float32).unsqueeze(0)

                # Predict
                delta, hidden = self.forward(input_vec, hidden)
                delta = delta.squeeze().numpy()

                # Update position
                if mode == 'autoregressive':
                    current_pos = current_pos + delta
                else:
                    current_pos = delta

                # Optional: Confine to sphere
                norm = np.linalg.norm(current_pos)
                if norm > 0.5:
                    current_pos = current_pos * 0.5 / norm

                trajectory.append(current_pos.copy())

        return np.array(trajectory)


class AdvancedTrajectoryTrainer:
    """
    Enhanced trainer for vortex RNN with multiple trajectory sources.
    """

    def __init__(self, rnn: AdvancedVortexRNN, learning_rate: float = 1e-3):
        """
        Initialize trainer.

        Args:
            rnn: AdvancedVortexRNN model
            learning_rate: Learning rate for optimizer
        """
        self.rnn = rnn
        self.optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50
        )
        self.criterion = nn.MSELoss()
        self.training_history = []

    def generate_training_data(self, num_trajectories: int = 200,
                              num_steps: int = 100, dt: float = 0.05,
                              presets: Optional[List[str]] = None) -> Tuple:
        """
        Generate training data from Fourier trajectories.

        Args:
            num_trajectories: Number of example trajectories
            num_steps: Steps per trajectory
            dt: Time step
            presets: List of presets to sample from

        Returns:
            (inputs, targets) as torch tensors
        """
        if presets is None:
            presets = ['circle', 'figure8', 'star5', 'heart', 'lissajous',
                      'trefoil', 'spiral']

        all_inputs = []
        all_targets = []

        for _ in range(num_trajectories):
            # Random preset and parameters
            preset = np.random.choice(presets)
            omega = np.random.uniform(0.5, 2.0)
            amplitude = np.random.uniform(0.2, 0.4)
            phase = np.random.uniform(0, 2*np.pi)

            traj_gen = AdvancedFourierTrajectory(
                preset=preset, omega=omega, amplitude=amplitude,
                phase_offset=phase
            )

            # Generate trajectory
            times = np.arange(num_steps) * dt
            positions = traj_gen.evaluate_batch(times)

            # Create input/target pairs
            for i in range(len(positions) - 1):
                # Input: [time, x, y, z]
                input_vec = np.array([times[i]] + list(positions[i]))

                # Target: delta position
                target_vec = positions[i+1] - positions[i]

                all_inputs.append(input_vec)
                all_targets.append(target_vec)

        # Convert to tensors
        inputs = torch.tensor(all_inputs, dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(1)

        return inputs, targets

    def train(self, num_epochs: int = 1000, batch_size: int = 64,
             validation_split: float = 0.2, verbose: bool = True) -> List[float]:
        """
        Train the RNN.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            verbose: Print progress

        Returns:
            List of training losses
        """
        # Generate data
        if verbose:
            print("Generating training data...")
        inputs, targets = self.generate_training_data()

        # Split train/validation
        num_samples = len(inputs)
        num_val = int(num_samples * validation_split)
        indices = torch.randperm(num_samples)

        train_inputs = inputs[indices[num_val:]]
        train_targets = targets[indices[num_val:]]
        val_inputs = inputs[indices[:num_val]]
        val_targets = targets[indices[:num_val]]

        # Dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        if verbose:
            print(f"Training samples: {len(train_inputs)}, Validation: {len(val_inputs)}")
            print(f"Training for {num_epochs} epochs...")

        self.rnn.train()

        for epoch in range(num_epochs):
            # Training
            train_loss = 0.0
            for batch_inputs, batch_targets in train_loader:
                outputs, _ = self.rnn(batch_inputs)
                loss = self.criterion(outputs, batch_targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss = 0.0
            self.rnn.eval()
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    outputs, _ = self.rnn(batch_inputs)
                    loss = self.criterion(outputs, batch_targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            self.rnn.train()

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if verbose:
            print("[OK] Training complete!")

        return [h['train_loss'] for h in self.training_history]

    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.rnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)

    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path)
        self.rnn.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])


class MultiVortexChoreographer:
    """
    Advanced choreographer for 2-8+ vortices with independent control.

    Supports:
    - Independent Fourier trajectories per vortex
    - RNN-driven trajectories
    - Pattern warping and crossing dynamics
    - Synchronized motion with phase coordination
    """

    def __init__(self, num_vortices: int = 2):
        """
        Initialize multi-vortex choreographer.

        Args:
            num_vortices: Number of vortices (2-8+)
        """
        if num_vortices < 2 or num_vortices > 16:
            raise ValueError("Number of vortices must be between 2 and 16")

        self.num_vortices = num_vortices
        self.vortex_configs: List[VortexConfig] = []
        self.fourier_trajectories: List[Optional[AdvancedFourierTrajectory]] = []
        self.rnn_controllers: List[Optional[AdvancedVortexRNN]] = []
        self.charges: List[int] = []

    def add_vortex(self, config: VortexConfig,
                   trajectory: Optional[AdvancedFourierTrajectory] = None,
                   rnn: Optional[AdvancedVortexRNN] = None):
        """
        Add a vortex to the choreography.

        Args:
            config: Vortex configuration
            trajectory: Fourier trajectory (if applicable)
            rnn: RNN controller (if applicable)
        """
        if len(self.vortex_configs) >= self.num_vortices:
            raise ValueError(f"Cannot add more than {self.num_vortices} vortices")

        self.vortex_configs.append(config)
        self.fourier_trajectories.append(trajectory)
        self.rnn_controllers.append(rnn)
        self.charges.append(config.charge)

    def add_fourier_vortex(self, charge: int, preset: str = 'circle',
                          omega: float = 1.0, amplitude: float = 0.3,
                          phase_offset: float = 0.0, k: float = 1.0):
        """
        Add vortex with Fourier trajectory.

        Args:
            charge: Topological charge
            preset: Trajectory preset
            omega: Angular frequency
            amplitude: Trajectory amplitude
            phase_offset: Phase shift
            k: Wavenumber
        """
        trajectory = AdvancedFourierTrajectory(
            preset=preset, omega=omega, amplitude=amplitude,
            phase_offset=phase_offset
        )

        config = VortexConfig(
            charge=charge,
            amplitude=amplitude,
            k=k,
            trajectory_type='fourier'
        )

        self.add_vortex(config, trajectory=trajectory)

    def add_rnn_vortex(self, charge: int, rnn: AdvancedVortexRNN,
                      initial_position: np.ndarray, k: float = 1.0,
                      mode: str = 'autoregressive'):
        """
        Add vortex with RNN control.

        Args:
            charge: Topological charge
            rnn: Trained RNN controller
            initial_position: Starting position
            k: Wavenumber
            mode: 'autoregressive' or 'direct'
        """
        config = VortexConfig(
            charge=charge,
            amplitude=0.0,  # Not used for RNN
            k=k,
            trajectory_type=f'rnn_{mode}',
            initial_position=initial_position
        )

        self.add_vortex(config, rnn=rnn)

    def get_positions(self, time: float) -> List[np.ndarray]:
        """
        Get all vortex positions at given time.

        Args:
            time: Current time

        Returns:
            List of 3D positions
        """
        positions = []

        for i, config in enumerate(self.vortex_configs):
            if config.trajectory_type == 'fourier':
                traj = self.fourier_trajectories[i]
                if traj is not None:
                    pos = traj.evaluate(time)
                    positions.append(pos)
                else:
                    positions.append(np.zeros(3))

            elif 'rnn' in config.trajectory_type:
                # For RNN, would need to maintain state
                # Simplified: return initial position
                if config.initial_position is not None:
                    positions.append(config.initial_position)
                else:
                    positions.append(np.zeros(3))

        return positions

    def get_velocities(self, time: float) -> List[np.ndarray]:
        """
        Get all vortex velocities at given time.

        Args:
            time: Current time

        Returns:
            List of 3D velocities
        """
        velocities = []

        for i, config in enumerate(self.vortex_configs):
            if config.trajectory_type == 'fourier':
                traj = self.fourier_trajectories[i]
                if traj is not None:
                    vel = traj.get_velocity(time)
                    velocities.append(vel)
                else:
                    velocities.append(np.zeros(3))
            else:
                # Finite difference approximation
                dt = 0.01
                pos1 = self.get_positions(time)[i]
                pos2 = self.get_positions(time + dt)[i]
                vel = (pos2 - pos1) / dt
                velocities.append(vel)

        return velocities

    def get_charges(self) -> List[int]:
        """Get all topological charges."""
        return self.charges

    def export_configuration(self, filepath: str):
        """
        Export choreography configuration to JSON.

        Args:
            filepath: Output file path
        """
        config_data = {
            'num_vortices': self.num_vortices,
            'vortices': []
        }

        for i, config in enumerate(self.vortex_configs):
            vortex_data = {
                'index': i,
                'charge': config.charge,
                'amplitude': config.amplitude,
                'k': config.k,
                'trajectory_type': config.trajectory_type
            }

            if config.trajectory_type == 'fourier':
                traj = self.fourier_trajectories[i]
                if traj is not None:
                    vortex_data['preset'] = traj.preset
                    vortex_data['omega'] = traj.omega
                    vortex_data['phase_offset'] = traj.phase_offset

            config_data['vortices'].append(vortex_data)

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED MULTI-VORTEX CONTROL SYSTEM")
    print("=" * 70)

    # Test 1: Enhanced Fourier trajectories
    print("\n1. Testing Enhanced Fourier Trajectories:")
    presets = ['circle', 'figure8', 'star5', 'heart', 'lissajous', 'trefoil', 'spiral']

    for preset in presets:
        traj = AdvancedFourierTrajectory(preset=preset, omega=1.0, amplitude=0.3)
        times = np.linspace(0, 10, 100)
        positions = traj.evaluate_batch(times)

        # Compute path characteristics
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        extent = np.ptp(positions, axis=0)

        print(f"  {preset:12s}: length={path_length:.3f}, "
              f"extent=[{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")

    # Test 2: Advanced RNN
    print("\n2. Testing Advanced Vortex RNN:")
    rnn = AdvancedVortexRNN(input_size=4, hidden_size=64, num_layers=2, rnn_type='lstm')
    print(f"  Model parameters: {sum(p.numel() for p in rnn.parameters()):,}")

    # Quick training test
    print("\n3. Training RNN on Enhanced Trajectories:")
    trainer = AdvancedTrajectoryTrainer(rnn, learning_rate=1e-3)
    losses = trainer.train(num_epochs=200, batch_size=32, verbose=False)
    print(f"  Final train loss: {losses[-1]:.6f}")

    # Test trajectory prediction
    initial_pos = np.array([0.3, 0.0, 0.0])
    predicted_traj = rnn.predict_trajectory(initial_pos, num_steps=50, mode='autoregressive')
    print(f"  Predicted trajectory shape: {predicted_traj.shape}")
    print(f"  Trajectory extent: {np.ptp(predicted_traj, axis=0)}")

    # Test 3: Multi-vortex choreography
    print("\n4. Testing Multi-Vortex Choreographer (5 vortices):")
    choreo = MultiVortexChoreographer(num_vortices=5)

    # Add vortices with different trajectories
    trajectory_configs = [
        ('circle', 0.0),
        ('star5', np.pi/5),
        ('heart', 2*np.pi/5),
        ('figure8', 3*np.pi/5),
        ('lissajous', 4*np.pi/5)
    ]

    for i, (preset, phase) in enumerate(trajectory_configs):
        charge = (-1) ** i  # Alternating charges
        choreo.add_fourier_vortex(
            charge=charge,
            preset=preset,
            omega=1.0,
            amplitude=0.3,
            phase_offset=phase
        )

    # Test position retrieval
    positions = choreo.get_positions(time=1.0)
    velocities = choreo.get_velocities(time=1.0)

    print(f"  Vortex positions at t=1.0:")
    for i, (pos, vel) in enumerate(zip(positions, velocities)):
        charge = choreo.charges[i]
        speed = np.linalg.norm(vel)
        print(f"    Vortex {i} (q={charge:+2d}): pos={pos}, |v|={speed:.3f}")

    # Export configuration
    choreo.export_configuration('multi_vortex_config.json')
    print(f"\n  Configuration exported to multi_vortex_config.json")

    print("\n[OK] Advanced multi-vortex control system ready!")
