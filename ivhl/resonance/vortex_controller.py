"""
VHL Vortex Trajectory Controller

Advanced control systems for multi-vortex dynamics:
- Fourier series path generation (circles, figure-8, stars, custom shapes)
- Simple RNN for learned/autonomous trajectories
- Synchronized multi-vortex choreography
- Dynamic topological charge modulation

Physics Motivation:
- Vortex cores trace dark spots (phase singularities) in holographic field
- Moving vortices create time-varying interference → dynamic folded topologies
- Trajectories can encode information (holographic "writing")
- Complex paths reveal symmetry-breaking in resonant structures

Author: Zynerji
Date: 2025-12-15
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable
import json


class FourierTrajectory:
    """
    Generate smooth parametric trajectories using Fourier series.

    x(t) = Σ A_n * sin(n*ω*t + φ_n)
    y(t) = Σ B_n * sin(n*ω*t + ψ_n)
    z(t) = Σ C_n * sin(n*ω*t + θ_n)

    Supports preset shapes and custom coefficients.
    """

    def __init__(self, preset: str = 'circle', omega: float = 1.0,
                 num_harmonics: int = 5, amplitude: float = 0.3):
        """
        Initialize Fourier trajectory generator.

        Args:
            preset: 'circle', 'figure8', 'star', 'spiral', 'lissajous', 'custom'
            omega: Base angular frequency
            num_harmonics: Number of Fourier harmonics
            amplitude: Overall trajectory amplitude
        """
        self.preset = preset
        self.omega = omega
        self.num_harmonics = num_harmonics
        self.amplitude = amplitude

        # Fourier coefficients: (A_n, φ_n) for each harmonic
        self.coeffs_x = []
        self.coeffs_y = []
        self.coeffs_z = []

        self._initialize_preset()

    def _initialize_preset(self):
        """Initialize Fourier coefficients based on preset."""
        n = self.num_harmonics

        if self.preset == 'circle':
            # x = A*cos(ωt), y = A*sin(ωt), z = 0
            self.coeffs_x = [(self.amplitude, 0.0)]  # cos = sin(t + π/2)
            self.coeffs_y = [(self.amplitude, -np.pi/2)]
            self.coeffs_z = [(0.0, 0.0)]

        elif self.preset == 'figure8':
            # x = A*sin(ωt), y = A*sin(2ωt), z = 0
            self.coeffs_x = [(self.amplitude, 0.0)]
            self.coeffs_y = [(self.amplitude, 0.0)]  # Harmonic 2
            self.coeffs_z = [(0.0, 0.0)]
            # Add second harmonic for y
            self.coeffs_y.append((self.amplitude, 0.0))

        elif self.preset == 'star':
            # Multi-harmonic with odd symmetry
            for i in range(1, n+1):
                phase_offset = i * 2*np.pi / 5  # 5-fold symmetry
                self.coeffs_x.append((self.amplitude / i, phase_offset))
                self.coeffs_y.append((self.amplitude / i, phase_offset + np.pi/2))
                self.coeffs_z.append((0.0, 0.0))

        elif self.preset == 'spiral':
            # Rising spiral: r = const, z ∝ t
            self.coeffs_x = [(self.amplitude, 0.0)]
            self.coeffs_y = [(self.amplitude, -np.pi/2)]
            self.coeffs_z = [(0.1 * self.amplitude, 0.0)]  # Slow rise

        elif self.preset == 'lissajous':
            # x = A*sin(ωt), y = A*sin(2ωt), z = A*sin(3ωt)
            self.coeffs_x = [(self.amplitude, 0.0)]
            self.coeffs_y = [(self.amplitude, 0.0)]
            self.coeffs_z = [(self.amplitude * 0.5, 0.0)]
            # Add harmonics
            self.coeffs_y.append((self.amplitude * 0.5, np.pi/4))
            self.coeffs_z.append((self.amplitude * 0.3, np.pi/3))

        else:  # custom - user must set coefficients
            for i in range(n):
                self.coeffs_x.append((0.0, 0.0))
                self.coeffs_y.append((0.0, 0.0))
                self.coeffs_z.append((0.0, 0.0))

    def set_custom_coefficients(self, coeffs_x: List[Tuple[float, float]],
                               coeffs_y: List[Tuple[float, float]],
                               coeffs_z: List[Tuple[float, float]]):
        """Set custom Fourier coefficients."""
        self.coeffs_x = coeffs_x
        self.coeffs_y = coeffs_y
        self.coeffs_z = coeffs_z

    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate trajectory at time t.

        Args:
            t: Time parameter

        Returns:
            3D position (x, y, z)
        """
        position = np.zeros(3)

        # Sum Fourier series
        for n, (amp_x, phase_x) in enumerate(self.coeffs_x, start=1):
            position[0] += amp_x * np.sin(n * self.omega * t + phase_x)

        for n, (amp_y, phase_y) in enumerate(self.coeffs_y, start=1):
            position[1] += amp_y * np.sin(n * self.omega * t + phase_y)

        for n, (amp_z, phase_z) in enumerate(self.coeffs_z, start=1):
            position[2] += amp_z * np.sin(n * self.omega * t + phase_z)

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


class VortexRNN(nn.Module):
    """
    Simple LSTM-based RNN for learning vortex trajectories.

    Architecture:
    - Input: Time + current position (4D)
    - LSTM: Hidden state captures trajectory dynamics
    - Output: Delta position (3D) or full position

    Can be trained on example trajectories or used for autonomous generation.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 32,
                 num_layers: int = 2, output_size: int = 3):
        """
        Initialize RNN.

        Args:
            input_size: Input dimension (default: time + xyz = 4)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_size: Output dimension (xyz delta or position)
        """
        super(VortexRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()  # Bound output

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional hidden state

        Returns:
            Output tensor of shape (batch, seq_len, output_size)
        """
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Fully connected
        output = self.fc(lstm_out)
        output = self.tanh(output) * 0.5  # Scale to [-0.5, 0.5]

        return output, hidden

    def predict_trajectory(self, initial_pos: np.ndarray, num_steps: int,
                          dt: float = 0.05) -> np.ndarray:
        """
        Generate trajectory autoregressively.

        Args:
            initial_pos: Starting position (3D)
            num_steps: Number of time steps
            dt: Time step

        Returns:
            Trajectory of shape (num_steps, 3)
        """
        self.eval()

        trajectory = [initial_pos]
        current_pos = initial_pos
        hidden = None

        with torch.no_grad():
            for step in range(num_steps - 1):
                t = step * dt

                # Input: [time, x, y, z]
                input_tensor = torch.tensor([[t] + list(current_pos)],
                                           dtype=torch.float32).unsqueeze(0)

                # Predict delta
                delta, hidden = self.forward(input_tensor, hidden)
                delta = delta.squeeze().numpy()

                # Update position
                current_pos = current_pos + delta

                # Confine to sphere (optional)
                norm = np.linalg.norm(current_pos)
                if norm > 0.5:
                    current_pos = current_pos * 0.5 / norm

                trajectory.append(current_pos)

        return np.array(trajectory)


class TrajectoryTrainer:
    """
    Train RNN on example Fourier trajectories.
    """

    def __init__(self, rnn: VortexRNN):
        """
        Initialize trainer.

        Args:
            rnn: VortexRNN model to train
        """
        self.rnn = rnn
        self.optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def generate_training_data(self, num_trajectories: int = 100,
                              num_steps: int = 100, dt: float = 0.05) -> Tuple:
        """
        Generate training data from random Fourier trajectories.

        Args:
            num_trajectories: Number of example trajectories
            num_steps: Steps per trajectory
            dt: Time step

        Returns:
            (inputs, targets) as torch tensors
        """
        all_inputs = []
        all_targets = []

        presets = ['circle', 'figure8', 'star', 'spiral', 'lissajous']

        for _ in range(num_trajectories):
            # Random preset
            preset = np.random.choice(presets)
            omega = np.random.uniform(0.5, 2.0)
            amplitude = np.random.uniform(0.2, 0.4)

            traj_gen = FourierTrajectory(preset=preset, omega=omega,
                                        amplitude=amplitude)

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
        inputs = torch.tensor(all_inputs, dtype=torch.float32).unsqueeze(1)  # Add seq dim
        targets = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(1)

        return inputs, targets

    def train(self, num_epochs: int = 500, batch_size: int = 32,
             verbose: bool = True) -> List[float]:
        """
        Train the RNN.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print progress

        Returns:
            List of losses
        """
        # Generate data
        if verbose:
            print("Generating training data...")
        inputs, targets = self.generate_training_data()

        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True)

        losses = []

        if verbose:
            print(f"Training for {num_epochs} epochs...")

        self.rnn.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch_inputs, batch_targets in dataloader:
                # Forward
                outputs, _ = self.rnn(batch_inputs)

                # Loss
                loss = self.criterion(outputs, batch_targets)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        if verbose:
            print("✓ Training complete!")

        return losses

    def save_model(self, path: str):
        """Save trained model."""
        torch.save(self.rnn.state_dict(), path)

    def load_model(self, path: str):
        """Load trained model."""
        self.rnn.load_state_dict(torch.load(path))


class MultiVortexChoreographer:
    """
    Coordinate multiple vortices with synchronized or independent trajectories.
    """

    def __init__(self, num_vortices: int = 2):
        """
        Initialize choreographer.

        Args:
            num_vortices: Number of vortices to control
        """
        self.num_vortices = num_vortices
        self.trajectories: List[FourierTrajectory] = []
        self.rnn_controllers: List[VortexRNN] = []
        self.control_modes: List[str] = []  # 'fourier' or 'rnn'

        # Topological charges
        self.charges = [(-1)**i for i in range(num_vortices)]  # Alternating

    def add_fourier_vortex(self, preset: str = 'circle', omega: float = 1.0,
                          amplitude: float = 0.3, phase_offset: float = 0.0):
        """Add vortex with Fourier trajectory control."""
        traj = FourierTrajectory(preset=preset, omega=omega, amplitude=amplitude)

        # Apply phase offset for coordination
        for i in range(len(traj.coeffs_x)):
            amp, phase = traj.coeffs_x[i]
            traj.coeffs_x[i] = (amp, phase + phase_offset)
        for i in range(len(traj.coeffs_y)):
            amp, phase = traj.coeffs_y[i]
            traj.coeffs_y[i] = (amp, phase + phase_offset)

        self.trajectories.append(traj)
        self.control_modes.append('fourier')

    def add_rnn_vortex(self, rnn: VortexRNN):
        """Add vortex with RNN trajectory control."""
        self.rnn_controllers.append(rnn)
        self.control_modes.append('rnn')
        self.trajectories.append(None)  # Placeholder

    def get_positions(self, time: float) -> List[np.ndarray]:
        """
        Get all vortex positions at given time.

        Args:
            time: Current time

        Returns:
            List of 3D positions
        """
        positions = []

        for i, mode in enumerate(self.control_modes):
            if mode == 'fourier':
                pos = self.trajectories[i].evaluate(time)
                positions.append(pos)
            elif mode == 'rnn':
                # RNN needs to be evaluated differently (sequence)
                # For simplicity, use last predicted position
                # In practice, maintain state
                positions.append(np.zeros(3))  # Placeholder

        return positions

    def get_charges(self) -> List[int]:
        """Get topological charges for all vortices."""
        return self.charges

    def set_charges(self, charges: List[int]):
        """Set custom topological charges."""
        self.charges = charges


# Example usage and testing
if __name__ == "__main__":
    print("VHL Vortex Controller Module")
    print("=" * 70)

    # Test Fourier trajectories
    print("\n1. Testing Fourier Trajectories:")
    presets = ['circle', 'figure8', 'star', 'spiral']

    for preset in presets:
        traj = FourierTrajectory(preset=preset, omega=1.0, amplitude=0.3)
        positions = traj.evaluate_batch(np.linspace(0, 10, 100))
        print(f"  {preset:10s}: {len(positions)} points, "
              f"range=[{positions.min():.3f}, {positions.max():.3f}]")

    # Test RNN
    print("\n2. Testing Vortex RNN:")
    rnn = VortexRNN(input_size=4, hidden_size=32, num_layers=2)
    print(f"  Model parameters: {sum(p.numel() for p in rnn.parameters()):,}")

    # Quick training test
    print("\n3. Training RNN on Fourier examples:")
    trainer = TrajectoryTrainer(rnn)
    losses = trainer.train(num_epochs=100, verbose=False)
    print(f"  Final loss: {losses[-1]:.6f}")

    # Test trajectory prediction
    initial_pos = np.array([0.3, 0.0, 0.0])
    predicted_traj = rnn.predict_trajectory(initial_pos, num_steps=50)
    print(f"  Predicted trajectory: {predicted_traj.shape}")

    # Test choreographer
    print("\n4. Testing Multi-Vortex Choreographer:")
    choreo = MultiVortexChoreographer(num_vortices=3)
    choreo.add_fourier_vortex(preset='circle', phase_offset=0.0)
    choreo.add_fourier_vortex(preset='figure8', phase_offset=np.pi/3)
    choreo.add_fourier_vortex(preset='star', phase_offset=2*np.pi/3)

    positions = choreo.get_positions(time=1.0)
    print(f"  Vortex positions at t=1.0:")
    for i, pos in enumerate(positions):
        print(f"    Vortex {i}: {pos}")

    print("\n✓ Vortex controller ready!")
