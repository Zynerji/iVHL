"""
Group Field Theory and Advanced Tensor Models
==============================================

Implements the UV-complete tensorial foundation for emergent quantum geometry:

1. Colored Tensor Models: Full U(N)^d invariance with colored indices
2. Melonic Dominance: Leading order in large-N expansion
3. Schwinger-Dyson Equations: Self-consistency equations for propagators
4. Double-Scaling Limit: Criticality and continuum limit
5. Statistical Mechanics: Partition functions and free energy

Tensor models provide the statistical mechanics framework where:
- Pre-geometric phase: Disordered tensor fluctuations (small N or weak coupling)
- Geometric phase: Ordered tensor structures (large N, strong coupling)
- Phase transition: Critical point with universal behavior

The colored tensor model action:
    S[T] = (1/2) Tr(T† T) + (λ/d!) Tr(T^d + T†^d)

where T is a rank-d tensor with d colored indices, each in U(N).

Large-N limit: Melonic diagrams dominate (analogous to planar diagrams in matrix models)

Author: Zynerji
Date: 2025-12-15
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from scipy.optimize import fsolve, minimize
from scipy.integrate import odeint
import itertools


@dataclass
class TensorModelConfig:
    """Configuration for tensor model."""
    rank: int = 3  # d (number of colored indices)
    dimension: int = 5  # N (size of each index)
    coupling: float = 0.1  # λ
    mass: float = 1.0  # m^2 in kinetic term
    colored: bool = True  # Use colored vs uncolored model


class ColoredTensorModel:
    """
    Colored tensor model with U(N)^d invariance.

    The field T is a rank-d tensor: T_{i1,i2,...,id} where each index
    i_c is in {1,...,N} and transforms under U(N) for color c.

    Invariance: T → U1 ⊗ U2 ⊗ ... ⊗ Ud · T · (U1 ⊗ U2 ⊗ ... ⊗ Ud)†

    The interaction is the trace of T^d which contracts indices in a specific pattern.
    Melonic diagrams dominate in the large-N limit.
    """

    def __init__(self, config: TensorModelConfig):
        """
        Initialize colored tensor model.

        Args:
            config: Tensor model configuration
        """
        self.config = config
        self.d = config.rank
        self.N = config.dimension
        self.lambda_coupling = config.coupling
        self.mass_sq = config.mass

        # Propagator (inverse kinetic operator)
        self.propagator_value = 1.0 / self.mass_sq

        # Melonic amplitude (leading order)
        self.melonic_amplitude = None

        # Partition function
        self.log_Z = None

    def generate_tensor(self, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate random tensor field configuration.

        Args:
            amplitude: Overall amplitude scale

        Returns:
            Tensor field array of shape (N, N, ..., N) with d indices
        """
        shape = tuple([self.N] * self.d)

        # Complex Gaussian random field
        real_part = np.random.randn(*shape)
        imag_part = np.random.randn(*shape)
        tensor = (real_part + 1j * imag_part) * amplitude / np.sqrt(2)

        return tensor

    def action(self, tensor: np.ndarray) -> float:
        """
        Compute action S[T] = (1/2) Tr(T† T) + (λ/d!) Tr(T^d)

        Args:
            tensor: Tensor field configuration

        Returns:
            Action value
        """
        # Kinetic term: (m^2/2) Tr(T† T)
        kinetic = (self.mass_sq / 2) * np.sum(np.abs(tensor)**2)

        # Interaction term: (λ/d!) Tr(T^d)
        # For simplicity, we approximate with average contraction
        interaction = self._compute_interaction(tensor)

        return kinetic + interaction

    def _compute_interaction(self, tensor: np.ndarray) -> float:
        """
        Compute interaction term Tr(T^d).

        For a rank-d tensor, T^d means contracting indices in the melonic pattern.
        This is computationally expensive, so we use approximations.

        Args:
            tensor: Tensor field

        Returns:
            Interaction term value
        """
        # Simplified: Sum over selected index contractions
        # Full implementation would compute all d! permutations

        if self.d == 3:
            # For d=3: T_{ijk} T_{jkl} T_{lim} contracts to scalar
            # Approximate with trace of matricization
            T_matrix = tensor.reshape(self.N, self.N * self.N)
            trace_est = np.trace(T_matrix @ T_matrix.conj().T @ T_matrix) / self.N

        elif self.d == 4:
            # For d=4: Similar matricization approach
            T_matrix = tensor.reshape(self.N**2, self.N**2)
            trace_est = np.trace(T_matrix @ T_matrix @ T_matrix @ T_matrix.conj().T) / self.N**2

        else:
            # General case: Use Frobenius norm as proxy
            trace_est = np.linalg.norm(tensor)**(2*self.d) / (self.N**(self.d-1))

        interaction = (self.lambda_coupling / np.math.factorial(self.d)) * trace_est.real

        return interaction

    def melonic_propagator_sd_equation(self, G: float) -> float:
        """
        Schwinger-Dyson equation for melonic 2-point function.

        At leading order in 1/N:
        G^{-1} = m^2 + Σ(G)

        where Σ(G) = λ (d-1) G^{d-1} is the melonic self-energy.

        Args:
            G: Propagator value (to solve for)

        Returns:
            Residual of SD equation (should be zero at solution)
        """
        # Self-energy from melonic diagrams
        sigma = self.lambda_coupling * (self.d - 1) * G**(self.d - 1)

        # SD equation: G^{-1} = m^2 + Σ
        residual = (1.0 / G) - self.mass_sq - sigma

        return residual

    def solve_melonic_propagator(self) -> float:
        """
        Solve Schwinger-Dyson equation for dressed propagator.

        Returns:
            Dressed propagator value G*
        """
        # Initial guess: bare propagator
        G0 = 1.0 / self.mass_sq

        # Solve SD equation
        G_solution = fsolve(self.melonic_propagator_sd_equation, G0)[0]

        self.propagator_value = G_solution

        return G_solution

    def free_energy_large_N(self, G: float) -> float:
        """
        Compute free energy per degree of freedom in large-N limit.

        F/N^d = (1/2) ln(G^{-1}/m^2) + (1/2) m^2 G + (λ/d!) G^d

        Args:
            G: Propagator value

        Returns:
            Free energy density
        """
        # Three contributions
        kinetic = 0.5 * np.log(G**(-1) / self.mass_sq)
        mass_term = 0.5 * self.mass_sq * G
        interaction = (self.lambda_coupling / np.math.factorial(self.d)) * G**self.d

        free_energy = kinetic + mass_term + interaction

        return free_energy

    def partition_function_large_N(self) -> float:
        """
        Compute partition function in large-N (melonic) limit.

        Z = exp(-N^d F[G*])

        Returns:
            Log of partition function
        """
        # Solve for dressed propagator
        G_star = self.solve_melonic_propagator()

        # Compute free energy
        F = self.free_energy_large_N(G_star)

        # Log partition function
        log_Z = -self.N**self.d * F

        self.log_Z = log_Z

        return log_Z

    def critical_coupling_estimate(self) -> float:
        """
        Estimate critical coupling λ_c where melonic propagator diverges.

        At criticality: G → ∞, self-energy balances kinetic term.

        Condition: m^2 ~ λ_c (d-1) G_c^{d-2}

        Returns:
            Estimated critical coupling
        """
        # At large G, SD equation becomes:
        # G^{-1} ≈ λ (d-1) G^{d-1}
        # This gives: G^d ≈ 1 / (λ (d-1))

        # Critical point when denominator vanishes
        # For m^2 = λ_c (d-1) / G_c^{d-2}, we need G_c → ∞

        # Approximate: λ_c ~ m^2 (some factor)
        lambda_c = self.mass_sq * self.d / (self.d - 1)

        return lambda_c

    def double_scaling_limit(self, N_values: np.ndarray) -> Dict:
        """
        Explore double-scaling limit near criticality.

        Take N → ∞ while λ → λ_c such that:
        (λ_c - λ) N^{d/(d-1)} = fixed

        This reveals continuum critical behavior.

        Args:
            N_values: Array of N values to explore

        Returns:
            Dictionary with scaling data
        """
        lambda_c = self.critical_coupling_estimate()

        results = {
            'N_values': N_values,
            'lambda_c': lambda_c,
            'propagators': [],
            'susceptibilities': [],
            'scaling_variable': []
        }

        for N in N_values:
            # Set N
            old_N = self.N
            self.N = N

            # Tune λ near critical point
            epsilon = 0.1 / N**(self.d / (self.d - 1))
            self.lambda_coupling = lambda_c - epsilon

            # Solve for propagator
            G = self.solve_melonic_propagator()

            # Susceptibility (response to external field)
            chi = G * self.N**self.d

            # Scaling variable
            t = epsilon * N**(self.d / (self.d - 1))

            results['propagators'].append(G)
            results['susceptibilities'].append(chi)
            results['scaling_variable'].append(t)

            # Restore N
            self.N = old_N

        results['propagators'] = np.array(results['propagators'])
        results['susceptibilities'] = np.array(results['susceptibilities'])
        results['scaling_variable'] = np.array(results['scaling_variable'])

        return results

    def melonic_diagram_amplitude(self, n_bubbles: int = 1) -> complex:
        """
        Compute amplitude of melonic diagram with n bubble insertions.

        Melonic diagrams have a tree-like structure in the colored graph representation.
        Each bubble contributes a factor of λ G^d.

        Args:
            n_bubbles: Number of bubble (interaction) vertices

        Returns:
            Diagram amplitude
        """
        G = self.propagator_value

        # Each bubble vertex contributes λ/(d!) and d propagators
        amplitude = (self.lambda_coupling / np.math.factorial(self.d))**n_bubbles
        amplitude *= G**(self.d * n_bubbles)
        amplitude *= self.N**(self.d * n_bubbles)  # Combinatorial factor

        self.melonic_amplitude = amplitude

        return amplitude

    def generate_colored_graph(self, n_vertices: int = 5) -> Dict:
        """
        Generate a colored graph representing a Feynman diagram.

        In colored tensor models, each line carries d colors.
        Melonic graphs are tree-like when edges of different colors are removed.

        Args:
            n_vertices: Number of interaction vertices

        Returns:
            Graph data structure
        """
        # Simplified colored graph generation
        graph = {
            'vertices': list(range(n_vertices)),
            'edges': [],
            'colors': []
        }

        # Create melonic structure: tree for each color
        for vertex in range(1, n_vertices):
            parent = vertex // 2

            for color in range(self.d):
                edge = (parent, vertex)
                graph['edges'].append(edge)
                graph['colors'].append(color)

        return graph


class UncoloredTensorModel:
    """
    Uncolored tensor model (simpler, but less structure).

    Field T_{i1...id} with all indices equivalent (no color distinction).
    Useful for comparison and as a simpler test case.
    """

    def __init__(self, config: TensorModelConfig):
        """Initialize uncolored tensor model."""
        self.config = config
        self.d = config.rank
        self.N = config.dimension
        self.lambda_coupling = config.coupling
        self.mass_sq = config.mass

    def action(self, tensor: np.ndarray) -> float:
        """Compute action (similar to colored case but without color structure)."""
        kinetic = (self.mass_sq / 2) * np.sum(np.abs(tensor)**2)

        # Interaction: sum of all contractions (less structured than colored case)
        interaction_approx = (self.lambda_coupling / np.math.factorial(self.d)) * \
                            np.linalg.norm(tensor)**(2*self.d) / self.N**(self.d-1)

        return kinetic + interaction_approx.real


class TensorModelStatistics:
    """
    Statistical mechanics analysis of tensor models.

    Computes:
    - Partition function
    - Free energy
    - Correlation functions
    - Phase transitions
    """

    def __init__(self, model: ColoredTensorModel):
        """
        Initialize statistics calculator.

        Args:
            model: Tensor model instance
        """
        self.model = model

    def compute_partition_function_numerical(self, n_samples: int = 1000) -> float:
        """
        Compute partition function via Monte Carlo.

        Z = ∫ DT exp(-S[T])

        Args:
            n_samples: Number of Monte Carlo samples

        Returns:
            Estimated log partition function
        """
        actions = []

        for _ in range(n_samples):
            T = self.model.generate_tensor(amplitude=1.0)
            S = self.model.action(T)
            actions.append(S)

        actions = np.array(actions)

        # Estimate log Z using importance sampling
        log_Z = -np.mean(actions) + np.log(n_samples)

        return log_Z

    def correlation_function_2point(self, separation: float) -> float:
        """
        Compute 2-point correlation function ⟨T(x) T†(y)⟩.

        In melonic limit: G(x-y) ~ exp(-m |x-y|) / |x-y|^{d-1}

        Args:
            separation: Distance |x-y|

        Returns:
            Correlation value
        """
        G = self.model.propagator_value

        # Approximate form
        if separation < 1e-10:
            correlation = G
        else:
            # Exponential decay with power-law pre-factor
            correlation = G * np.exp(-np.sqrt(self.model.mass_sq) * separation) / \
                         separation**(self.model.d - 1)

        return correlation

    def susceptibility(self) -> float:
        """
        Compute susceptibility χ = ∂⟨T⟩/∂J (response to external field).

        In large-N: χ ~ G N^d

        Returns:
            Susceptibility value
        """
        G = self.model.propagator_value
        chi = G * self.model.N**self.model.d

        return chi

    def correlation_length(self) -> float:
        """
        Compute correlation length ξ.

        From 2-point function decay: ξ = 1/m_eff

        where m_eff is the effective mass from dressed propagator.

        Returns:
            Correlation length
        """
        # Effective mass from propagator pole
        G = self.model.propagator_value

        # From G = 1/(m_eff^2), we get m_eff
        m_eff = np.sqrt(1.0 / G)

        xi = 1.0 / m_eff

        return xi


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED TENSOR MODELS FOR QUANTUM GEOMETRY")
    print("=" * 70)

    # Test 1: Colored tensor model with melonic analysis
    print("\n1. Colored Tensor Model (rank d=3, dimension N=10):")
    config = TensorModelConfig(rank=3, dimension=10, coupling=0.1, mass=1.0)
    model = ColoredTensorModel(config)

    print(f"  Configuration: d={model.d}, N={model.N}, λ={model.lambda_coupling}, m^2={model.mass_sq}")

    # Solve Schwinger-Dyson
    print("\n2. Solving Melonic Schwinger-Dyson Equation:")
    G_dressed = model.solve_melonic_propagator()
    print(f"  Bare propagator: G0 = {1.0/model.mass_sq:.6f}")
    print(f"  Dressed propagator: G* = {G_dressed:.6f}")
    print(f"  Enhancement factor: G*/G0 = {G_dressed * model.mass_sq:.6f}")

    # Free energy and partition function
    print("\n3. Thermodynamics in Large-N Limit:")
    log_Z = model.partition_function_large_N()
    F = model.free_energy_large_N(G_dressed)
    print(f"  Free energy density: F/N^d = {F:.6f}")
    print(f"  Log partition function: ln Z = {log_Z:.2f}")

    # Critical coupling
    print("\n4. Critical Behavior:")
    lambda_c = model.critical_coupling_estimate()
    print(f"  Estimated critical coupling: λ_c = {lambda_c:.6f}")
    print(f"  Current coupling: λ = {model.lambda_coupling:.6f}")
    print(f"  Distance from criticality: (λ_c - λ)/λ_c = {(lambda_c - model.lambda_coupling)/lambda_c:.4f}")

    # Melonic diagram amplitude
    print("\n5. Melonic Diagram Amplitudes:")
    for n in range(1, 4):
        amp = model.melonic_diagram_amplitude(n_bubbles=n)
        print(f"  {n}-bubble melonic diagram: A_{n} = {amp.real:.4e}")

    # Double-scaling limit
    print("\n6. Double-Scaling Limit Exploration:")
    N_values = np.array([5, 10, 20, 40])
    ds_results = model.double_scaling_limit(N_values)

    print(f"  Scaling near λ_c = {ds_results['lambda_c']:.6f}:")
    for i, N in enumerate(N_values):
        print(f"    N={N:3d}: G={ds_results['propagators'][i]:.6f}, "
              f"χ={ds_results['susceptibilities'][i]:.4e}, "
              f"t={ds_results['scaling_variable'][i]:.6f}")

    # Statistical mechanics
    print("\n7. Statistical Mechanics Analysis:")
    stats = TensorModelStatistics(model)

    chi = stats.susceptibility()
    xi = stats.correlation_length()

    print(f"  Susceptibility: χ = {chi:.4e}")
    print(f"  Correlation length: ξ = {xi:.6f}")

    # Test various separations
    print(f"\n  2-point correlations:")
    for r in [0.1, 0.5, 1.0, 2.0]:
        G_r = stats.correlation_function_2point(r)
        print(f"    G(r={r:.1f}) = {G_r:.6f}")

    # Colored graph generation
    print("\n8. Colored Graph Structure:")
    graph = model.generate_colored_graph(n_vertices=5)
    print(f"  Generated melonic graph: {len(graph['vertices'])} vertices, {len(graph['edges'])} edges")
    print(f"  Colors: {model.d} (one for each index)")

    # Test action computation
    print("\n9. Action Computation Test:")
    T = model.generate_tensor(amplitude=1.0)
    S = model.action(T)
    print(f"  Random tensor field shape: {T.shape}")
    print(f"  Action S[T] = {S:.6f}")

    print("\n[OK] Advanced tensor models ready!")
    print(f"\nKey insight: Melonic diagrams dominate at large N, providing")
    print(f"solvable dynamics that bridge to geometric phases via criticality.")
