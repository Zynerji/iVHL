"""
GFT Condensate Dynamics and Phase Transitions

This module implements the mean-field analysis of Group Field Theory (GFT) condensates,
exploring the transition between pre-geometric (non-geometric) and geometric phases.
The condensate field ⟨φ⟩ ~ σ acts as an order parameter, with σ=0 representing
quantum-geometric disorder and σ≠0 representing emergent classical spacetime.

Key Physics:
-----------
1. **Mean-Field Effective Action**:
   S_eff[σ] = ∫ dg [|∂σ|² + m² |σ|² + λ |σ|^d]

2. **Effective Potential**:
   V_eff(σ) = (m²/2) |σ|² + (λ/d!) |σ|^d

3. **Phase Transition**:
   - m² > 0: Non-geometric phase (σ = 0 is minimum)
   - m² < 0: Geometric phase (σ ≠ 0 is minimum, spontaneous symmetry breaking)

4. **Cosmological Interpretation**:
   - Condensate density ρ ~ |σ|² → emergent energy density
   - Condensate gradient ∂σ → emergent spatial curvature
   - Time evolution: Gross-Pitaevskii dynamics for σ(g,t)

5. **Critical Behavior**:
   - Critical coupling: λ_c ~ m²^(d/2)
   - Universality class: Mean-field exponents (β = 1/2, γ = 1, ν = 1/2)
   - Order parameter: σ_eq ~ √(-m²/λ) for m² < 0

Integration with Tensor Models:
-------------------------------
- Tensor model partition function Z[T] provides fluctuations around mean-field
- Melonic dominance ensures mean-field validity in large-N limit
- Critical behavior from tensor models matches condensate phase transition

Author: Advanced Holographic Framework
Date: 2025-12-15
"""

import numpy as np
from scipy.optimize import fsolve, minimize, brentq
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp2d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import json
import math

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CondensateConfig:
    """Configuration for GFT condensate dynamics"""

    # Field theory parameters
    rank: int = 3  # d (tensor rank, related to dimension)
    mass_squared: float = 1.0  # m² (bare mass term)
    coupling: float = 0.1  # λ (interaction strength)

    # Numerical parameters
    num_group_points: int = 100  # Discretization of group manifold
    num_time_steps: int = 200  # Time evolution steps
    time_max: float = 10.0  # Maximum evolution time

    # Phase diagram parameters
    mass_range: Tuple[float, float] = (-2.0, 2.0)
    coupling_range: Tuple[float, float] = (0.01, 1.0)
    grid_resolution: int = 50

    # Cosmological parameters
    planck_mass: float = 1.0  # M_Pl (Planck mass in natural units)
    hubble_scale: float = 0.1  # H₀ (fiducial Hubble scale)


# ============================================================================
# Mean-Field Effective Potential
# ============================================================================

class EffectivePotential:
    """
    Effective potential for GFT condensate in mean-field approximation

    V_eff(σ) = (m²/2) |σ|² + (λ/d!) |σ|^d + radiative corrections

    This determines the vacuum structure and phase transitions.
    """

    def __init__(self, config: CondensateConfig):
        self.config = config
        self.d = config.rank
        self.m2 = config.mass_squared
        self.lambda_coupling = config.coupling

        # Cache for minimum
        self._minimum_sigma = None
        self._minimum_value = None

    def __call__(self, sigma: float) -> float:
        """
        Compute effective potential V_eff(σ)

        Args:
            sigma: Condensate field value (real, assuming real field)

        Returns:
            V_eff(σ)
        """
        # Quadratic term
        quadratic = (self.m2 / 2.0) * sigma**2

        # Interaction term
        interaction = (self.lambda_coupling / math.factorial(self.d)) * sigma**self.d

        # Total potential
        V = quadratic + interaction

        return V

    def derivative(self, sigma: float) -> float:
        """
        dV/dσ = m² σ + λ σ^(d-1)

        Extrema satisfy this = 0
        """
        return self.m2 * sigma + self.lambda_coupling * sigma**(self.d - 1)

    def second_derivative(self, sigma: float) -> float:
        """
        d²V/dσ² = m² + λ (d-1) σ^(d-2)

        Determines stability of extrema
        """
        return self.m2 + self.lambda_coupling * (self.d - 1) * sigma**(self.d - 2)

    def find_minima(self) -> List[Tuple[float, float, bool]]:
        """
        Find all extrema and determine stability

        Returns:
            List of (sigma, V(sigma), is_stable) tuples
        """
        extrema = []

        # Trivial extremum at σ = 0
        sigma_0 = 0.0
        V_0 = self(sigma_0)
        stable_0 = self.second_derivative(sigma_0) > 0
        extrema.append((sigma_0, V_0, stable_0))

        # Non-trivial extrema (if m² < 0)
        # From dV/dσ = 0: σ (m² + λ σ^(d-2)) = 0
        # Non-zero solution: σ^(d-2) = -m²/λ

        if self.m2 < 0 and self.lambda_coupling > 0:
            sigma_nt = (-self.m2 / self.lambda_coupling)**(1.0 / (self.d - 2))
            V_nt = self(sigma_nt)
            stable_nt = self.second_derivative(sigma_nt) > 0
            extrema.append((sigma_nt, V_nt, stable_nt))

            # By symmetry (if field is real), also negative solution
            extrema.append((-sigma_nt, V_nt, stable_nt))

        return extrema

    def global_minimum(self) -> Tuple[float, float]:
        """
        Find global minimum of potential

        Returns:
            (sigma_min, V_min)
        """
        extrema = self.find_minima()
        stable_extrema = [(s, V) for s, V, stable in extrema if stable]

        if not stable_extrema:
            # No stable minimum found, return origin
            return (0.0, self(0.0))

        # Return extremum with lowest energy
        sigma_min, V_min = min(stable_extrema, key=lambda x: x[1])

        self._minimum_sigma = sigma_min
        self._minimum_value = V_min

        return (sigma_min, V_min)

    def critical_mass_squared(self) -> float:
        """
        Critical value m²_c where phase transition occurs

        For d > 2, transition at m² = 0 (second-order)
        """
        return 0.0

    def order_parameter(self) -> float:
        """
        Compute equilibrium order parameter σ_eq

        Returns:
            σ_eq (condensate expectation value)
        """
        sigma_min, _ = self.global_minimum()
        return sigma_min


# ============================================================================
# Phase Diagram Explorer
# ============================================================================

class PhaseDiagram:
    """
    Explore phase structure in (m², λ) parameter space

    Identifies:
    - Non-geometric phase: σ = 0 (disordered, quantum gravity foam)
    - Geometric phase: σ ≠ 0 (ordered, emergent classical spacetime)
    - Critical line: phase transition boundary
    """

    def __init__(self, config: CondensateConfig):
        self.config = config
        self.d = config.rank

        # Phase diagram data
        self.mass_grid = None
        self.coupling_grid = None
        self.order_parameter_grid = None
        self.free_energy_grid = None
        self.phase_labels = None

    def compute_phase_diagram(self) -> Dict:
        """
        Compute full phase diagram

        Returns:
            Dictionary with grid data and analysis
        """
        # Set up parameter grids
        m2_min, m2_max = self.config.mass_range
        lambda_min, lambda_max = self.config.coupling_range
        N = self.config.grid_resolution

        m2_values = np.linspace(m2_min, m2_max, N)
        lambda_values = np.linspace(lambda_min, lambda_max, N)

        self.mass_grid, self.coupling_grid = np.meshgrid(m2_values, lambda_values)

        # Initialize result grids
        self.order_parameter_grid = np.zeros_like(self.mass_grid)
        self.free_energy_grid = np.zeros_like(self.mass_grid)
        self.phase_labels = np.zeros_like(self.mass_grid, dtype=int)

        # Scan parameter space
        for i in range(N):
            for j in range(N):
                m2 = self.mass_grid[i, j]
                lambda_val = self.coupling_grid[i, j]

                # Create potential for this parameter point
                config = CondensateConfig(
                    rank=self.d,
                    mass_squared=m2,
                    coupling=lambda_val
                )
                potential = EffectivePotential(config)

                # Find equilibrium
                sigma_eq, V_min = potential.global_minimum()

                # Store results
                self.order_parameter_grid[i, j] = np.abs(sigma_eq)
                self.free_energy_grid[i, j] = V_min

                # Classify phase
                if np.abs(sigma_eq) < 1e-6:
                    self.phase_labels[i, j] = 0  # Non-geometric
                else:
                    self.phase_labels[i, j] = 1  # Geometric

        # Identify critical line
        critical_line = self._extract_critical_line()

        return {
            'mass_grid': self.mass_grid,
            'coupling_grid': self.coupling_grid,
            'order_parameter': self.order_parameter_grid,
            'free_energy': self.free_energy_grid,
            'phase_labels': self.phase_labels,
            'critical_line': critical_line,
            'phase_names': {0: 'Non-Geometric', 1: 'Geometric'}
        }

    def _extract_critical_line(self) -> np.ndarray:
        """
        Extract phase transition boundary

        Returns:
            Array of (m², λ) points on critical line
        """
        # Find boundary between phases
        critical_points = []

        N = self.config.grid_resolution
        for i in range(1, N):
            for j in range(1, N):
                # Check if neighboring points have different phases
                if self.phase_labels[i, j] != self.phase_labels[i-1, j] or \
                   self.phase_labels[i, j] != self.phase_labels[i, j-1]:
                    critical_points.append([
                        self.mass_grid[i, j],
                        self.coupling_grid[i, j]
                    ])

        if critical_points:
            return np.array(critical_points)
        else:
            return np.array([])

    def susceptibility(self, m2: float, lambda_val: float) -> float:
        """
        Compute susceptibility χ = ∂σ/∂h near phase transition

        Diverges at critical point
        """
        config = CondensateConfig(rank=self.d, mass_squared=m2, coupling=lambda_val)
        potential = EffectivePotential(config)
        sigma_eq = potential.order_parameter()

        # χ⁻¹ = d²V/dσ² at equilibrium
        chi_inv = potential.second_derivative(sigma_eq)

        if np.abs(chi_inv) < 1e-10:
            return 1e10  # Divergent susceptibility
        else:
            return 1.0 / chi_inv


# ============================================================================
# Gross-Pitaevskii Dynamics
# ============================================================================

class GrossPitaevskiiEvolution:
    """
    Time evolution of GFT condensate via Gross-Pitaevskii equation

    i ∂σ/∂t = -∇² σ + m² σ + λ |σ|^(d-2) σ

    This is the mean-field dynamics beyond equilibrium, relevant for
    cosmological scenarios like bouncing universes.
    """

    def __init__(self, config: CondensateConfig):
        self.config = config
        self.d = config.rank

        # Store evolution history
        self.times = None
        self.sigma_real_history = None
        self.sigma_imag_history = None

    def evolve(self, initial_sigma: np.ndarray,
               spatial_grid: Optional[np.ndarray] = None) -> Dict:
        """
        Evolve condensate field in time

        Args:
            initial_sigma: Initial field configuration σ(x, t=0)
            spatial_grid: Spatial coordinate grid (optional)

        Returns:
            Evolution history dictionary
        """
        # Set up time grid
        times = np.linspace(0, self.config.time_max, self.config.num_time_steps)
        self.times = times

        # For simplicity, evolve spatially homogeneous mode
        # σ(t) satisfies: i dσ/dt = m² σ + λ |σ|^(d-2) σ

        # Convert to real equations: σ = σ_r + i σ_i
        # dσ_r/dt = -(m² σ_i + λ |σ|^(d-2) σ_i)
        # dσ_i/dt = +(m² σ_r + λ |σ|^(d-2) σ_r)

        def gp_equations(t, y):
            """GP equations for homogeneous mode"""
            sigma_r, sigma_i = y
            sigma_abs = np.sqrt(sigma_r**2 + sigma_i**2)

            if sigma_abs < 1e-10:
                nonlinear = 0.0
            else:
                nonlinear = self.config.coupling * sigma_abs**(self.d - 2)

            dsigma_r_dt = -(self.config.mass_squared * sigma_i + nonlinear * sigma_i)
            dsigma_i_dt = +(self.config.mass_squared * sigma_r + nonlinear * sigma_r)

            return [dsigma_r_dt, dsigma_i_dt]

        # Initial conditions
        if np.iscomplexobj(initial_sigma):
            y0 = [np.real(initial_sigma[0]), np.imag(initial_sigma[0])]
        else:
            y0 = [initial_sigma[0], 0.0]

        # Integrate
        sol = solve_ivp(gp_equations, (0, self.config.time_max), y0,
                       t_eval=times, method='RK45')

        self.sigma_real_history = sol.y[0]
        self.sigma_imag_history = sol.y[1]

        # Compute observables
        density = self.sigma_real_history**2 + self.sigma_imag_history**2
        energy = self._compute_energy_density(self.sigma_real_history, self.sigma_imag_history)

        return {
            'times': times,
            'sigma_real': self.sigma_real_history,
            'sigma_imag': self.sigma_imag_history,
            'density': density,
            'energy': energy
        }

    def _compute_energy_density(self, sigma_r: np.ndarray, sigma_i: np.ndarray) -> np.ndarray:
        """
        Energy density: ρ = |∂σ/∂t|² + V(σ)
        """
        # Kinetic energy (time derivative term)
        # For homogeneous mode, only potential contributes
        potential = EffectivePotential(self.config)

        sigma_abs = np.sqrt(sigma_r**2 + sigma_i**2)
        energy = np.array([potential(s) for s in sigma_abs])

        return energy


# ============================================================================
# Cosmological Interpretation
# ============================================================================

class CosmologicalCondensate:
    """
    Interpret GFT condensate as emergent cosmological spacetime

    Key ideas:
    - Condensate density ρ ~ |σ|² → emergent energy density
    - Condensate pressure P ~ -V(σ) → equation of state
    - Friedmann equations emerge from GP dynamics
    - Bouncing cosmology: condensate avoids singularity
    """

    def __init__(self, config: CondensateConfig):
        self.config = config
        self.d = config.rank

    def effective_metric_friedmann(self, sigma_history: np.ndarray,
                                   times: np.ndarray) -> Dict:
        """
        Extract emergent FLRW metric from condensate dynamics

        Scale factor: a(t) ~ |σ(t)|^(2/d)
        Hubble rate: H(t) = ȧ/a

        Args:
            sigma_history: Condensate field evolution
            times: Time array

        Returns:
            Cosmological observables
        """
        # Condensate density
        rho = np.abs(sigma_history)**2

        # Scale factor (normalized)
        a = rho**(1.0 / self.d)
        a = a / a[0]  # Normalize to a(0) = 1

        # Hubble rate
        H = np.gradient(a, times) / a

        # Effective potential
        potential = EffectivePotential(self.config)
        V = np.array([potential(np.abs(s)) for s in sigma_history])

        # Energy density and pressure from potential
        energy_density = rho + V
        pressure = rho - V  # Simplified; exact form depends on dynamics

        # Equation of state
        w = pressure / (energy_density + 1e-10)

        return {
            'times': times,
            'scale_factor': a,
            'hubble': H,
            'energy_density': energy_density,
            'pressure': pressure,
            'eos_parameter': w,
            'condensate_density': rho
        }

    def detect_bounce(self, a: np.ndarray, times: np.ndarray) -> Optional[Dict]:
        """
        Detect bouncing cosmology (a reaches minimum, then increases)

        Returns:
            Bounce parameters if detected, None otherwise
        """
        # Find minimum of scale factor
        idx_min = np.argmin(a)

        # Check if minimum is not at boundaries
        if idx_min > 0 and idx_min < len(a) - 1:
            # Check that a is decreasing before and increasing after
            if a[idx_min - 1] > a[idx_min] and a[idx_min + 1] > a[idx_min]:
                return {
                    'bounce_time': times[idx_min],
                    'bounce_scale_factor': a[idx_min],
                    'detected': True
                }

        return None

    def compute_page_curve_analog(self, sigma_history: np.ndarray,
                                  times: np.ndarray) -> Dict:
        """
        Compute Page curve analog for condensate entropy

        S_condensate ~ N_eff ln(|σ|) where N_eff ~ |σ|^d

        This tracks entanglement between geometric and non-geometric modes
        """
        # Effective number of quanta
        N_eff = np.abs(sigma_history)**self.d

        # Entropy (von Neumann-like)
        # S ~ N ln(N) - N for large N
        entropy = N_eff * np.log(N_eff + 1.0) - N_eff

        # Normalize
        entropy = entropy - np.min(entropy)

        return {
            'times': times,
            'entropy': entropy,
            'effective_quanta': N_eff
        }


# ============================================================================
# Main Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GFT CONDENSATE DYNAMICS & PHASE TRANSITIONS")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. Effective Potential Analysis
    # ========================================================================

    print("1. Testing Effective Potential:")
    print("-" * 70)

    # Non-geometric phase (m² > 0)
    config_nongeom = CondensateConfig(rank=3, mass_squared=1.0, coupling=0.1)
    V_nongeom = EffectivePotential(config_nongeom)
    sigma_min_ng, V_min_ng = V_nongeom.global_minimum()

    print(f"  Non-Geometric Phase (m^2 = {config_nongeom.mass_squared}):")
    print(f"    sigma_min = {sigma_min_ng:.6f}")
    print(f"    V_min = {V_min_ng:.6f}")
    print(f"    Phase: {'Non-Geometric' if abs(sigma_min_ng) < 1e-6 else 'Geometric'}")
    print()

    # Geometric phase (m^2 < 0)
    config_geom = CondensateConfig(rank=3, mass_squared=-1.0, coupling=0.1)
    V_geom = EffectivePotential(config_geom)
    sigma_min_g, V_min_g = V_geom.global_minimum()

    print(f"  Geometric Phase (m^2 = {config_geom.mass_squared}):")
    print(f"    sigma_min = {sigma_min_g:.6f}")
    print(f"    V_min = {V_min_g:.6f}")
    print(f"    Phase: {'Non-Geometric' if abs(sigma_min_g) < 1e-6 else 'Geometric'}")
    print()

    # Critical point
    config_crit = CondensateConfig(rank=3, mass_squared=0.0, coupling=0.1)
    V_crit = EffectivePotential(config_crit)
    sigma_min_c, V_min_c = V_crit.global_minimum()

    print(f"  Critical Point (m^2 = {config_crit.mass_squared}):")
    print(f"    sigma_min = {sigma_min_c:.6f}")
    print(f"    V_min = {V_min_c:.6f}")
    print()

    # ========================================================================
    # 2. Phase Diagram Exploration
    # ========================================================================

    print("2. Computing Phase Diagram:")
    print("-" * 70)

    phase_config = CondensateConfig(
        rank=3,
        mass_range=(-2.0, 2.0),
        coupling_range=(0.01, 1.0),
        grid_resolution=30  # Reduced for speed
    )

    phase_diagram = PhaseDiagram(phase_config)
    phase_data = phase_diagram.compute_phase_diagram()

    # Count phases
    num_nongeom = np.sum(phase_data['phase_labels'] == 0)
    num_geom = np.sum(phase_data['phase_labels'] == 1)
    total_points = phase_data['phase_labels'].size

    print(f"  Grid resolution: {phase_config.grid_resolution} x {phase_config.grid_resolution}")
    print(f"  Non-Geometric phase: {num_nongeom}/{total_points} points ({100*num_nongeom/total_points:.1f}%)")
    print(f"  Geometric phase: {num_geom}/{total_points} points ({100*num_geom/total_points:.1f}%)")
    print(f"  Critical line points: {len(phase_data['critical_line'])}")
    print()

    # Sample susceptibility near critical line
    if len(phase_data['critical_line']) > 0:
        m2_crit, lambda_crit = phase_data['critical_line'][len(phase_data['critical_line'])//2]
        chi_crit = phase_diagram.susceptibility(m2_crit, lambda_crit)
        print(f"  Susceptibility at critical point:")
        print(f"    (m^2, lambda) = ({m2_crit:.3f}, {lambda_crit:.3f})")
        print(f"    chi = {chi_crit:.6f}")
        print()

    # ========================================================================
    # 3. Gross-Pitaevskii Dynamics
    # ========================================================================

    print("3. Testing Gross-Pitaevskii Evolution:")
    print("-" * 70)

    gp_config = CondensateConfig(
        rank=3,
        mass_squared=-0.5,
        coupling=0.05,
        num_time_steps=100,
        time_max=20.0
    )

    gp = GrossPitaevskiiEvolution(gp_config)

    # Initial condition: small perturbation
    initial_sigma = np.array([0.5 + 0.1j])

    evolution = gp.evolve(initial_sigma)

    print(f"  Evolution parameters:")
    print(f"    Time range: 0 to {gp_config.time_max}")
    print(f"    Time steps: {gp_config.num_time_steps}")
    print(f"    Initial sigma: {initial_sigma[0]:.6f}")
    print()
    print(f"  Final state:")
    print(f"    sigma(t_final) = {evolution['sigma_real'][-1]:.6f} + {evolution['sigma_imag'][-1]:.6f}i")
    print(f"    Density: rho = {evolution['density'][-1]:.6f}")
    print(f"    Energy: E = {evolution['energy'][-1]:.6f}")
    print()

    # ========================================================================
    # 4. Cosmological Interpretation
    # ========================================================================

    print("4. Testing Cosmological Interpretation:")
    print("-" * 70)

    cosmo = CosmologicalCondensate(gp_config)

    sigma_complex = evolution['sigma_real'] + 1j * evolution['sigma_imag']
    cosmo_data = cosmo.effective_metric_friedmann(sigma_complex, evolution['times'])

    print(f"  Emergent FLRW cosmology:")
    print(f"    Initial scale factor: a(0) = {cosmo_data['scale_factor'][0]:.6f}")
    print(f"    Final scale factor: a(t_f) = {cosmo_data['scale_factor'][-1]:.6f}")
    print(f"    Initial Hubble: H(0) = {cosmo_data['hubble'][0]:.6f}")
    print(f"    Final Hubble: H(t_f) = {cosmo_data['hubble'][-1]:.6f}")
    print(f"    Mean EoS parameter: <w> = {np.mean(cosmo_data['eos_parameter']):.6f}")
    print()

    # Check for bounce
    bounce = cosmo.detect_bounce(cosmo_data['scale_factor'], evolution['times'])
    if bounce and bounce['detected']:
        print(f"  Bounce detected:")
        print(f"    Bounce time: t_bounce = {bounce['bounce_time']:.6f}")
        print(f"    Minimum scale factor: a_min = {bounce['bounce_scale_factor']:.6f}")
    else:
        print(f"  No bounce detected (expanding or contracting)")
    print()

    # Page curve analog
    page_data = cosmo.compute_page_curve_analog(sigma_complex, evolution['times'])

    print(f"  Page curve analog:")
    print(f"    Initial entropy: S(0) = {page_data['entropy'][0]:.6f}")
    print(f"    Final entropy: S(t_f) = {page_data['entropy'][-1]:.6f}")
    print(f"    Max entropy: S_max = {np.max(page_data['entropy']):.6f}")
    print(f"    Effective quanta: N_eff(0) = {page_data['effective_quanta'][0]:.6f}")
    print()

    # ========================================================================
    # Export Data
    # ========================================================================

    print("5. Exporting Results:")
    print("-" * 70)

    export_data = {
        'phase_diagram': {
            'mass_range': list(phase_config.mass_range),
            'coupling_range': list(phase_config.coupling_range),
            'resolution': phase_config.grid_resolution,
            'critical_line': phase_data['critical_line'].tolist() if len(phase_data['critical_line']) > 0 else [],
            'num_geometric_points': int(num_geom),
            'num_nongeometric_points': int(num_nongeom)
        },
        'dynamics': {
            'times': evolution['times'].tolist(),
            'sigma_real': evolution['sigma_real'].tolist(),
            'sigma_imag': evolution['sigma_imag'].tolist(),
            'density': evolution['density'].tolist(),
            'energy': evolution['energy'].tolist()
        },
        'cosmology': {
            'scale_factor': cosmo_data['scale_factor'].tolist(),
            'hubble': cosmo_data['hubble'].tolist(),
            'eos_parameter': cosmo_data['eos_parameter'].tolist(),
            'bounce_detected': bounce is not None and bounce.get('detected', False),
            'bounce_time': bounce['bounce_time'] if bounce and bounce.get('detected') else None
        },
        'page_curve': {
            'entropy': page_data['entropy'].tolist(),
            'effective_quanta': page_data['effective_quanta'].tolist()
        }
    }

    output_file = 'gft_condensate_results.json'
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"  Exported to: {output_file}")
    print()

    print("[OK] GFT condensate dynamics ready!")
    print()
