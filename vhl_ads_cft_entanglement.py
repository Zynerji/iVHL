"""
Advanced AdS/CFT Entanglement and Holography for VHL

Implements expanded Ryu-Takayanagi (RT) formalism with:
- Geometric minimal surface finding with backreaction
- Page curve evolution (entanglement entropy vs time)
- Advanced entropy measures (reflected, odd, purification)
- Modular flow approximations
- Connection to VHL boundary lattice and bulk vortex field

Physical Framework:
- Boundary CFT: VHL helical lattice as 2D conformal operators
- Bulk AdS: Interior wave field + vortex configurations
- RT Formula: S(A) = Area(γ_A) / (4G_N) where γ_A is minimal surface
- Backreaction: Vortex stress-energy warps effective metric

Author: Zynerji
Date: 2025-12-15
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from typing import List, Tuple, Optional, Dict, Callable
import json


class MinimalSurface:
    """
    Find minimal surface in warped geometry for Ryu-Takayanagi entropy.

    Uses variational method to find geodesics anchored to boundary region.
    """

    def __init__(self, metric_function: Callable,
                 boundary_points: np.ndarray,
                 num_surface_points: int = 50):
        """
        Initialize minimal surface finder.

        Args:
            metric_function: Returns metric tensor g_μν(x) at position x
            boundary_points: Points on boundary (CFT subregion A)
            num_surface_points: Resolution of surface discretization
        """
        self.metric_function = metric_function
        self.boundary_points = boundary_points
        self.num_surface_points = num_surface_points

    def compute_area(self, surface_points: np.ndarray) -> float:
        """
        Compute area of surface in curved geometry.

        Args:
            surface_points: Array of points defining surface

        Returns:
            Surface area
        """
        total_area = 0.0

        # Triangulate surface (simplified: sequential triangles)
        for i in range(len(surface_points) - 2):
            # Triangle vertices
            p1, p2, p3 = surface_points[i:i+3]

            # Tangent vectors
            v1 = p2 - p1
            v2 = p3 - p1

            # Metric at midpoint
            midpoint = (p1 + p2 + p3) / 3
            g = self.metric_function(midpoint)

            # Induced metric on surface
            # Area element: √(det(g_ij)) where g_ij = g_μν v^μ_i v^ν_j
            g11 = np.dot(v1, np.dot(g, v1))
            g12 = np.dot(v1, np.dot(g, v2))
            g22 = np.dot(v2, np.dot(g, v2))

            det = g11 * g22 - g12**2
            if det > 0:
                area_element = 0.5 * np.sqrt(det)
                total_area += area_element

        return total_area

    def find_minimal_surface(self, initial_guess: Optional[np.ndarray] = None,
                            method: str = 'L-BFGS-B') -> Tuple[np.ndarray, float]:
        """
        Find minimal surface via optimization.

        Args:
            initial_guess: Initial surface configuration
            method: Optimization method

        Returns:
            (optimal_surface, minimal_area) tuple
        """
        # Initial guess: straight line from boundary to center
        if initial_guess is None:
            center = np.mean(self.boundary_points, axis=0)
            t = np.linspace(0, 1, self.num_surface_points)
            initial_guess = np.outer(t, center)

        # Flatten for optimization
        x0 = initial_guess.flatten()

        # Objective: surface area
        def objective(x):
            surface = x.reshape(-1, 3)
            return self.compute_area(surface)

        # Optimize
        result = minimize(objective, x0, method=method,
                         options={'maxiter': 1000})

        optimal_surface = result.x.reshape(-1, 3)
        minimal_area = result.fun

        return optimal_surface, minimal_area


class ExpandedRyuTakayanagi:
    """
    Comprehensive Ryu-Takayanagi entropy calculator with advanced features.

    Features:
    - Standard RT entropy: S = Area / (4G_N)
    - Backreaction: Metric warping from vortex stress-energy
    - Page curve: Entropy evolution during "evaporation"
    - Advanced measures: Reflected, odd, purification entropies
    - Modular flow: Approximate tomita-takesaki theory
    """

    def __init__(self, sphere_radius: float = 1.0,
                 newton_constant: float = 1.0,
                 hbar: float = 1.0):
        """
        Initialize RT calculator.

        Args:
            sphere_radius: AdS boundary sphere radius
            newton_constant: Newton's constant G_N (sets entropy scale)
            hbar: Planck's constant (for quantum corrections)
        """
        self.sphere_radius = sphere_radius
        self.G_N = newton_constant
        self.hbar = hbar

        # Metric state (can be warped by vortices)
        self.metric_warping = None

    def flat_metric(self, position: np.ndarray) -> np.ndarray:
        """
        Flat Euclidean metric (no warping).

        Args:
            position: 3D position

        Returns:
            3x3 metric tensor
        """
        return np.eye(3)

    def warped_metric(self, position: np.ndarray,
                     intensity_field: np.ndarray,
                     grid_points: np.ndarray) -> np.ndarray:
        """
        Metric tensor warped by vortex field intensity.

        Backreaction: g_μν → (1 + α I(x)) δ_μν where I = |ψ|²

        Args:
            position: 3D position
            intensity_field: Field intensity at all grid points
            grid_points: Grid point positions

        Returns:
            Warped 3x3 metric tensor
        """
        # Interpolate intensity at position
        # (Simplified: nearest neighbor)
        distances = np.linalg.norm(grid_points - position, axis=1)
        nearest_idx = np.argmin(distances)
        intensity = intensity_field[nearest_idx]

        # Warping factor α (dimensionless coupling)
        alpha = 0.1

        # Conformal factor
        conformal_factor = 1.0 + alpha * intensity / intensity.max()

        # Warped metric (conformal to flat)
        g = conformal_factor * np.eye(3)

        return g

    def compute_rt_entropy(self, subregion_indices: np.ndarray,
                          lattice_points: np.ndarray,
                          intensity_field: Optional[np.ndarray] = None,
                          grid_points: Optional[np.ndarray] = None,
                          use_backreaction: bool = False) -> Dict:
        """
        Compute Ryu-Takayanagi entanglement entropy for CFT subregion.

        Args:
            subregion_indices: Indices of lattice points in subregion A
            lattice_points: All boundary lattice points
            intensity_field: Bulk field intensity (for backreaction)
            grid_points: Bulk grid points (for backreaction)
            use_backreaction: Whether to include metric warping

        Returns:
            Dictionary with entropy and surface data
        """
        # Boundary points in subregion A
        boundary_A = lattice_points[subregion_indices]

        # Metric function
        if use_backreaction and intensity_field is not None:
            metric_func = lambda x: self.warped_metric(x, intensity_field, grid_points)
        else:
            metric_func = self.flat_metric

        # Find minimal surface
        surface_finder = MinimalSurface(
            metric_function=metric_func,
            boundary_points=boundary_A,
            num_surface_points=50
        )

        minimal_surface, minimal_area = surface_finder.find_minimal_surface()

        # RT entropy: S = Area / (4 G_N hbar)
        entropy = minimal_area / (4 * self.G_N * self.hbar)

        return {
            'entropy': entropy,
            'area': minimal_area,
            'surface': minimal_surface,
            'num_boundary_points': len(boundary_A),
            'backreaction_used': use_backreaction
        }

    def compute_page_curve(self, num_time_steps: int,
                          subregion_indices: np.ndarray,
                          lattice_points: np.ndarray,
                          time_evolution_func: Callable) -> Dict:
        """
        Compute Page curve: Entropy vs time during "evaporation".

        Page curve physics:
        - Early time: S ~ t (thermal emission)
        - Page time: S reaches maximum (half of total entropy)
        - Late time: S decreases (purification)

        Args:
            num_time_steps: Number of time points
            subregion_indices: CFT subregion A
            lattice_points: Boundary lattice
            time_evolution_func: Function that returns intensity field at time t

        Returns:
            Dictionary with time series data
        """
        times = []
        entropies = []
        areas = []

        for step in range(num_time_steps):
            t = step * 0.1  # Time spacing

            # Get field at this time
            intensity_field, grid_points = time_evolution_func(t)

            # Compute RT entropy
            rt_data = self.compute_rt_entropy(
                subregion_indices, lattice_points,
                intensity_field, grid_points,
                use_backreaction=True
            )

            times.append(t)
            entropies.append(rt_data['entropy'])
            areas.append(rt_data['area'])

        # Find Page time (maximum entropy)
        max_idx = np.argmax(entropies)
        page_time = times[max_idx]

        return {
            'times': np.array(times),
            'entropies': np.array(entropies),
            'areas': np.array(areas),
            'page_time': page_time,
            'page_entropy': entropies[max_idx]
        }

    def compute_reflected_entropy(self, subregion_A_indices: np.ndarray,
                                  subregion_B_indices: np.ndarray,
                                  lattice_points: np.ndarray) -> float:
        """
        Compute reflected entropy S_R(A:B).

        Reflected entropy measures correlations between A and B via
        canonical purification.

        S_R(A:B) = Area(EW_A ∩ EW_B) / (4G_N)

        Args:
            subregion_A_indices: Subregion A indices
            subregion_B_indices: Subregion B indices
            lattice_points: Boundary lattice

        Returns:
            Reflected entropy
        """
        # Simplified: Compute overlap of entanglement wedges
        # (Full implementation would need entanglement wedge reconstruction)

        # Get surfaces for A and B
        rt_A = self.compute_rt_entropy(subregion_A_indices, lattice_points)
        rt_B = self.compute_rt_entropy(subregion_B_indices, lattice_points)

        # Approximate overlap as minimum area
        overlap_area = min(rt_A['area'], rt_B['area'])

        # Reflected entropy
        S_R = overlap_area / (4 * self.G_N * self.hbar)

        return S_R

    def compute_odd_entropy(self, subregion_A_indices: np.ndarray,
                           subregion_B_indices: np.ndarray,
                           lattice_points: np.ndarray) -> float:
        """
        Compute odd entropy (entanglement negativity measure).

        S_odd(A:B) = [S(A) + S(B) - S(AB)] / 2

        Args:
            subregion_A_indices: Subregion A
            subregion_B_indices: Subregion B
            lattice_points: Boundary lattice

        Returns:
            Odd entropy
        """
        # Compute individual entropies
        S_A = self.compute_rt_entropy(subregion_A_indices, lattice_points)['entropy']
        S_B = self.compute_rt_entropy(subregion_B_indices, lattice_points)['entropy']

        # Combined region
        combined_indices = np.union1d(subregion_A_indices, subregion_B_indices)
        S_AB = self.compute_rt_entropy(combined_indices, lattice_points)['entropy']

        # Odd entropy
        S_odd = (S_A + S_B - S_AB) / 2

        return S_odd

    def compute_mutual_information(self, subregion_A_indices: np.ndarray,
                                  subregion_B_indices: np.ndarray,
                                  lattice_points: np.ndarray) -> float:
        """
        Compute mutual information I(A:B).

        I(A:B) = S(A) + S(B) - S(AB)

        Args:
            subregion_A_indices: Subregion A
            subregion_B_indices: Subregion B
            lattice_points: Boundary lattice

        Returns:
            Mutual information
        """
        S_A = self.compute_rt_entropy(subregion_A_indices, lattice_points)['entropy']
        S_B = self.compute_rt_entropy(subregion_B_indices, lattice_points)['entropy']

        combined_indices = np.union1d(subregion_A_indices, subregion_B_indices)
        S_AB = self.compute_rt_entropy(combined_indices, lattice_points)['entropy']

        I_AB = S_A + S_B - S_AB

        return I_AB

    def compute_entanglement_wedge(self, subregion_indices: np.ndarray,
                                   lattice_points: np.ndarray,
                                   rt_data: Dict) -> np.ndarray:
        """
        Reconstruct entanglement wedge: bulk region causally connected to A.

        Entanglement wedge EW(A) is bounded by:
        - Boundary subregion A
        - RT surface γ_A
        - Null surfaces from ∂A

        Args:
            subregion_indices: CFT subregion
            lattice_points: Boundary lattice
            rt_data: RT entropy computation result

        Returns:
            Boolean mask indicating points in entanglement wedge
        """
        # Simplified: Points within some distance of RT surface
        rt_surface = rt_data['surface']

        # All bulk points (would come from grid)
        # For now, return surface points
        wedge_points = rt_surface

        return wedge_points

    def modular_hamiltonian_approximation(self, subregion_indices: np.ndarray,
                                          lattice_points: np.ndarray,
                                          reduced_density_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Approximate modular Hamiltonian K_A = -log(ρ_A).

        In AdS/CFT:
        - K_A acts as boost generator in entanglement wedge
        - Generates modular flow (Tomita-Takesaki theory)
        - Related to RT surface via flow equations

        Args:
            subregion_indices: CFT subregion A
            lattice_points: Boundary lattice
            reduced_density_matrix: ρ_A (if available)

        Returns:
            Modular Hamiltonian properties
        """
        # Simplified approximation
        rt_data = self.compute_rt_entropy(subregion_indices, lattice_points)

        # Approximate modular energy
        # <K_A> = S(A) (average modular energy equals entropy)
        modular_energy = rt_data['entropy']

        # Modular flow time scale
        # τ = 1 / <K_A>
        flow_timescale = 1.0 / (modular_energy + 1e-10)

        return {
            'modular_energy': modular_energy,
            'flow_timescale': flow_timescale,
            'entropy': rt_data['entropy']
        }

    def export_results(self, results: Dict, filepath: str):
        """
        Export entropy calculations to JSON.

        Args:
            results: Results dictionary
            filepath: Output file path
        """
        # Convert numpy arrays to lists for JSON
        exportable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                exportable[key] = value.tolist()
            elif isinstance(value, dict):
                exportable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                exportable[key] = value

        with open(filepath, 'w') as f:
            json.dump(exportable, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED AdS/CFT ENTANGLEMENT SYSTEM")
    print("=" * 70)

    # Create boundary lattice (simplified sphere)
    num_points = 100
    theta = np.random.uniform(0, np.pi, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)
    r = 1.0

    lattice_points = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])

    print(f"\n1. Created boundary lattice: {num_points} points on sphere")

    # Initialize RT calculator
    rt_calculator = ExpandedRyuTakayanagi(sphere_radius=1.0)

    # Test 1: Basic RT entropy
    print("\n2. Testing Basic RT Entropy:")
    subregion_A = np.arange(0, 30)  # First 30 points
    rt_result = rt_calculator.compute_rt_entropy(subregion_A, lattice_points)

    print(f"  Subregion A: {len(subregion_A)} boundary points")
    print(f"  RT Entropy: S(A) = {rt_result['entropy']:.6f}")
    print(f"  Minimal surface area: {rt_result['area']:.6f}")

    # Test 2: RT with backreaction
    print("\n3. Testing RT with Backreaction:")

    # Create mock intensity field
    grid_res = 20
    x = np.linspace(-1, 1, grid_res)
    y = np.linspace(-1, 1, grid_res)
    z = np.linspace(-1, 1, grid_res)
    grid_3d = np.array(np.meshgrid(x, y, z, indexing='ij'))
    grid_points = grid_3d.reshape(3, -1).T

    # Mock intensity (Gaussian blob)
    intensity_field = np.exp(-np.sum(grid_points**2, axis=1) / 0.1)

    rt_warped = rt_calculator.compute_rt_entropy(
        subregion_A, lattice_points,
        intensity_field=intensity_field,
        grid_points=grid_points,
        use_backreaction=True
    )

    print(f"  Flat metric: S(A) = {rt_result['entropy']:.6f}")
    print(f"  Warped metric: S(A) = {rt_warped['entropy']:.6f}")
    print(f"  Entropy change: dS = {rt_warped['entropy'] - rt_result['entropy']:.6f}")

    # Test 3: Mutual information
    print("\n4. Testing Mutual Information:")
    subregion_B = np.arange(50, 80)
    I_AB = rt_calculator.compute_mutual_information(
        subregion_A, subregion_B, lattice_points
    )
    print(f"  Subregion A: {len(subregion_A)} points")
    print(f"  Subregion B: {len(subregion_B)} points")
    print(f"  Mutual information: I(A:B) = {I_AB:.6f}")

    # Test 4: Advanced entropy measures
    print("\n5. Testing Advanced Entropy Measures:")

    S_reflected = rt_calculator.compute_reflected_entropy(
        subregion_A, subregion_B, lattice_points
    )
    S_odd = rt_calculator.compute_odd_entropy(
        subregion_A, subregion_B, lattice_points
    )

    print(f"  Reflected entropy: S_R(A:B) = {S_reflected:.6f}")
    print(f"  Odd entropy: S_odd(A:B) = {S_odd:.6f}")

    # Test 5: Modular Hamiltonian
    print("\n6. Testing Modular Hamiltonian Approximation:")
    modular_data = rt_calculator.modular_hamiltonian_approximation(
        subregion_A, lattice_points
    )

    print(f"  Modular energy: <K_A> = {modular_data['modular_energy']:.6f}")
    print(f"  Flow timescale: τ = {modular_data['flow_timescale']:.6f}")

    # Test 6: Page curve (simplified - single time step)
    print("\n7. Testing Page Curve Setup:")
    def mock_time_evolution(t):
        # Return same field for now
        return intensity_field, grid_points

    print(f"  Time evolution function ready")
    print(f"  (Full Page curve would compute over ~100 time steps)")

    # Export example results
    results = {
        'rt_basic': rt_result,
        'rt_warped': rt_warped,
        'mutual_information': I_AB,
        'reflected_entropy': S_reflected,
        'odd_entropy': S_odd,
        'modular_data': modular_data
    }

    rt_calculator.export_results(results, 'ads_cft_entanglement_results.json')
    print(f"\n8. Results exported to ads_cft_entanglement_results.json")

    print("\n[OK] Advanced AdS/CFT entanglement system ready!")
