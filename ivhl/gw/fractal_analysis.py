"""
Fractal and Harmonic Analysis for LIGO-Inspired GW Lattice

Implements analysis tools for detecting:
- Fractal dimension in log-space field slicing
- Harmonic series and mathematical constant residues
- Lattice detectors for stable nodal sets
- Self-similar scaling patterns

Conceptual Foundation:
- LIGO fractal harmonics → iVHL field intensity self-similarity
- Log-space analysis reveals scale-invariant structure
- Mathematical constants emerge as harmonic residues
- Lattice persistence indicates non-random spacetime structure

Author: iVHL Framework (LIGO Integration)
Date: 2025-12-15
"""

import numpy as np
import torch
import torch.fft as fft
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.optimize import curve_fit


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FractalAnalysisConfig:
    """Configuration for fractal and harmonic analysis"""

    # Fractal dimension
    box_sizes: List[int] = None  # Box sizes for box-counting
    min_box_size: int = 2
    max_box_size: int = 128

    # Harmonic detection
    fft_resolution: int = 8192  # FFT size
    peak_threshold: float = 3.0  # SNR threshold for peak detection
    harmonic_tolerance: float = 0.01  # Fractional tolerance for harmonic matching

    # Mathematical constants
    test_constants: List[Tuple[str, float]] = None  # (name, value) pairs

    # Log-space slicing
    log_bins: int = 50  # Bins for log-space histograms
    log_min: float = 1e-6  # Minimum field value

    # Lattice detection
    clustering_threshold: float = 0.1  # Distance threshold for nodal clustering
    persistence_threshold: float = 0.8  # Similarity threshold for persistence

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.box_sizes is None:
            # Logarithmically spaced box sizes
            self.box_sizes = [
                int(s) for s in np.logspace(
                    np.log10(self.min_box_size),
                    np.log10(self.max_box_size),
                    10
                )
            ]

        if self.test_constants is None:
            # Standard mathematical constants
            self.test_constants = [
                ('π', np.pi),
                ('e', np.e),
                ('φ', (1 + np.sqrt(5)) / 2),  # Golden ratio
                ('√2', np.sqrt(2)),
                ('√3', np.sqrt(3)),
                ('ln(2)', np.log(2)),
                ('ζ(3)', 1.2020569),  # Apéry's constant
                ('γ', 0.5772156649),  # Euler-Mascheroni constant
            ]


# ============================================================================
# Fractal Dimension Computation
# ============================================================================

class FractalDimensionAnalyzer:
    """
    Compute fractal dimension using box-counting method

    For field isosurface or nodal set:
    D_box = lim_{ε→0} log(N(ε)) / log(1/ε)

    where N(ε) = number of boxes of size ε containing the set
    """

    def __init__(self, config: FractalAnalysisConfig):
        self.config = config
        self.device = torch.device(config.device)

    def box_count(
        self,
        data: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[List[int], List[int]]:
        """
        Box-counting algorithm for 3D scalar field

        Args:
            data: (X, Y, Z) field values
            threshold: Isosurface threshold

        Returns:
            box_sizes: List of box sizes
            counts: Number of boxes at each size
        """
        # Binarize data
        binary = (data > threshold).float()

        box_sizes = self.config.box_sizes
        counts = []

        for box_size in box_sizes:
            # Coarse-grain by max-pooling
            if box_size == 1:
                # Count occupied voxels
                count = binary.sum().item()
            else:
                # Downsample by taking max in each box
                pooled = torch.nn.functional.max_pool3d(
                    binary.unsqueeze(0).unsqueeze(0),
                    kernel_size=box_size,
                    stride=box_size
                )
                # Count occupied boxes
                count = (pooled > 0).sum().item()

            counts.append(int(count))

        return box_sizes, counts

    def compute_fractal_dimension(
        self,
        box_sizes: List[int],
        counts: List[int]
    ) -> Dict[str, float]:
        """
        Compute fractal dimension from box-counting data

        Fits: log(N) = D * log(1/ε) + c

        Returns:
            results: {
                'fractal_dimension': D,
                'r_squared': Fit quality,
                'slope': Fitted slope,
                'intercept': Fitted intercept
            }
        """
        # Convert to log-log
        eps = np.array(box_sizes, dtype=np.float64)
        N = np.array(counts, dtype=np.float64)

        # Remove zeros
        valid = N > 0
        eps = eps[valid]
        N = N[valid]

        if len(eps) < 3:
            return {
                'fractal_dimension': np.nan,
                'r_squared': 0.0,
                'slope': np.nan,
                'intercept': np.nan
            }

        log_eps = np.log(eps)
        log_N = np.log(N)

        # Linear fit: log(N) = slope * log(eps) + intercept
        # slope = -D (negative because log(1/eps) = -log(eps))
        coeffs = np.polyfit(log_eps, log_N, deg=1)
        slope, intercept = coeffs

        # Fractal dimension
        D = -slope

        # R-squared
        fit_values = slope * log_eps + intercept
        ss_res = np.sum((log_N - fit_values) ** 2)
        ss_tot = np.sum((log_N - log_N.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return {
            'fractal_dimension': float(D),
            'r_squared': float(r_squared),
            'slope': float(slope),
            'intercept': float(intercept)
        }

    def analyze_field(
        self,
        field: torch.Tensor,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, any]:
        """
        Analyze fractal dimension at multiple isosurface thresholds

        Args:
            field: (X, Y, Z) scalar field
            thresholds: List of isosurface values (default: quantiles)

        Returns:
            results: Fractal dimensions and fits for each threshold
        """
        if thresholds is None:
            # Use quantiles
            thresholds = [
                torch.quantile(field, q).item()
                for q in [0.25, 0.5, 0.75, 0.9]
            ]

        results = {}

        for i, thresh in enumerate(thresholds):
            box_sizes, counts = self.box_count(field, threshold=thresh)
            dim_results = self.compute_fractal_dimension(box_sizes, counts)

            results[f'threshold_{i}'] = {
                'threshold_value': thresh,
                **dim_results
            }

        return results


# ============================================================================
# Harmonic Series Detection
# ============================================================================

class HarmonicSeriesDetector:
    """
    Detect harmonic series and mathematical constant residues in waveforms

    Uses FFT peak detection and rational approximation to identify:
    - Integer harmonic series: f_n = n * f_0
    - Constant-related frequencies: f_n = f_0 * c (c = π, e, φ, etc.)
    - Fractal self-similarity in frequency domain
    """

    def __init__(self, config: FractalAnalysisConfig):
        self.config = config
        self.device = torch.device(config.device)

    def compute_power_spectrum(
        self,
        signal_data: torch.Tensor,
        sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density

        Args:
            signal_data: (T,) time-series
            sampling_rate: Hz

        Returns:
            frequencies: (N,) frequency array
            power: (N,) power spectral density
        """
        # FFT
        n = self.config.fft_resolution
        fft_result = torch.fft.rfft(signal_data, n=n)

        # Power spectrum
        power = torch.abs(fft_result) ** 2
        power = power.cpu().numpy()

        # Frequency array
        frequencies = np.fft.rfftfreq(n, d=1/sampling_rate)

        return frequencies, power

    def detect_peaks(
        self,
        frequencies: np.ndarray,
        power: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Detect significant peaks in power spectrum

        Args:
            frequencies: (N,) frequency array
            power: (N,) power values

        Returns:
            peaks: List of (frequency, power) tuples
        """
        # Normalize power
        power_norm = power / (power.mean() + 1e-10)

        # Find peaks above threshold
        peak_indices, properties = signal.find_peaks(
            power_norm,
            height=self.config.peak_threshold,
            distance=5  # Minimum separation
        )

        peaks = [
            (frequencies[i], power[i])
            for i in peak_indices
        ]

        # Sort by power
        peaks.sort(key=lambda x: x[1], reverse=True)

        return peaks

    def detect_harmonic_series(
        self,
        peaks: List[Tuple[float, float]],
        max_harmonics: int = 10
    ) -> Dict[str, any]:
        """
        Detect integer harmonic series f_n = n * f_0

        Args:
            peaks: List of (frequency, power) tuples
            max_harmonics: Maximum harmonic number to check

        Returns:
            results: Detected fundamental frequency and harmonic series
        """
        if len(peaks) < 2:
            return {
                'fundamental_frequency': None,
                'harmonics_detected': [],
                'harmonic_ratio': 0.0
            }

        # Try each peak as potential fundamental
        best_f0 = None
        best_score = 0

        for f0, _ in peaks[:5]:  # Test top 5 peaks
            # Check how many peaks match harmonics
            matches = 0

            for n in range(1, max_harmonics + 1):
                f_expected = n * f0

                # Check if any peak is close
                for f_peak, _ in peaks:
                    if abs(f_peak - f_expected) / f_expected < self.config.harmonic_tolerance:
                        matches += 1
                        break

            score = matches / max_harmonics

            if score > best_score:
                best_score = score
                best_f0 = f0

        # Extract matched harmonics
        harmonics = []
        if best_f0 is not None:
            for n in range(1, max_harmonics + 1):
                f_expected = n * best_f0

                for f_peak, p_peak in peaks:
                    if abs(f_peak - f_expected) / f_expected < self.config.harmonic_tolerance:
                        harmonics.append({
                            'harmonic_number': n,
                            'frequency': f_peak,
                            'expected_frequency': f_expected,
                            'power': p_peak
                        })
                        break

        return {
            'fundamental_frequency': best_f0,
            'harmonics_detected': harmonics,
            'harmonic_ratio': best_score
        }

    def detect_constant_residues(
        self,
        peaks: List[Tuple[float, float]],
        base_frequency: Optional[float] = None
    ) -> Dict[str, List[Dict]]:
        """
        Detect mathematical constant ratios in peak frequencies

        Tests if f_peak / f_base ≈ constant (π, e, φ, etc.)

        Args:
            peaks: List of (frequency, power) tuples
            base_frequency: Reference frequency (default: lowest peak)

        Returns:
            matches: Dict of constant_name → list of matching peaks
        """
        if base_frequency is None:
            if len(peaks) == 0:
                return {}
            base_frequency = peaks[0][0]

        matches = {name: [] for name, _ in self.config.test_constants}

        for f_peak, p_peak in peaks:
            ratio = f_peak / base_frequency

            # Test against constants
            for const_name, const_value in self.config.test_constants:
                # Check if ratio ≈ constant or 1/constant
                for r in [const_value, 1.0 / const_value]:
                    if abs(ratio - r) / r < self.config.harmonic_tolerance:
                        matches[const_name].append({
                            'frequency': f_peak,
                            'ratio': ratio,
                            'expected_ratio': r,
                            'power': p_peak,
                            'deviation': abs(ratio - r) / r
                        })

        # Filter out empty matches
        matches = {k: v for k, v in matches.items() if len(v) > 0}

        return matches


# ============================================================================
# Lattice Constant Detectors
# ============================================================================

class LatticeConstantDetector:
    """
    Detect stable lattice structures and mathematical constant clusters

    Identifies:
    - Persistent nodal sets across time/scrambling
    - Clusters at constant-related positions
    - Attractor basins in configuration space
    """

    def __init__(self, config: FractalAnalysisConfig):
        self.config = config
        self.device = torch.device(config.device)

    def detect_nodal_sets(
        self,
        field: torch.Tensor,
        threshold: float = 0.0
    ) -> torch.Tensor:
        """
        Extract nodal set (zero-crossings) from field

        Args:
            field: (X, Y, Z) scalar field
            threshold: Zero-crossing threshold

        Returns:
            nodal_points: (M, 3) positions of nodal points
        """
        # Find zero-crossing voxels
        nodal_mask = torch.abs(field) < threshold

        # Extract positions
        indices = torch.nonzero(nodal_mask, as_tuple=False)  # (M, 3)

        return indices.float()

    def cluster_nodal_points(
        self,
        nodal_points: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Cluster nodal points to identify stable structures

        Uses simple distance-based clustering

        Args:
            nodal_points: (M, 3) nodal positions

        Returns:
            clusters: List of (N_i, 3) cluster tensors
        """
        if len(nodal_points) == 0:
            return []

        # Simple greedy clustering
        points = nodal_points.clone()
        clusters = []
        threshold = self.config.clustering_threshold

        while len(points) > 0:
            # Start new cluster with first point
            seed = points[0:1]
            cluster = [seed]
            remaining = []

            for i in range(1, len(points)):
                p = points[i:i+1]

                # Distance to cluster
                dists = torch.cdist(p, torch.cat(cluster, dim=0))
                min_dist = dists.min()

                if min_dist < threshold:
                    cluster.append(p)
                else:
                    remaining.append(p)

            clusters.append(torch.cat(cluster, dim=0))

            if len(remaining) == 0:
                break
            points = torch.cat(remaining, dim=0)

        return clusters

    def detect_constant_clusters(
        self,
        positions: torch.Tensor,
        reference_point: Optional[torch.Tensor] = None
    ) -> Dict[str, List[int]]:
        """
        Detect clusters at constant-related distances from reference

        Args:
            positions: (N, 3) point positions
            reference_point: (3,) reference (default: origin)

        Returns:
            constant_clusters: Dict of constant_name → list of point indices
        """
        if reference_point is None:
            reference_point = torch.zeros(3, device=self.device)

        # Distances from reference
        dists = torch.norm(positions - reference_point.unsqueeze(0), dim=1)

        # Normalize by median
        median_dist = torch.median(dists)
        ratios = dists / median_dist

        # Test against constants
        constant_clusters = {name: [] for name, _ in self.config.test_constants}

        for i, r in enumerate(ratios):
            for const_name, const_value in self.config.test_constants:
                # Check if ratio ≈ constant
                if abs(r.item() - const_value) / const_value < self.config.harmonic_tolerance:
                    constant_clusters[const_name].append(i)

        # Filter empty
        constant_clusters = {k: v for k, v in constant_clusters.items() if len(v) > 0}

        return constant_clusters

    def persistence_across_runs(
        self,
        lattice_sequence: List[torch.Tensor]
    ) -> float:
        """
        Measure lattice persistence across multiple runs/scrambling

        Computes average pairwise similarity

        Args:
            lattice_sequence: List of (N, 3) lattice configurations

        Returns:
            persistence_score: Average similarity in [0, 1]
        """
        if len(lattice_sequence) < 2:
            return 1.0

        similarities = []

        for i in range(len(lattice_sequence)):
            for j in range(i + 1, len(lattice_sequence)):
                # Procrustes similarity
                sim = self._procrustes_similarity(
                    lattice_sequence[i],
                    lattice_sequence[j]
                )
                similarities.append(sim)

        persistence = np.mean(similarities) if similarities else 0.0

        return persistence

    def _procrustes_similarity(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor
    ) -> float:
        """Procrustes similarity (see PersistenceAnalyzer in gw_lattice_mode.py)"""
        # Center
        p1 = pos1 - pos1.mean(dim=0, keepdim=True)
        p2 = pos2 - pos2.mean(dim=0, keepdim=True)

        # SVD for optimal rotation
        H = p1.T @ p2
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T

        # Aligned
        p2_aligned = p2 @ R

        # Distance
        dist = torch.norm(p1 - p2_aligned, dim=1).mean()
        spread = torch.norm(p1, dim=1).mean()

        similarity = torch.exp(-dist / (spread + 1e-8))

        return similarity.item()


# ============================================================================
# Log-Space Analysis
# ============================================================================

class LogSpaceAnalyzer:
    """
    Analyze field in log-space for scale-invariant structure

    Detects:
    - Self-similar patterns across log-scales
    - Power-law distributions
    - Fractal layering
    """

    def __init__(self, config: FractalAnalysisConfig):
        self.config = config

    def log_histogram(
        self,
        field_values: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram in log-space

        Args:
            field_values: (N,) field intensity values

        Returns:
            bin_centers: Log-spaced bin centers
            counts: Histogram counts
        """
        # Remove non-positive
        field_positive = field_values[field_values > self.config.log_min]

        if len(field_positive) == 0:
            return np.array([]), np.array([])

        # Log-transform
        log_values = torch.log10(field_positive).cpu().numpy()

        # Histogram
        counts, bin_edges = np.histogram(log_values, bins=self.config.log_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, counts

    def detect_power_law(
        self,
        bin_centers: np.ndarray,
        counts: np.ndarray
    ) -> Dict[str, float]:
        """
        Fit power-law distribution: P(x) ∝ x^(-α)

        In log-space: log(P) = -α * log(x) + c

        Returns:
            results: {
                'exponent': α,
                'r_squared': Fit quality
            }
        """
        if len(bin_centers) < 3 or counts.sum() == 0:
            return {'exponent': np.nan, 'r_squared': 0.0}

        # Remove zeros
        valid = counts > 0
        x = bin_centers[valid]
        y = counts[valid]

        # Log-log fit
        log_x = x
        log_y = np.log(y)

        coeffs = np.polyfit(log_x, log_y, deg=1)
        alpha = -coeffs[0]

        # R-squared
        fit_values = coeffs[0] * log_x + coeffs[1]
        ss_res = np.sum((log_y - fit_values) ** 2)
        ss_tot = np.sum((log_y - log_y.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return {
            'exponent': float(alpha),
            'r_squared': float(r_squared)
        }


# ============================================================================
# Integrated Analysis Pipeline
# ============================================================================

class FractalHarmonicAnalyzer:
    """
    Integrated fractal and harmonic analysis pipeline

    Combines all analysis tools for comprehensive GW lattice characterization
    """

    def __init__(self, config: Optional[FractalAnalysisConfig] = None):
        if config is None:
            config = FractalAnalysisConfig()

        self.config = config
        self.device = torch.device(config.device)

        # Components
        self.fractal_analyzer = FractalDimensionAnalyzer(config)
        self.harmonic_detector = HarmonicSeriesDetector(config)
        self.lattice_detector = LatticeConstantDetector(config)
        self.logspace_analyzer = LogSpaceAnalyzer(config)

    def analyze_full(
        self,
        field_3d: torch.Tensor,
        strain_waveform: torch.Tensor,
        sampling_rate: float,
        lattice_history: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, any]:
        """
        Complete fractal and harmonic analysis

        Args:
            field_3d: (X, Y, Z) 3D scalar field
            strain_waveform: (T,) time-series
            sampling_rate: Hz
            lattice_history: List of (N, 3) lattice configurations

        Returns:
            results: Comprehensive analysis results
        """
        print("Running comprehensive fractal/harmonic analysis...")
        print()

        results = {}

        # 1. Fractal dimension
        print("Computing fractal dimensions...")
        fractal_results = self.fractal_analyzer.analyze_field(field_3d)
        results['fractal_dimensions'] = fractal_results
        print(f"  Fractal dimensions: {[r['fractal_dimension'] for r in fractal_results.values()]}")
        print()

        # 2. Harmonic series
        print("Detecting harmonic series...")
        frequencies, power = self.harmonic_detector.compute_power_spectrum(
            strain_waveform,
            sampling_rate
        )
        peaks = self.harmonic_detector.detect_peaks(frequencies, power)
        harmonic_series = self.harmonic_detector.detect_harmonic_series(peaks)
        constant_residues = self.harmonic_detector.detect_constant_residues(peaks)

        results['harmonic_series'] = harmonic_series
        results['constant_residues'] = constant_residues
        print(f"  Fundamental frequency: {harmonic_series['fundamental_frequency']:.2f} Hz")
        print(f"  Harmonics detected: {len(harmonic_series['harmonics_detected'])}")
        print(f"  Constant residues: {list(constant_residues.keys())}")
        print()

        # 3. Lattice structures
        print("Detecting lattice structures...")
        nodal_points = self.lattice_detector.detect_nodal_sets(field_3d)
        clusters = self.lattice_detector.cluster_nodal_points(nodal_points)

        results['nodal_clusters'] = {
            'num_clusters': len(clusters),
            'cluster_sizes': [len(c) for c in clusters]
        }
        print(f"  Nodal clusters: {len(clusters)}")
        print()

        # 4. Persistence
        if lattice_history is not None and len(lattice_history) > 1:
            print("Measuring lattice persistence...")
            persistence = self.lattice_detector.persistence_across_runs(lattice_history)
            results['persistence_score'] = persistence
            print(f"  Persistence score: {persistence:.3f}")
            print()

        # 5. Log-space analysis
        print("Analyzing log-space structure...")
        field_flat = field_3d.flatten()
        bin_centers, counts = self.logspace_analyzer.log_histogram(field_flat)
        power_law = self.logspace_analyzer.detect_power_law(bin_centers, counts)

        results['log_space'] = {
            'power_law_exponent': power_law['exponent'],
            'power_law_r_squared': power_law['r_squared']
        }
        print(f"  Power-law exponent: {power_law['exponent']:.3f} (R²={power_law['r_squared']:.3f})")
        print()

        print("Analysis complete!")

        return results


# ============================================================================
# Demo
# ============================================================================

def demo_fractal_harmonic_analysis():
    """Demonstrate fractal and harmonic analysis"""

    print("\n" + "=" * 70)
    print("FRACTAL AND HARMONIC ANALYSIS DEMO")
    print("=" * 70)
    print()

    # Generate synthetic data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 3D field with fractal structure
    X, Y, Z = 64, 64, 64
    x = torch.linspace(-1, 1, X, device=device)
    y = torch.linspace(-1, 1, Y, device=device)
    z = torch.linspace(-1, 1, Z, device=device)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    # Fractal field (sum of octaves)
    field_3d = torch.zeros_like(xx)
    for octave in range(4):
        freq = 2 ** octave
        amplitude = 0.5 ** octave
        field_3d += amplitude * torch.sin(freq * np.pi * xx) * torch.cos(freq * np.pi * yy) * torch.sin(freq * np.pi * zz)

    # Strain waveform with harmonics and constant residues
    T = 4096
    t = torch.linspace(0, 1, T, device=device)
    f0 = 100.0  # Hz

    strain = torch.zeros_like(t)
    # Fundamental + harmonics
    for n in [1, 2, 3, 4, 5]:
        strain += 0.5 ** n * torch.sin(2 * np.pi * f0 * n * t)

    # Add constant-related frequencies
    strain += 0.1 * torch.sin(2 * np.pi * f0 * np.pi * t)  # π * f0
    strain += 0.05 * torch.sin(2 * np.pi * f0 * (1 + np.sqrt(5)) / 2 * t)  # φ * f0

    sampling_rate = T  # 1 Hz effective

    # Lattice history (mock)
    lattice_history = [
        torch.randn(100, 3, device=device) + i * 0.1
        for i in range(5)
    ]

    # Analysis
    analyzer = FractalHarmonicAnalyzer()
    results = analyzer.analyze_full(
        field_3d,
        strain,
        sampling_rate,
        lattice_history
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Fractal dimensions: {results['fractal_dimensions']}")
    print(f"Harmonic series: {results['harmonic_series']}")
    print(f"Constant residues: {list(results['constant_residues'].keys())}")
    print(f"Persistence score: {results.get('persistence_score', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    demo_fractal_harmonic_analysis()
