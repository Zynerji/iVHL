"""
Performance Benchmarks for iVHL Compiled Operations

Compares eager PyTorch vs torch.compile() performance across all critical
operations: field superposition, tensor contractions, GFT condensate, etc.

Provides detailed timing statistics and speedup factors to validate the
benefits of torch.compile() on H100/A100+ hardware.

Usage:
    python benchmarks.py                    # Run all benchmarks
    python benchmarks.py --operation field  # Run specific operation
    python benchmarks.py --size large       # Use large problem sizes
    python benchmarks.py --export results.json  # Export results

Author: iVHL Framework
Date: 2025-12-15
"""

import torch
import time
import json
import argparse
from typing import Dict, List, Tuple
import sys

# Import utilities and compiled operations
try:
    from utils.device import (
        get_device, get_compile_mode, set_compile_mode,
        SimpleProfiler, get_compile_stats, CompileMode
    )
    from compiled_ops import (
        compute_field_superposition,
        compute_vortex_field,
        compute_intensity,
        contract_mera_layer,
        compute_entanglement_entropy_svd,
        compute_condensate_potential,
        evolve_condensate_gp,
        normalize_field
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    MODULES_AVAILABLE = False
    sys.exit(1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

BENCHMARK_CONFIGS = {
    'small': {
        'field_grid': 512,
        'field_sources': 10,
        'vortex_grid': 512,
        'vortex_count': 3,
        'mera_tensors': 32,
        'mera_bond_dim': 4,
        'condensate_grid': 16,
        'iterations': 100
    },
    'medium': {
        'field_grid': 2048,
        'field_sources': 20,
        'vortex_grid': 2048,
        'vortex_count': 5,
        'mera_tensors': 64,
        'mera_bond_dim': 8,
        'condensate_grid': 32,
        'iterations': 50
    },
    'large': {
        'field_grid': 8192,
        'field_sources': 50,
        'vortex_grid': 8192,
        'vortex_count': 10,
        'mera_tensors': 128,
        'mera_bond_dim': 16,
        'condensate_grid': 64,
        'iterations': 20
    },
    'extreme': {
        'field_grid': 32768,
        'field_sources': 100,
        'vortex_grid': 32768,
        'vortex_count': 20,
        'mera_tensors': 256,
        'mera_bond_dim': 32,
        'condensate_grid': 128,
        'iterations': 10
    }
}


# ============================================================================
# Benchmark Utilities
# ============================================================================

class BenchmarkResult:
    """Store and analyze benchmark results"""

    def __init__(self, name: str, eager_times: List[float], compiled_times: List[float]):
        self.name = name
        self.eager_times = eager_times
        self.compiled_times = compiled_times

        # Statistics
        self.eager_mean = sum(eager_times) / len(eager_times)
        self.eager_std = (sum((t - self.eager_mean)**2 for t in eager_times) / len(eager_times))**0.5

        self.compiled_mean = sum(compiled_times) / len(compiled_times)
        self.compiled_std = (sum((t - self.compiled_mean)**2 for t in compiled_times) / len(compiled_times))**0.5

        self.speedup = self.eager_mean / self.compiled_mean if self.compiled_mean > 0 else 1.0

    def __str__(self):
        return (f"{self.name}:\n"
                f"  Eager:    {self.eager_mean:.3f} ± {self.eager_std:.3f} ms\n"
                f"  Compiled: {self.compiled_mean:.3f} ± {self.compiled_std:.3f} ms\n"
                f"  Speedup:  {self.speedup:.2f}x")

    def to_dict(self):
        return {
            'name': self.name,
            'eager_mean_ms': self.eager_mean,
            'eager_std_ms': self.eager_std,
            'compiled_mean_ms': self.compiled_mean,
            'compiled_std_ms': self.compiled_std,
            'speedup': self.speedup
        }


def warm_up(fn, *args, n_warmup=5, **kwargs):
    """Warm up function (trigger compilation if compiled)"""
    for _ in range(n_warmup):
        _ = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def benchmark_function(fn, args, kwargs, iterations=100, device=None):
    """
    Benchmark a function

    Returns:
        List of times in milliseconds
    """
    times = []

    for _ in range(iterations):
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            _ = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    return times


# ============================================================================
# Individual Benchmarks
# ============================================================================

def benchmark_field_superposition(config: dict, device: torch.device) -> BenchmarkResult:
    """Benchmark field superposition (eager vs compiled)"""
    print("  Running field superposition benchmark...")

    N_grid = config['field_grid']
    N_sources = config['field_sources']
    iterations = config['iterations']

    # Generate data
    grid_points = torch.randn(N_grid, 3, device=device)
    sources = torch.randn(N_sources, 3, device=device) * 0.5
    frequencies = torch.ones(N_sources, device=device)
    amplitudes = torch.ones(N_sources, device=device)
    phases = torch.zeros(N_sources, device=device)

    args = (grid_points, sources, frequencies, amplitudes, phases)
    kwargs = {'t': 0.0}

    # Warm up
    warm_up(compute_field_superposition, *args, **kwargs)

    # Benchmark compiled version
    compiled_times = benchmark_function(
        compute_field_superposition, args, kwargs, iterations, device
    )

    # Create eager version (get original if compiled)
    from compiled_ops import compute_field_superposition as compiled_fn
    if hasattr(compiled_fn, '_original_func'):
        eager_fn = compiled_fn._original_func
    else:
        # Already eager
        eager_fn = compiled_fn

    # Warm up eager
    warm_up(eager_fn, *args, **kwargs)

    # Benchmark eager
    eager_times = benchmark_function(eager_fn, args, kwargs, iterations, device)

    return BenchmarkResult("Field Superposition", eager_times, compiled_times)


def benchmark_vortex_field(config: dict, device: torch.device) -> BenchmarkResult:
    """Benchmark vortex field computation"""
    print("  Running vortex field benchmark...")

    N_grid = config['vortex_grid']
    N_vortices = config['vortex_count']
    iterations = config['iterations']

    # Generate data
    grid_points = torch.randn(N_grid, 3, device=device)
    vortex_centers = torch.randn(N_vortices, 3, device=device) * 0.3
    vortex_charges = torch.randint(-2, 3, (N_vortices,), device=device, dtype=torch.float32)

    args = (grid_points, vortex_centers, vortex_charges)
    kwargs = {}

    # Warm up
    warm_up(compute_vortex_field, *args, **kwargs)

    # Benchmark compiled
    compiled_times = benchmark_function(compute_vortex_field, args, kwargs, iterations, device)

    # Eager version
    from compiled_ops import compute_vortex_field as compiled_fn
    eager_fn = compiled_fn._original_func if hasattr(compiled_fn, '_original_func') else compiled_fn

    # Warm up and benchmark eager
    warm_up(eager_fn, *args, **kwargs)
    eager_times = benchmark_function(eager_fn, args, kwargs, iterations, device)

    return BenchmarkResult("Vortex Field", eager_times, compiled_times)


def benchmark_mera_contraction(config: dict, device: torch.device) -> BenchmarkResult:
    """Benchmark MERA layer contraction"""
    print("  Running MERA contraction benchmark...")

    N_tensors = config['mera_tensors']
    d = config['mera_bond_dim']
    d_coarse = d * 2
    iterations = config['iterations']

    # Generate data
    tensors = torch.randn(N_tensors, d, d, device=device, dtype=torch.complex64)
    disentanglers = torch.randn(N_tensors // 2, 2*d*d, 2*d*d, device=device, dtype=torch.complex64)
    isometries = torch.randn(N_tensors // 2, 2*d*d, d_coarse, device=device, dtype=torch.complex64)

    args = (tensors, disentanglers, isometries)
    kwargs = {}

    # Warm up
    warm_up(contract_mera_layer, *args, **kwargs)

    # Benchmark compiled
    compiled_times = benchmark_function(contract_mera_layer, args, kwargs, iterations, device)

    # Eager version
    from compiled_ops import contract_mera_layer as compiled_fn
    eager_fn = compiled_fn._original_func if hasattr(compiled_fn, '_original_func') else compiled_fn

    # Warm up and benchmark eager
    warm_up(eager_fn, *args, **kwargs)
    eager_times = benchmark_function(eager_fn, args, kwargs, iterations, device)

    return BenchmarkResult("MERA Contraction", eager_times, compiled_times)


def benchmark_condensate_potential(config: dict, device: torch.device) -> BenchmarkResult:
    """Benchmark GFT condensate potential computation"""
    print("  Running condensate potential benchmark...")

    grid_size = config['condensate_grid']
    iterations = config['iterations']

    # Generate data
    phi_field = torch.randn(grid_size, grid_size, grid_size, device=device, dtype=torch.complex64) * 0.1

    args = (phi_field,)
    kwargs = {'mass_sq': -0.5, 'lambda_coupling': 0.1, 'rank': 3}

    # Warm up
    warm_up(compute_condensate_potential, *args, **kwargs)

    # Benchmark compiled
    compiled_times = benchmark_function(compute_condensate_potential, args, kwargs, iterations, device)

    # Eager version
    from compiled_ops import compute_condensate_potential as compiled_fn
    eager_fn = compiled_fn._original_func if hasattr(compiled_fn, '_original_func') else compiled_fn

    # Warm up and benchmark eager
    warm_up(eager_fn, *args, **kwargs)
    eager_times = benchmark_function(eager_fn, args, kwargs, iterations, device)

    return BenchmarkResult("Condensate Potential", eager_times, compiled_times)


def benchmark_gp_evolution(config: dict, device: torch.device) -> BenchmarkResult:
    """Benchmark Gross-Pitaevskii evolution step"""
    print("  Running GP evolution benchmark...")

    grid_size = config['condensate_grid']
    iterations = config['iterations']

    # Generate data
    phi_field = torch.randn(grid_size, grid_size, grid_size, device=device, dtype=torch.complex64) * 0.1

    args = (phi_field,)
    kwargs = {'dt': 0.01, 'mass_sq': -0.5, 'lambda_coupling': 0.1, 'rank': 3}

    # Warm up
    warm_up(evolve_condensate_gp, *args, **kwargs)

    # Benchmark compiled
    compiled_times = benchmark_function(evolve_condensate_gp, args, kwargs, iterations, device)

    # Eager version
    from compiled_ops import evolve_condensate_gp as compiled_fn
    eager_fn = compiled_fn._original_func if hasattr(compiled_fn, '_original_func') else compiled_fn

    # Warm up and benchmark eager
    warm_up(eager_fn, *args, **kwargs)
    eager_times = benchmark_function(eager_fn, args, kwargs, iterations, device)

    return BenchmarkResult("GP Evolution", eager_times, compiled_times)


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_all_benchmarks(size='medium', operations=None):
    """
    Run all benchmarks

    Args:
        size: 'small', 'medium', 'large', or 'extreme'
        operations: List of operation names, or None for all

    Returns:
        Dict of BenchmarkResult objects
    """
    device = get_device()
    config = BENCHMARK_CONFIGS[size]

    print("=" * 70)
    print(f"iVHL PERFORMANCE BENCHMARKS (Size: {size.upper()})")
    print("=" * 70)
    print()

    # Print device info
    stats = get_compile_stats()
    print("Device Configuration:")
    print("-" * 70)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Available benchmarks
    all_benchmarks = {
        'field': benchmark_field_superposition,
        'vortex': benchmark_vortex_field,
        'mera': benchmark_mera_contraction,
        'condensate': benchmark_condensate_potential,
        'gp': benchmark_gp_evolution
    }

    # Filter operations if specified
    if operations:
        benchmarks = {k: v for k, v in all_benchmarks.items() if k in operations}
    else:
        benchmarks = all_benchmarks

    # Run benchmarks
    results = {}
    print("Running Benchmarks:")
    print("-" * 70)

    for name, benchmark_fn in benchmarks.items():
        try:
            result = benchmark_fn(config, device)
            results[name] = result
            print(f"  [OK] {result.name}: {result.speedup:.2f}x speedup")
        except Exception as e:
            print(f"  [FAIL] {name} failed: {e}")

    print()

    # Summary
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    print()

    for result in results.values():
        print(result)
        print()

    # Overall speedup
    if results:
        avg_speedup = sum(r.speedup for r in results.values()) / len(results)
        print("=" * 70)
        print(f"AVERAGE SPEEDUP: {avg_speedup:.2f}x")
        print("=" * 70)
        print()

    return results


# ============================================================================
# Export Results
# ============================================================================

def export_results(results: Dict[str, BenchmarkResult], filename: str):
    """Export benchmark results to JSON"""
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(get_device()),
        'compile_mode': get_compile_mode(),
        'results': {name: result.to_dict() for name, result in results.items()}
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results exported to {filename}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iVHL Performance Benchmarks')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'extreme'],
                       default='medium', help='Problem size')
    parser.add_argument('--operation', choices=['field', 'vortex', 'mera', 'condensate', 'gp', 'all'],
                       default='all', help='Specific operation to benchmark')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    parser.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune', 'off'],
                       help='Override compile mode')

    args = parser.parse_args()

    # Set compile mode if specified
    if args.compile_mode:
        if args.compile_mode == 'off':
            set_compile_mode(None)
        else:
            set_compile_mode(args.compile_mode)

    # Determine operations
    operations = None if args.operation == 'all' else [args.operation]

    # Run benchmarks
    results = run_all_benchmarks(size=args.size, operations=operations)

    # Export if requested
    if args.export:
        export_results(results, args.export)
