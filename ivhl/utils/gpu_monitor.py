"""
GPU Monitor and OOM Handler for SPOF #2
========================================

Monitors GPU memory usage and handles Out-of-Memory errors gracefully
with automatic parameter downscaling.
"""

import torch
import gc
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization_percent: float


class GPUMonitor:
    """
    Monitors GPU memory and handles OOM errors.

    Features:
    - Real-time memory tracking
    - OOM detection and recovery
    - Automatic parameter downscaling
    - CPU fallback when needed
    """

    def __init__(self, device: str = "cuda", safety_margin_gb: float = 2.0):
        self.device = torch.device(device)
        self.safety_margin_gb = safety_margin_gb
        self.oom_count = 0
        self.downscale_history = []

        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()

        if self.cuda_available:
            self.device_properties = torch.cuda.get_device_properties(0)
            print(f"🎮 GPU Detected: {self.device_properties.name}")
            print(f"   Total VRAM: {self.device_properties.total_memory / (1024**3):.1f} GB")
        else:
            print("⚠️  No GPU detected, using CPU")

    def get_memory_stats(self) -> Optional[GPUMemoryStats]:
        """Get current GPU memory statistics."""
        if not self.cuda_available:
            return None

        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = self.device_properties.total_memory / (1024**3)
        free = total - reserved

        stats = GPUMemoryStats(
            total_gb=total,
            allocated_gb=allocated,
            reserved_gb=reserved,
            free_gb=free,
            utilization_percent=(reserved / total) * 100
        )

        return stats

    def print_memory_stats(self, prefix: str = ""):
        """Print current memory usage."""
        stats = self.get_memory_stats()

        if stats:
            print(f"{prefix}GPU Memory:")
            print(f"  Allocated: {stats.allocated_gb:.2f} GB")
            print(f"  Reserved:  {stats.reserved_gb:.2f} GB")
            print(f"  Free:      {stats.free_gb:.2f} GB")
            print(f"  Usage:     {stats.utilization_percent:.1f}%")
        else:
            print(f"{prefix}CPU mode (no GPU stats)")

    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if self.cuda_available:
            torch.cuda.empty_cache()
        gc.collect()

    def check_available_memory(self, required_gb: float) -> bool:
        """
        Check if enough GPU memory is available.

        Args:
            required_gb: Memory required in GB

        Returns:
            True if sufficient memory available
        """
        stats = self.get_memory_stats()

        if not stats:
            return False  # No GPU

        available = stats.free_gb - self.safety_margin_gb
        return available >= required_gb

    def calculate_downscale_factor(self, current_dim: int, target_memory_gb: float) -> Tuple[int, float]:
        """
        Calculate new dimension to fit in target memory.

        Args:
            current_dim: Current tensor dimension
            target_memory_gb: Target memory budget

        Returns:
            (new_dimension, scale_factor)
        """
        stats = self.get_memory_stats()

        if not stats:
            # Fallback for CPU
            return max(current_dim // 4, 8), 0.25

        # Memory scales roughly with dim^3 for 3D tensors
        current_memory = stats.allocated_gb
        available_memory = stats.free_gb - self.safety_margin_gb

        if current_memory > target_memory_gb:
            # Need to reduce
            scale_factor = (target_memory_gb / current_memory) ** (1/3)
            new_dim = max(int(current_dim * scale_factor), 8)
        else:
            # Currently okay
            new_dim = current_dim
            scale_factor = 1.0

        return new_dim, scale_factor

    def handle_oom_error(self, current_config: Dict) -> Optional[Dict]:
        """
        Handle OOM error by downscaling parameters.

        Args:
            current_config: Current simulation configuration

        Returns:
            New downscaled config, or None if can't recover
        """
        self.oom_count += 1
        print(f"\n⚠️  OOM Error #{self.oom_count} detected!")

        # Clear cache first
        self.clear_cache()
        self.print_memory_stats("  After cache clear: ")

        # Try to downscale
        if self.oom_count > 3:
            print("❌ Too many OOM errors, falling back to CPU")
            return {
                **current_config,
                'device': 'cpu',
                'base_dimension': max(current_config.get('base_dimension', 64) // 4, 8),
                'bond_dimension': max(current_config.get('bond_dimension', 16) // 2, 4),
            }

        # Aggressive downscale
        downscale_factor = 0.5 ** self.oom_count

        new_config = {
            **current_config,
            'base_dimension': max(int(current_config.get('base_dimension', 64) * downscale_factor), 8),
            'bond_dimension': max(int(current_config.get('bond_dimension', 16) * downscale_factor), 4),
            'batch_size': max(current_config.get('batch_size', 1) // 2, 1),
        }

        print(f"📉 Downscaling parameters by {downscale_factor:.1%}:")
        print(f"   base_dimension: {current_config.get('base_dimension')} → {new_config['base_dimension']}")
        print(f"   bond_dimension: {current_config.get('bond_dimension')} → {new_config['bond_dimension']}")

        self.downscale_history.append({
            'oom_count': self.oom_count,
            'old_config': current_config,
            'new_config': new_config
        })

        return new_config


def with_oom_handling(gpu_monitor: GPUMonitor):
    """
    Decorator to wrap functions with automatic OOM error handling.

    Usage:
        @with_oom_handling(gpu_monitor)
        def create_large_tensor(config):
            return torch.randn(config['size'], device=config['device'])
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            max_retries = 3
            attempt = 0

            while attempt < max_retries:
                try:
                    # Try to execute function
                    result = func(*args, **kwargs)
                    return result

                except torch.cuda.OutOfMemoryError as e:
                    attempt += 1
                    print(f"\n❌ CUDA OOM Error (attempt {attempt}/{max_retries})")

                    if attempt >= max_retries:
                        print("❌ Max retries exceeded, re-raising error")
                        raise

                    # Clear cache
                    gpu_monitor.clear_cache()

                    # Try to get config from kwargs
                    if 'config' in kwargs:
                        new_config = gpu_monitor.handle_oom_error(kwargs['config'])
                        if new_config:
                            kwargs['config'] = new_config
                            print(f"🔄 Retrying with downscaled config...")
                        else:
                            raise
                    else:
                        print("⚠️  No config found, cannot auto-recover")
                        raise

                except Exception as e:
                    # Non-OOM error, re-raise
                    raise

        return wrapper
    return decorator


def estimate_tensor_memory(shape: Tuple[int, ...], dtype=torch.float32) -> float:
    """
    Estimate memory required for a tensor in GB.

    Args:
        shape: Tensor shape
        dtype: Data type

    Returns:
        Memory in GB
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    # Bytes per element
    if dtype in [torch.float32, torch.int32]:
        bytes_per_element = 4
    elif dtype in [torch.float64, torch.int64]:
        bytes_per_element = 8
    elif dtype in [torch.float16, torch.bfloat16]:
        bytes_per_element = 2
    else:
        bytes_per_element = 4  # Default

    total_bytes = num_elements * bytes_per_element
    return total_bytes / (1024**3)


if __name__ == "__main__":
    # Test GPU monitor
    monitor = GPUMonitor()

    print("\n" + "="*60)
    print("GPU Monitor Test")
    print("="*60)

    monitor.print_memory_stats("\n")

    # Test memory estimation
    test_shape = (128, 128, 128)
    estimated = estimate_tensor_memory(test_shape)
    print(f"\nEstimated memory for {test_shape} float32 tensor: {estimated:.3f} GB")

    # Test downscaling
    test_config = {
        'base_dimension': 64,
        'bond_dimension': 16,
        'device': 'cuda',
        'batch_size': 4
    }

    print("\nSimulating OOM error...")
    new_config = monitor.handle_oom_error(test_config)
    print(f"New config: {new_config}")
