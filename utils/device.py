"""
Device and Compilation Management for iVHL

Provides unified device detection (CUDA/CPU), torch.compile() mode selection,
and helper functions for maximum performance on H100+ hardware while maintaining
CPU fallback compatibility.

Key Features:
- Automatic CUDA detection with fallback to CPU
- Compile mode selection (default, reduce-overhead, max-autotune)
- Conditional compilation (CUDA only, eager on CPU for stability)
- AMP (Automatic Mixed Precision) context management
- Performance profiling integration

Usage:
    from utils.device import get_device, get_compiled

    device = get_device()

    @get_compiled
    def my_hot_function(x):
        return x @ x.T
"""

import torch
import os
from enum import Enum
from typing import Callable, Optional, Any
from functools import wraps
import warnings


# ============================================================================
# Compile Modes
# ============================================================================

class CompileMode(Enum):
    """torch.compile() optimization modes"""
    OFF = None
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"


# ============================================================================
# Global State
# ============================================================================

_device = None
_compile_mode = None
_use_amp = None


# ============================================================================
# Device Detection
# ============================================================================

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the optimal compute device (CUDA if available, else CPU)

    Args:
        force_cpu: Force CPU usage even if CUDA is available

    Returns:
        torch.device: Detected device
    """
    global _device

    if _device is None or force_cpu:
        if force_cpu:
            _device = torch.device('cpu')
            print("[Device] Forced CPU mode")
        elif torch.cuda.is_available():
            _device = torch.device('cuda')
            props = torch.cuda.get_device_properties(0)
            print(f"[Device] CUDA detected: {props.name}")
            print(f"         Compute capability: {props.major}.{props.minor}")
            print(f"         Memory: {props.total_memory / 1e9:.2f} GB")
        else:
            _device = torch.device('cpu')
            print("[Device] No CUDA available, using CPU")

    return _device


def get_device_name() -> str:
    """Get human-readable device name"""
    device = get_device()
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(0).name
    else:
        return "CPU"


# ============================================================================
# Compile Mode Management
# ============================================================================

def get_compile_mode(auto_detect: bool = True) -> Optional[str]:
    """
    Get the current torch.compile() mode

    Args:
        auto_detect: Automatically detect optimal mode based on device

    Returns:
        Compile mode string or None (eager mode)
    """
    global _compile_mode

    if _compile_mode is None and auto_detect:
        device = get_device()

        if device.type == 'cuda':
            # Check compute capability for optimal mode selection
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(0)
                compute_cap = props.major * 10 + props.minor

                # H100 is compute capability 9.0, A100 is 8.0
                if compute_cap >= 90:  # H100+
                    _compile_mode = CompileMode.MAX_AUTOTUNE.value
                    print("[Compile] H100+ detected: Using 'max-autotune' mode")
                elif compute_cap >= 80:  # A100+
                    _compile_mode = CompileMode.REDUCE_OVERHEAD.value
                    print("[Compile] A100+ detected: Using 'reduce-overhead' mode")
                else:  # Older GPUs
                    _compile_mode = CompileMode.DEFAULT.value
                    print("[Compile] Using 'default' compile mode")
            else:
                _compile_mode = CompileMode.REDUCE_OVERHEAD.value
                print("[Compile] CUDA available: Using 'reduce-overhead' mode")
        else:
            # CPU: Don't compile (eager mode for better stability)
            _compile_mode = None
            print("[Compile] CPU mode: Compilation disabled (eager execution)")

    return _compile_mode


def set_compile_mode(mode: CompileMode | str | None):
    """
    Manually set the torch.compile() mode

    Args:
        mode: CompileMode enum, string, or None (eager)
    """
    global _compile_mode

    if isinstance(mode, CompileMode):
        _compile_mode = mode.value
    elif isinstance(mode, str):
        # Validate mode string
        valid_modes = [m.value for m in CompileMode if m.value is not None]
        if mode in valid_modes:
            _compile_mode = mode
        else:
            raise ValueError(f"Invalid compile mode: {mode}. Valid: {valid_modes}")
    elif mode is None:
        _compile_mode = None
    else:
        raise TypeError(f"mode must be CompileMode, str, or None, got {type(mode)}")

    print(f"[Compile] Mode set to: {_compile_mode or 'OFF (eager)'}")


# ============================================================================
# Compilation Helper
# ============================================================================

def get_compiled(
    fn: Optional[Callable] = None,
    *,
    mode: Optional[str] = None,
    fullgraph: bool = True,
    dynamic: bool = True,
    backend: str = "inductor",
    disable_on_cpu: bool = True,
    fallback_on_error: bool = True
) -> Callable:
    """
    Decorator/wrapper to compile functions with torch.compile()

    Automatically handles:
    - Mode selection (uses global mode if not specified)
    - Device-aware compilation (CUDA only by default)
    - Fallback to eager mode on errors
    - Dynamic shape support

    Usage as decorator:
        @get_compiled
        def my_function(x):
            return x @ x.T

    Usage with custom settings:
        @get_compiled(mode='max-autotune', fullgraph=False)
        def my_function(x):
            return x @ x.T

    Usage as wrapper:
        compiled_fn = get_compiled(my_function)

    Args:
        fn: Function to compile
        mode: Compile mode override (None uses global)
        fullgraph: Whether to compile entire graph (True = better perf, False = more stable)
        dynamic: Support dynamic input shapes
        backend: Compilation backend ('inductor' default)
        disable_on_cpu: Don't compile on CPU (better stability)
        fallback_on_error: Fall back to eager on compilation errors

    Returns:
        Compiled (or original) function
    """
    def decorator(func: Callable) -> Callable:
        # Determine if we should compile
        device = get_device()
        compile_mode_to_use = mode if mode is not None else get_compile_mode()

        # Don't compile on CPU if disabled
        if disable_on_cpu and device.type == 'cpu':
            return func

        # Don't compile if mode is None (eager)
        if compile_mode_to_use is None:
            return func

        # Try to compile
        try:
            compiled_func = torch.compile(
                func,
                mode=compile_mode_to_use,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend
            )

            # Add metadata
            compiled_func._is_compiled = True
            compiled_func._compile_mode = compile_mode_to_use
            compiled_func._original_func = func

            return compiled_func

        except Exception as e:
            if fallback_on_error:
                warnings.warn(
                    f"torch.compile() failed for {func.__name__}: {e}. "
                    f"Falling back to eager mode.",
                    RuntimeWarning
                )
                return func
            else:
                raise

    # Handle both decorator and wrapper usage
    if fn is None:
        # Called as @get_compiled(args...)
        return decorator
    else:
        # Called as @get_compiled or get_compiled(fn)
        return decorator(fn)


def is_compiled(fn: Callable) -> bool:
    """Check if a function has been compiled"""
    return hasattr(fn, '_is_compiled') and fn._is_compiled


def get_original(fn: Callable) -> Callable:
    """Get original (uncompiled) function if available"""
    if hasattr(fn, '_original_func'):
        return fn._original_func
    return fn


# ============================================================================
# AMP (Automatic Mixed Precision) Management
# ============================================================================

def use_amp(force: Optional[bool] = None) -> bool:
    """
    Check if AMP (Automatic Mixed Precision) should be used

    Args:
        force: Force enable/disable AMP (None = auto-detect)

    Returns:
        Whether to use AMP
    """
    global _use_amp

    if force is not None:
        _use_amp = force
        return _use_amp

    if _use_amp is None:
        device = get_device()
        # Use AMP on CUDA devices with tensor cores (compute capability >= 7.0)
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(0)
            compute_cap = props.major * 10 + props.minor
            _use_amp = compute_cap >= 70  # Volta (V100) and newer
            if _use_amp:
                print(f"[AMP] Enabled (compute capability {props.major}.{props.minor})")
            else:
                print(f"[AMP] Disabled (compute capability {props.major}.{props.minor} < 7.0)")
        else:
            _use_amp = False
            print("[AMP] Disabled (CPU mode)")

    return _use_amp


def autocast_context(enabled: Optional[bool] = None, dtype=torch.float16):
    """
    Get autocast context manager for AMP

    Args:
        enabled: Override auto-detection
        dtype: AMP dtype (float16 or bfloat16)

    Returns:
        torch.amp.autocast context manager
    """
    device = get_device()
    use_autocast = enabled if enabled is not None else use_amp()

    if device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', enabled=use_autocast, dtype=dtype)
    else:
        # CPU doesn't support autocast in older PyTorch, use dummy context
        from contextlib import nullcontext
        return nullcontext()


# ============================================================================
# Performance Profiling
# ============================================================================

class SimpleProfiler:
    """Simple performance profiler for benchmarking"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records = []

    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            import time
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled and torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
            self.records.append(elapsed_ms)
        else:
            import time
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.records.append(elapsed_ms)

    def get_average(self) -> float:
        """Get average time in milliseconds"""
        return sum(self.records) / len(self.records) if self.records else 0.0

    def get_last(self) -> float:
        """Get last recorded time in milliseconds"""
        return self.records[-1] if self.records else 0.0


# ============================================================================
# Utility Functions
# ============================================================================

def warm_up_compile(fn: Callable, *args, **kwargs):
    """
    Warm up a compiled function with dummy inputs

    torch.compile() compiles on first invocation, so this ensures
    the compilation overhead doesn't affect first real run.

    Args:
        fn: Compiled function
        *args: Sample arguments
        **kwargs: Sample keyword arguments
    """
    if is_compiled(fn):
        try:
            # Run once to trigger compilation
            _ = fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[Warmup] Compiled {fn._original_func.__name__}")
        except Exception as e:
            warnings.warn(f"Warmup failed for {fn.__name__}: {e}", RuntimeWarning)


def clear_compile_cache():
    """Clear torch.compile() cache to force recompilation"""
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
        print("[Compile] Cache cleared")


def get_compile_stats() -> dict:
    """Get compilation statistics"""
    stats = {
        'device': str(get_device()),
        'device_name': get_device_name(),
        'compile_mode': get_compile_mode(),
        'amp_enabled': use_amp(),
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        stats.update({
            'gpu_name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory_gb': props.total_memory / 1e9,
            'allocated_memory_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_memory_gb': torch.cuda.memory_reserved() / 1e9,
        })

    return stats


# ============================================================================
# Main - Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("iVHL DEVICE & COMPILE MANAGEMENT TEST")
    print("=" * 70)
    print()

    # Test device detection
    print("1. Device Detection:")
    print("-" * 70)
    device = get_device()
    print(f"  Device: {device}")
    print(f"  Device name: {get_device_name()}")
    print()

    # Test compile mode
    print("2. Compile Mode:")
    print("-" * 70)
    mode = get_compile_mode()
    print(f"  Mode: {mode or 'OFF (eager)'}")
    print()

    # Test AMP
    print("3. AMP Status:")
    print("-" * 70)
    amp_enabled = use_amp()
    print(f"  Enabled: {amp_enabled}")
    print()

    # Test compilation
    print("4. Compilation Test:")
    print("-" * 70)

    # Define test function
    @get_compiled
    def test_matmul(x: torch.Tensor) -> torch.Tensor:
        """Test function for compilation"""
        return x @ x.T

    # Create test data
    size = 1024
    x = torch.randn(size, size, device=device)

    print(f"  Function: test_matmul")
    print(f"  Input shape: {x.shape}")
    print(f"  Compiled: {is_compiled(test_matmul)}")
    print()

    # Warm up
    print("  Warming up...")
    with SimpleProfiler() as warmup_prof:
        y = test_matmul(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    print(f"  Warmup time: {warmup_prof.get_last():.2f} ms")
    print()

    # Benchmark
    print("  Benchmarking (10 runs)...")
    profiler = SimpleProfiler()
    for _ in range(10):
        with profiler:
            y = test_matmul(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    print(f"  Average time: {profiler.get_average():.2f} ms")
    print()

    # Stats
    print("5. Compile Statistics:")
    print("-" * 70)
    stats = get_compile_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("[OK] Device and compile management ready!")
