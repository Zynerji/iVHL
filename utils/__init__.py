"""
iVHL Utilities Module

Provides device management, torch.compile() helpers, and performance utilities
for GPU-accelerated quantum gravity simulations.
"""

from .device import (
    get_device,
    get_compile_mode,
    get_compiled,
    set_compile_mode,
    CompileMode
)

__all__ = [
    'get_device',
    'get_compile_mode',
    'get_compiled',
    'set_compile_mode',
    'CompileMode'
]
