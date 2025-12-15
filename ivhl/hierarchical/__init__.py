"""
Hierarchical Information Dynamics
==================================

SCIENTIFIC DISCLAIMER:
This module explores computational patterns in hierarchical tensor networks
with information compression. It does NOT claim to model physical systems,
explain dark matter/energy, or validate multiverse theories.

This is MATHEMATICAL/COMPUTATIONAL exploration only.

Framework for studying:
- Multi-scale tensor network compression
- Information flow in hierarchical systems
- Entropy distribution under pruning strategies
- Emergent correlation structures

All results should be interpreted as patterns in abstract mathematical
systems, not as predictions about physical reality.
"""

__version__ = "1.0.0"
__author__ = "iVHL Development Team"

from .tensor_hierarchy import TensorHierarchy, HierarchyConfig
from .compression_engine import CompressionEngine, CompressionConfig
from .entropy_analyzer import EntropyAnalyzer, EntropyConfig
from .correlation_tracker import CorrelationTracker, CorrelationConfig

__all__ = [
    "TensorHierarchy",
    "HierarchyConfig",
    "CompressionEngine",
    "CompressionConfig",
    "EntropyAnalyzer",
    "EntropyConfig",
    "CorrelationTracker",
    "CorrelationConfig",
]
