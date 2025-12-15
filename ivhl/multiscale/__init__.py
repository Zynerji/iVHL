"""
iVHL Multi-Scale Simulation Framework
======================================

IMPORTANT DISCLAIMER:
This is a speculative computational framework for mathematical exploration.
It does NOT claim to explain real physical phenomena, discover new laws,
or predict dark matter/dark energy.

This framework explores:
- Holographic resonance patterns on spherical boundaries
- Group Field Theory (GFT) dynamics with colored tensor structure
- MERA (Multiscale Entanglement Renormalization Ansatz) tensor networks
- Emergent phenomena from geometric information compression
- Reinforcement learning for configuration discovery

All results should be interpreted as mathematical patterns in computational
models, not as predictions about physical reality.
"""

__version__ = "1.0.0"
__author__ = "iVHL Development Team"

from .boundary_resonance import BoundaryResonanceSimulator, BoundaryConfig
from .gft_field import GFTFieldEvolver, GFTConfig
from .mera_bulk import MERABulkReconstructor, MERAConfig
from .perturbation_engine import PerturbationEngine, PerturbationConfig
from .rl_discovery import RLDiscoveryAgent, RLDiscoveryConfig
from .upscaling import MultiScaleUpscaler, UpscalingConfig
from .analysis import SimulationAnalyzer, AnalysisConfig

__all__ = [
    "BoundaryResonanceSimulator",
    "BoundaryConfig",
    "GFTFieldEvolver",
    "GFTConfig",
    "MERABulkReconstructor",
    "MERAConfig",
    "PerturbationEngine",
    "PerturbationConfig",
    "RLDiscoveryAgent",
    "RLDiscoveryConfig",
    "MultiScaleUpscaler",
    "UpscalingConfig",
    "SimulationAnalyzer",
    "AnalysisConfig",
]
