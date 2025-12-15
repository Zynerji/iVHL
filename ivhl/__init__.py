"""
iVHL - Vibrational Helical Lattice Framework
=============================================

A computational research platform for quantum gravity phenomenology exploring:
- Holographic resonance on spherical boundaries
- Group Field Theory (GFT) condensate dynamics
- Tensor network holography (MERA)
- LIGO-inspired gravitational wave analysis
- Reinforcement learning discovery

This is NOT a theory of everything - it is a research tool for computational experiments.

Submodules:
-----------
- resonance: Holographic boundary dynamics and vortex control
- gft: Group Field Theory and colored tensor models
- tensor_networks: MERA, RT formula, AdS/CFT
- gw: Gravitational wave lattice analysis and fractal harmonics
- rl: Reinforcement learning (SAC, TD3-SAC hybrid)
- integration: API, integration utilities, report generation
- legacy: Deprecated modules (CCSD, orbital theory, predictions)

Author: iVHL Framework
Repository: https://github.com/Zynerji/iVHL
"""

__version__ = "1.0.0"
__author__ = "iVHL Framework"

# Package-level imports
from .resonance import holographic_resonance
from .gft import condensate_dynamics, tensor_models
from .tensor_networks import holography
from .gw import lattice_mode
from .rl import sac_core, td3_sac_core
from .integration import api, report_generator

__all__ = [
    'resonance',
    'gft',
    'tensor_networks',
    'gw',
    'rl',
    'integration',
    'legacy'
]
