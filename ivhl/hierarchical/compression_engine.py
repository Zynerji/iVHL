"""
Compression Engine
==================

Manages compression strategies for hierarchical tensor networks.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CompressionConfig:
    strategy: str = "svd"  # "svd", "random", "adaptive"
    target_info_retention: float = 0.9  # Retain 90% of information
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CompressionEngine:
    """Orchestrates multi-layer compression with different strategies."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.compression_history = []

    def compress_hierarchy(self, hierarchy, timesteps: int = 100) -> Dict:
        """Run full compression sequence."""
        print(f"Running {timesteps}-step compression...")

        metrics_history = []

        for step in range(timesteps):
            # Compress each layer sequentially
            for layer_idx in range(hierarchy.config.num_layers - 1):
                metrics = hierarchy.compress_layer(layer_idx, self.config.strategy)
                metrics['step'] = step
                metrics_history.append(metrics)

            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{timesteps}")

        return {
            'metrics_history': metrics_history,
            'final_entropies': [h.get_layer_info(i)
                               for i in range(hierarchy.config.num_layers)],
        }
