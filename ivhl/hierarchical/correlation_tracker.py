"""Correlation Tracker - Measures layer-to-layer correlations"""
import torch
from dataclasses import dataclass

@dataclass
class CorrelationConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CorrelationTracker:
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.history = []
    
    def track_all_correlations(self, hierarchy):
        """Compute all pairwise layer correlations."""
        correlations = {}
        for i in range(len(hierarchy.layers)):
            for j in range(i+1, len(hierarchy.layers)):
                corr = hierarchy.compute_layer_correlation(i, j)
                correlations[f"{i}-{j}"] = corr
        return correlations
