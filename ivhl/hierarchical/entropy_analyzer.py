"""Entropy Analyzer - Tracks entropy flow through hierarchy"""
import torch
from dataclasses import dataclass

@dataclass
class EntropyConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class EntropyAnalyzer:
    def __init__(self, config: EntropyConfig):
        self.config = config
    
    def analyze_flow(self, hierarchy):
        """Analyze entropy distribution across layers."""
        entropies = []
        for i in range(len(hierarchy.layers)):
            ent = hierarchy._compute_entropy(hierarchy.layers[i])
            entropies.append(ent)
        return entropies
