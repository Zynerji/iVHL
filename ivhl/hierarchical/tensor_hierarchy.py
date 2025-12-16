"""
Tensor Hierarchy
================

Multi-layer tensor network for hierarchical information dynamics.

DISCLAIMER: Mathematical model for computational exploration only.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time


@dataclass
class HierarchyConfig:
    """Configuration for tensor hierarchy."""

    # Network structure
    num_layers: int = 5
    base_dimension: int = 64  # Dimension of bottom layer
    bond_dimension: int = 16   # Internal bond dimension
    compression_ratio: float = 0.5  # How much each layer compresses

    # Computation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Memory management (reserves for LLM + rendering)
    llm_reserved_gb: float = 6.0
    rendering_reserved_gb: float = 10.0

    def estimate_memory_usage(self) -> float:
        """Estimate VRAM usage in GB."""
        bytes_per_element = 4 if self.dtype == torch.float32 else 8

        total_bytes = 0
        current_dim = self.base_dimension

        for layer in range(self.num_layers):
            # Tensor at this layer: (bond_dim, current_dim, current_dim)
            layer_bytes = (self.bond_dimension * current_dim * current_dim *
                          bytes_per_element)
            total_bytes += layer_bytes

            # Next layer is compressed
            current_dim = int(current_dim * self.compression_ratio)

        return total_bytes / (1024**3)  # Convert to GB


class TensorHierarchy:
    """
    Hierarchical tensor network for information dynamics.

    Structure:
        Layer 0 (Base):    [64 x 64] boundary data
        Layer 1:           [32 x 32] compressed
        Layer 2:           [16 x 16] compressed
        Layer 3:           [8 x 8]   compressed
        Layer 4 (Bulk):    [4 x 4]   compressed

    Each compression step eliminates information while preserving
    correlations through tensor contractions.
    """

    def __init__(self, config: HierarchyConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Check GPU memory availability
        if self.device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            needed = (config.estimate_memory_usage() +
                     config.llm_reserved_gb +
                     config.rendering_reserved_gb)

            if needed > total_vram:
                print(f"⚠️ Warning: Need {needed:.1f}GB but only {total_vram:.1f}GB available")
                print(f"   Reducing bond dimension for safety...")
                self.config.bond_dimension = max(4, self.config.bond_dimension // 2)

        # Initialize layers
        self.layers = self._initialize_layers()
        self.layer_entropies = []
        self.layer_correlations = []

        print(f"TensorHierarchy initialized:")
        print(f"  - Layers: {config.num_layers}")
        print(f"  - Base dimension: {config.base_dimension}")
        print(f"  - Bond dimension: {config.bond_dimension}")
        print(f"  - Est. VRAM: {config.estimate_memory_usage():.2f} GB")
        print(f"  - Device: {self.device}")

    def _initialize_layers(self) -> List[torch.Tensor]:
        """Initialize tensor layers with random data."""
        layers = []
        current_dim = self.config.base_dimension

        for layer_idx in range(self.config.num_layers):
            # Create tensor: (bond_dim, spatial_dim, spatial_dim)
            tensor_shape = (
                self.config.bond_dimension,
                current_dim,
                current_dim
            )

            # Initialize with normalized random values
            tensor = torch.randn(
                *tensor_shape,
                device=self.device,
                dtype=self.config.dtype
            )
            tensor = tensor / torch.norm(tensor)

            layers.append(tensor)

            # Next layer dimension
            current_dim = int(current_dim * self.config.compression_ratio)

            print(f"    Layer {layer_idx}: shape {tensor_shape}")

        return layers

    def get_layer_info(self, layer_idx: int) -> Dict:
        """Get information about a specific layer."""
        if layer_idx >= len(self.layers):
            return {}

        tensor = self.layers[layer_idx]

        return {
            'index': layer_idx,
            'shape': tuple(tensor.shape),
            'norm': torch.norm(tensor).item(),
            'mean': torch.mean(tensor).item(),
            'std': torch.std(tensor).item(),
            'min': torch.min(tensor).item(),
            'max': torch.max(tensor).item(),
        }

    def get_all_layer_info(self) -> List[Dict]:
        """Get information about all layers."""
        return [self.get_layer_info(i) for i in range(len(self.layers))]

    def compress_layer(self, layer_idx: int, method: str = "svd") -> Dict:
        """
        Compress a layer by contracting tensors.

        Args:
            layer_idx: Layer to compress
            method: Compression method ("svd", "random", "learned")

        Returns:
            Dictionary with compression metrics
        """
        if layer_idx >= len(self.layers) - 1:
            return {"error": "Cannot compress top layer"}

        start_time = time.time()

        current_layer = self.layers[layer_idx]
        next_layer = self.layers[layer_idx + 1]

        # Original entropy
        original_entropy = self._compute_entropy(current_layer)

        # Perform compression
        if method == "svd":
            compressed = self._compress_svd(current_layer, next_layer.shape)
        elif method == "random":
            compressed = self._compress_random(current_layer, next_layer.shape)
        else:
            compressed = self._compress_learned(current_layer, next_layer.shape)

        # Update layer
        self.layers[layer_idx + 1] = compressed

        # Compressed entropy
        compressed_entropy = self._compute_entropy(compressed)

        # Information loss
        info_loss = original_entropy - compressed_entropy

        metrics = {
            'layer': layer_idx,
            'method': method,
            'original_entropy': original_entropy,
            'compressed_entropy': compressed_entropy,
            'information_loss': info_loss,
            'compression_time': time.time() - start_time,
        }

        self.layer_entropies.append(metrics)

        return metrics

    def _compress_svd(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Compress using SVD (preserves most information). FIXED version."""
        bond_dim, h, w = tensor.shape
        target_bond, target_h, target_w = target_shape

        # First, spatially downsample using interpolation
        downsampled = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),  # Add batch dim
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # Remove batch dim: [bond_dim, target_h, target_w]

        # Now compress bond dimension if needed
        if bond_dim > target_bond:
            # Reshape for SVD
            reshaped = downsampled.reshape(bond_dim, target_h * target_w)

            # SVD
            U, S, Vt = torch.linalg.svd(reshaped, full_matrices=False)

            # Reconstruct with truncated bonds
            k = min(target_bond, U.shape[1])
            compressed_flat = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

            # Take only target_bond rows and reshape
            compressed = compressed_flat[:target_bond, :].reshape(target_bond, target_h, target_w)
        else:
            # Bond dim already small enough, just pad if needed
            if bond_dim < target_bond:
                padding = torch.zeros(
                    target_bond - bond_dim, target_h, target_w,
                    device=downsampled.device,
                    dtype=downsampled.dtype
                )
                compressed = torch.cat([downsampled, padding], dim=0)
            else:
                compressed = downsampled

        return compressed

    def _compress_random(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Compress using random projection."""
        bond_dim, h, w = tensor.shape
        target_bond, target_h, target_w = target_shape

        # Random projection matrix
        proj = torch.randn(
            bond_dim, target_bond,
            device=self.device,
            dtype=self.config.dtype
        )
        proj = proj / torch.norm(proj, dim=0, keepdim=True)

        # Project
        compressed = torch.einsum('bij,bk->kij', tensor, proj)

        # Downsample spatially
        compressed = torch.nn.functional.interpolate(
            compressed.unsqueeze(0),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return compressed

    def _compress_learned(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Placeholder for learned compression (could use autoencoder)."""
        # For now, use SVD as fallback
        return self._compress_svd(tensor, target_shape)

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """
        Compute von Neumann entropy of tensor.

        S = -Tr(ρ log ρ), where ρ is density matrix
        """
        # Flatten to matrix
        flat = tensor.reshape(tensor.shape[0], -1)

        # Density matrix
        rho = flat @ flat.T
        rho = rho / torch.trace(rho)

        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Avoid log(0)

        # Entropy
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))

        return entropy.item()

    def compute_layer_correlation(self, layer1_idx: int, layer2_idx: int) -> float:
        """
        Compute correlation between two layers.

        Uses normalized inner product.
        """
        if layer1_idx >= len(self.layers) or layer2_idx >= len(self.layers):
            return 0.0

        t1 = self.layers[layer1_idx].flatten()
        t2 = self.layers[layer2_idx].flatten()

        # Pad shorter tensor
        if t1.numel() < t2.numel():
            t1 = torch.nn.functional.pad(t1, (0, t2.numel() - t1.numel()))
        elif t2.numel() < t1.numel():
            t2 = torch.nn.functional.pad(t2, (0, t1.numel() - t2.numel()))

        # Correlation
        correlation = torch.dot(t1 / torch.norm(t1), t2 / torch.norm(t2))

        return correlation.item()

    def to_numpy(self, layer_idx: int) -> np.ndarray:
        """Convert layer to numpy for visualization."""
        return self.layers[layer_idx].cpu().numpy()


if __name__ == "__main__":
    # Test
    config = HierarchyConfig(
        num_layers=5,
        base_dimension=32,
        bond_dimension=8
    )

    hierarchy = TensorHierarchy(config)

    print("\nLayer info:")
    for info in hierarchy.get_all_layer_info():
        print(f"  Layer {info['index']}: {info['shape']}, norm={info['norm']:.4f}")

    print("\nCompressing layer 0:")
    metrics = hierarchy.compress_layer(0, method="svd")
    print(f"  Info loss: {metrics['information_loss']:.4f}")
    print(f"  Time: {metrics['compression_time']:.4f}s")
