"""Tests for hierarchical information dynamics"""
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivhl.hierarchical import TensorHierarchy, HierarchyConfig

def test_hierarchy_initialization():
    config = HierarchyConfig(num_layers=3, base_dimension=8, bond_dimension=4)
    hierarchy = TensorHierarchy(config)
    assert len(hierarchy.layers) == 3

def test_compression():
    config = HierarchyConfig(num_layers=3, base_dimension=8, bond_dimension=4)
    hierarchy = TensorHierarchy(config)
    metrics = hierarchy.compress_layer(0, method="svd")
    assert 'information_loss' in metrics
    assert metrics['information_loss'] >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
