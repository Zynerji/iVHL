"""
Multi-Scale Framework Tests
============================

Basic unit tests for all framework components.
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivhl.multiscale import (
    BoundaryResonanceSimulator, BoundaryConfig,
    GFTFieldEvolver, GFTConfig,
    MERABulkReconstructor, MERAConfig,
    PerturbationEngine, PerturbationConfig,
    RLDiscoveryAgent, RLDiscoveryConfig,
    MultiScaleUpscaler, UpscalingConfig,
    SimulationAnalyzer, AnalysisConfig
)


class TestBoundaryResonance:
    def test_initialization(self):
        config = BoundaryConfig(num_nodes=126, grid_resolution=16, timesteps=10)
        simulator = BoundaryResonanceSimulator(config)
        assert simulator.source_positions.shape == (126, 3)
    
    def test_field_computation(self):
        config = BoundaryConfig(num_nodes=10, grid_resolution=8, timesteps=5)
        simulator = BoundaryResonanceSimulator(config)
        field = simulator.compute_field(t=0.0)
        assert field.shape == (8, 8)


class TestGFTEvolution:
    def test_initialization(self):
        config = GFTConfig(grid_size=8, num_colors=2, timesteps=10)
        evolver = GFTFieldEvolver(config)
        assert evolver.field.shape == (2, 8, 8, 8)
    
    def test_evolution_step(self):
        config = GFTConfig(grid_size=8, num_colors=2, timesteps=5)
        evolver = GFTFieldEvolver(config)
        initial_field = evolver.field.clone()
        evolver.step()
        # Field should change after step
        assert not torch.allclose(evolver.field, initial_field)


class TestMERAReconstruction:
    def test_initialization(self):
        config = MERAConfig(depth=3, bond_dimension=4)
        reconstructor = MERABulkReconstructor(config)
        assert len(reconstructor.tensors) == 2 * 3  # U and W for each layer
    
    def test_reconstruction(self):
        config = MERAConfig(depth=3, bond_dimension=4)
        reconstructor = MERABulkReconstructor(config)
        boundary_data = np.random.randn(16, 16)
        results = reconstructor.run_reconstruction(boundary_data)
        assert 'bulk_representation' in results


class TestPerturbationEngine:
    def test_waveform_generation(self):
        config = PerturbationConfig(waveform_type="inspiral", duration=1.0)
        engine = PerturbationEngine(config)
        assert len(engine.waveform) > 0
    
    def test_perturbation_application(self):
        config = PerturbationConfig(amplitude=0.1)
        engine = PerturbationEngine(config)
        lattice = torch.randn(10, 3)
        perturbed = engine.apply_perturbation(lattice, time_idx=0)
        assert perturbed.shape == lattice.shape


class TestRLDiscovery:
    def test_initialization(self):
        config = RLDiscoveryConfig(state_dim=5, action_dim=3)
        agent = RLDiscoveryAgent(config)
        assert agent.actor is not None
        assert agent.critic is not None
    
    def test_action_selection(self):
        config = RLDiscoveryConfig(state_dim=5, action_dim=3)
        agent = RLDiscoveryAgent(config)
        state = np.random.randn(5)
        action = agent.select_action(state, explore=False)
        assert action.shape == (3,)


class TestMultiScaleUpscaler:
    def test_initialization(self):
        config = UpscalingConfig(scales=[32, 16, 8])
        upscaler = MultiScaleUpscaler(config)
        assert len(upscaler.projection_matrices) == 2
    
    def test_projection(self):
        config = UpscalingConfig(scales=[32, 16, 8])
        upscaler = MultiScaleUpscaler(config)
        boundary = np.random.randn(10, 32)
        results = upscaler.run_multiscale_projection(boundary)
        assert 32 in results['projections']
        assert 16 in results['projections']


class TestSimulationAnalyzer:
    def test_initialization(self):
        config = AnalysisConfig(output_dir="test_results/")
        analyzer = SimulationAnalyzer(config)
        assert analyzer.config.output_dir == "test_results/"
    
    def test_boundary_analysis(self):
        config = AnalysisConfig()
        analyzer = SimulationAnalyzer(config)
        
        results = {
            'field_evolution': np.random.randn(10, 8, 8),
            'times': np.arange(10) * 0.01
        }
        
        analysis = analyzer.analyze_boundary_results(results)
        assert 'mean_amplitude' in analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
