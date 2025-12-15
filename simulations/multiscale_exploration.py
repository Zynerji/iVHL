"""
Multi-Scale Exploration Simulation
===================================

Main simulation runner that orchestrates all framework components.

DISCLAIMER: This is a computational exploration framework, not a physics theory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from ivhl.multiscale import (
    BoundaryResonanceSimulator, BoundaryConfig,
    GFTFieldEvolver, GFTConfig,
    MERABulkReconstructor, MERAConfig,
    PerturbationEngine, PerturbationConfig,
    MultiScaleUpscaler, UpscalingConfig,
    SimulationAnalyzer, AnalysisConfig
)


class MultiScaleExploration:
    """
    DISCLAIMER: This simulation is a COMPUTATIONAL EXPLORATION of mathematical models.
    It does NOT claim to explain real physical phenomena.
    """
    
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("Multi-Scale Holographic Exploration Framework")
        print("="*60)
        print()
        print("DISCLAIMER: This is a mathematical/computational exploration.")
        print("Results should NOT be interpreted as physics predictions.")
        print()
    
    def run_layer1_boundary(self):
        """Layer 1: Boundary Resonance"""
        print("\n[LAYER 1: Boundary Resonance]")
        print("-" * 40)
        
        config = BoundaryConfig(
            num_nodes=126,
            grid_resolution=32,
            timesteps=50,
            device="cpu"
        )
        
        simulator = BoundaryResonanceSimulator(config)
        results = simulator.run_simulation()
        
        # Save
        np.savez_compressed(
            f"{self.output_dir}/boundary_results.npz",
            **results
        )
        
        return results
    
    def run_layer2_gft(self):
        """Layer 2: GFT Field Evolution"""
        print("\n[LAYER 2: GFT Field Evolution]")
        print("-" * 40)
        
        config = GFTConfig(
            grid_size=16,
            num_colors=4,
            timesteps=50,
            device="cpu"
        )
        
        evolver = GFTFieldEvolver(config)
        results = evolver.run_evolution()
        
        # Save
        np.savez_compressed(
            f"{self.output_dir}/gft_results.npz",
            **results
        )
        
        return results
    
    def run_layer3_mera(self, boundary_results):
        """Layer 3: MERA Bulk Reconstruction"""
        print("\n[LAYER 3: MERA Bulk Reconstruction]")
        print("-" * 40)
        
        config = MERAConfig(
            depth=5,
            bond_dimension=8,
            device="cpu"
        )
        
        reconstructor = MERABulkReconstructor(config)
        
        # Use final boundary field
        boundary_field = boundary_results['field_evolution'][-1]
        
        results = reconstructor.run_reconstruction(boundary_field)
        
        return results
    
    def run_layer4_perturbation(self, boundary_results):
        """Layer 4: Perturbation Analysis"""
        print("\n[LAYER 4: Perturbation Analysis]")
        print("-" * 40)
        
        config = PerturbationConfig(
            waveform_type="inspiral",
            amplitude=0.05,
            duration=5.0
        )
        
        engine = PerturbationEngine(config)
        
        # Use source positions as lattice
        lattice = boundary_results['source_positions']
        
        results = engine.run_perturbation_campaign(lattice)
        
        return results
    
    def run_full_pipeline(self):
        """Run complete multi-scale pipeline"""
        print("\nStarting full pipeline...")
        print()
        
        # Layer 1: Boundary
        boundary_results = self.run_layer1_boundary()
        
        # Layer 2: GFT
        gft_results = self.run_layer2_gft()
        
        # Layer 3: MERA
        mera_results = self.run_layer3_mera(boundary_results)
        
        # Layer 4: Perturbation
        perturbation_results = self.run_layer4_perturbation(boundary_results)
        
        # Analyze
        print("\n[ANALYSIS]")
        print("-" * 40)
        
        analyzer_config = AnalysisConfig(output_dir=self.output_dir)
        analyzer = SimulationAnalyzer(analyzer_config)
        
        all_results = {
            'boundary': boundary_results,
            'gft': gft_results,
            'mera': mera_results,
            'perturbation': perturbation_results,
        }
        
        analysis = analyzer.run_full_analysis(all_results, output_prefix="multiscale")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}")
        
        return all_results, analysis


if __name__ == "__main__":
    exploration = MultiScaleExploration(output_dir="results/multiscale_exploration/")
    all_results, analysis = exploration.run_full_pipeline()
    
    print("\nKey Metrics:")
    if 'gft' in analysis:
        print(f"  - Final GFT energy: {analysis['gft'].get('final_energy', 'N/A')}")
    if 'mera' in analysis:
        print(f"  - MERA compression: {analysis['mera'].get('compression_ratio', 'N/A')}x")
