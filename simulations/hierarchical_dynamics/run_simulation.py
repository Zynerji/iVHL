"""
Hierarchical Information Dynamics - Main Runner
================================================

SCIENTIFIC DISCLAIMER:
This simulation explores computational patterns in hierarchical tensor
networks. Results represent mathematical structures in abstract systems,
NOT predictions about physical reality, dark matter, or cosmology.

Usage:
    python run_simulation.py --timesteps 500 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
from ivhl.hierarchical import (
    TensorHierarchy, HierarchyConfig,
    CompressionEngine, CompressionConfig,
    EntropyAnalyzer, EntropyConfig,
    CorrelationTracker, CorrelationConfig
)
from web_monitor.rendering.gpu_renderer import GPURenderer
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Run hierarchical information dynamics simulation')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of compression timesteps')
    parser.add_argument('--layers', type=int, default=5, help='Number of hierarchy layers')
    parser.add_argument('--base-dim', type=int, default=64, help='Base layer dimension')
    parser.add_argument('--bond-dim', type=int, default=16, help='Bond dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--output-dir', type=str, default='results/hierarchical', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Hierarchical Information Dynamics Simulation")
    print("="*60)
    print()
    print("DISCLAIMER: This is a computational exploration of")
    print("mathematical tensor networks. NOT a physics simulation.")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize hierarchy
    print("[1/5] Initializing tensor hierarchy...")
    hierarchy_config = HierarchyConfig(
        num_layers=args.layers,
        base_dimension=args.base_dim,
        bond_dimension=args.bond_dim,
        device=args.device
    )
    hierarchy = TensorHierarchy(hierarchy_config)

    # 2. Initialize compression engine
    print("\n[2/5] Initializing compression engine...")
    compression_config = CompressionConfig(
        strategy="svd",
        device=args.device
    )
    compressor = CompressionEngine(compression_config)

    # 3. Initialize analyzers
    print("\n[3/5] Initializing analyzers...")
    entropy_analyzer = EntropyAnalyzer(EntropyConfig(device=args.device))
    correlation_tracker = CorrelationTracker(CorrelationConfig(device=args.device))

    # 4. Initialize renderer (if viz enabled)
    renderer = None
    if not args.no_viz:
        print("\n[4/5] Initializing GPU renderer...")
        try:
            renderer = GPURenderer(resolution=(1920, 1080), device=args.device)
        except Exception as e:
            print(f"⚠️ Could not initialize renderer: {e}")
            print("   Continuing without visualization...")

    # 5. Run simulation
    print(f"\n[5/5] Running {args.timesteps}-step simulation...")
    print("-" * 60)

    start_time = time.time()

    results = {
        'timesteps': [],
        'entropies': [],
        'correlations': [],
        'layer_info': [],
    }

    for step in range(args.timesteps):
        # Compress each layer
        for layer_idx in range(args.layers - 1):
            metrics = hierarchy.compress_layer(layer_idx, method="svd")

            if layer_idx == 0:  # Only log first layer to avoid spam
                results['timesteps'].append(step)
                results['entropies'].append(metrics['information_loss'])

        # Analyze entropy flow
        entropies = entropy_analyzer.analyze_flow(hierarchy)

        # Track correlations
        correlations = correlation_tracker.track_all_correlations(hierarchy)
        results['correlations'].append(correlations)

        # Render frame (every 10 steps to save time)
        if renderer and step % 10 == 0:
            try:
                frame_bytes = renderer.create_animation_frame(hierarchy, step, args.timesteps)
                # Save frame
                frame_path = os.path.join(args.output_dir, f'frame_{step:04d}.jpg')
                with open(frame_path, 'wb') as f:
                    f.write(frame_bytes)
            except Exception as e:
                print(f"⚠️ Rendering error at step {step}: {e}")

        # Progress update
        if (step + 1) % 50 == 0:
            progress = (step + 1) / args.timesteps * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (args.timesteps - step - 1)
            print(f"  Step {step+1}/{args.timesteps} ({progress:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    # Final analysis
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)

    elapsed_total = time.time() - start_time
    print(f"\nTotal time: {elapsed_total:.2f} seconds")
    print(f"Steps/second: {args.timesteps / elapsed_total:.2f}")

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert tensors to lists for JSON
        json_results = {
            'config': {
                'timesteps': args.timesteps,
                'layers': args.layers,
                'base_dim': args.base_dim,
                'bond_dim': args.bond_dim,
            },
            'entropies': results['entropies'],
            'final_layer_info': hierarchy.get_all_layer_info(),
        }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Final layer info
    print("\nFinal layer states:")
    for info in hierarchy.get_all_layer_info():
        print(f"  Layer {info['index']}: norm={info['norm']:.4f}, "
              f"mean={info['mean']:.4f}, std={info['std']:.4f}")

    # Cleanup
    if renderer:
        renderer.cleanup()

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
