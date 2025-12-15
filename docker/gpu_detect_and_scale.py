#!/usr/bin/env python3
"""
GPU Detection and Auto-Scaling
===============================

Detects available GPU, reserves memory for LLM + rendering,
and scales simulation parameters to fit remaining VRAM.
"""

import subprocess
import json
import sys


def detect_gpu():
    """Detect GPU and get specs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )

        line = result.stdout.strip()
        if not line:
            return None

        parts = line.split(',')
        gpu_name = parts[0].strip()
        memory_str = parts[1].strip().replace(' MiB', '')
        memory_gb = int(memory_str) / 1024

        return {
            'name': gpu_name,
            'memory_gb': memory_gb,
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def calculate_allocation(total_memory_gb):
    """
    Calculate resource allocation.

    Reserves:
    - 6GB for Qwen2.5-2B LLM
    - 10GB for PyVista GPU rendering
    - 4GB safety buffer
    - Remainder for simulation
    """
    LLM_RESERVED = 6.0
    RENDERING_RESERVED = 10.0
    SAFETY_BUFFER = 4.0

    available_for_sim = total_memory_gb - LLM_RESERVED - RENDERING_RESERVED - SAFETY_BUFFER

    if available_for_sim < 5:
        print(f"⚠️ Warning: Only {available_for_sim:.1f}GB for simulation!")
        print("   Consider using smaller parameters or disabling rendering.")

    return {
        'total_vram': total_memory_gb,
        'llm_reserved': LLM_RESERVED,
        'rendering_reserved': RENDERING_RESERVED,
        'safety_buffer': SAFETY_BUFFER,
        'simulation_available': max(available_for_sim, 2.0),
    }


def scale_simulation_parameters(available_gb):
    """
    Scale simulation parameters based on available VRAM.

    Returns recommended config for hierarchical dynamics.
    """
    if available_gb >= 60:  # H100/H200 high capacity
        return {
            'base_dimension': 128,
            'bond_dimension': 32,
            'num_layers': 7,
            'timesteps': 1000,
            'rendering_resolution': (1920, 1080),
            'quality': 'ultra',
        }
    elif available_gb >= 30:  # H100 medium capacity
        return {
            'base_dimension': 64,
            'bond_dimension': 16,
            'num_layers': 5,
            'timesteps': 500,
            'rendering_resolution': (1920, 1080),
            'quality': 'high',
        }
    elif available_gb >= 15:  # A100 or reduced H100
        return {
            'base_dimension': 32,
            'bond_dimension': 8,
            'num_layers': 5,
            'timesteps': 300,
            'rendering_resolution': (1280, 720),
            'quality': 'medium',
        }
    else:  # Low memory
        return {
            'base_dimension': 16,
            'bond_dimension': 4,
            'num_layers': 4,
            'timesteps': 100,
            'rendering_resolution': (1280, 720),
            'quality': 'low',
        }


def main():
    print("="*60)
    print("GPU Detection and Auto-Scaling")
    print("="*60)

    # Detect GPU
    gpu_info = detect_gpu()

    if gpu_info is None:
        print("\n❌ No NVIDIA GPU detected!")
        print("   Falling back to CPU mode (will be slow)...")

        config = {
            'device': 'cpu',
            'llm_enabled': False,
            'rendering_enabled': False,
            'simulation_params': {
                'base_dimension': 8,
                'bond_dimension': 2,
                'num_layers': 3,
                'timesteps': 50,
            }
        }
    else:
        print(f"\n✅ GPU Detected: {gpu_info['name']}")
        print(f"   Total VRAM: {gpu_info['memory_gb']:.1f} GB")

        # Calculate allocation
        allocation = calculate_allocation(gpu_info['memory_gb'])

        print("\nResource Allocation:")
        print(f"  - LLM (Qwen2.5-2B): {allocation['llm_reserved']} GB")
        print(f"  - Rendering: {allocation['rendering_reserved']} GB")
        print(f"  - Safety Buffer: {allocation['safety_buffer']} GB")
        print(f"  - Simulation: {allocation['simulation_available']:.1f} GB")

        # Scale parameters
        sim_params = scale_simulation_parameters(allocation['simulation_available'])

        print("\nRecommended Simulation Parameters:")
        print(f"  - Base dimension: {sim_params['base_dimension']}")
        print(f"  - Bond dimension: {sim_params['bond_dimension']}")
        print(f"  - Layers: {sim_params['num_layers']}")
        print(f"  - Timesteps: {sim_params['timesteps']}")
        print(f"  - Quality: {sim_params['quality']}")

        config = {
            'device': 'cuda',
            'gpu_name': gpu_info['name'],
            'total_vram': gpu_info['memory_gb'],
            'llm_enabled': True,
            'rendering_enabled': True,
            'allocation': allocation,
            'simulation_params': sim_params,
        }

    # Save config
    with open('/app/auto_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n✅ Configuration saved to /app/auto_config.json")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
