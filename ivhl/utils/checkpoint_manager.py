"""
Checkpoint Manager for SPOF #1
===============================

Provides checkpoint/resume functionality to prevent data loss on failures.
"""

import os
import json
import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """
    Manages simulation checkpoints for fault tolerance.

    Saves state after each major phase:
    - Initial hierarchy creation
    - Compression steps (every N timesteps)
    - Rendering completion
    - Pre-report generation
    """

    def __init__(self, checkpoint_dir: str = "/results/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track current checkpoint metadata
        self.metadata_path = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata if exists."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            'last_checkpoint': None,
            'completed_phases': [],
            'checkpoints': []
        }

    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(
        self,
        phase: str,
        state: Dict[str, Any],
        step: Optional[int] = None
    ):
        """
        Save checkpoint for a simulation phase.

        Args:
            phase: Phase name ("init", "compression_step_100", "rendering", etc.)
            state: Dictionary containing saveable state (tensors, metrics, config)
            step: Optional timestep number
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create checkpoint filename
        if step is not None:
            checkpoint_name = f"{phase}_step_{step}_{timestamp}"
        else:
            checkpoint_name = f"{phase}_{timestamp}"

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.ckpt"

        print(f"💾 Saving checkpoint: {checkpoint_name}")

        try:
            # Save checkpoint data
            checkpoint_data = {
                'phase': phase,
                'step': step,
                'timestamp': timestamp,
                'state': state
            }

            # Use torch.save for tensor compatibility
            torch.save(checkpoint_data, checkpoint_path)

            # Update metadata
            self.metadata['last_checkpoint'] = checkpoint_name
            self.metadata['checkpoints'].append({
                'name': checkpoint_name,
                'phase': phase,
                'step': step,
                'timestamp': timestamp,
                'path': str(checkpoint_path)
            })

            if phase not in self.metadata['completed_phases']:
                self.metadata['completed_phases'].append(phase)

            self._save_metadata()

            print(f"✅ Checkpoint saved: {checkpoint_path.name}")
            return True

        except Exception as e:
            print(f"❌ Checkpoint save failed: {e}")
            return False

    def load_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Dict]:
        """
        Load a checkpoint.

        Args:
            checkpoint_name: Specific checkpoint to load, or None for most recent

        Returns:
            Checkpoint data dict or None if not found
        """
        if checkpoint_name is None:
            # Load most recent
            checkpoint_name = self.metadata.get('last_checkpoint')

        if checkpoint_name is None:
            print("No checkpoints found")
            return None

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.ckpt"

        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None

        try:
            print(f"📂 Loading checkpoint: {checkpoint_name}")
            data = torch.load(checkpoint_path)
            print(f"✅ Checkpoint loaded: {data['phase']} at step {data.get('step', 'N/A')}")
            return data

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return self.metadata.get('checkpoints', [])

    def get_last_completed_phase(self) -> Optional[str]:
        """Get the last successfully completed phase."""
        phases = self.metadata.get('completed_phases', [])
        return phases[-1] if phases else None

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Remove old checkpoints to save disk space.

        Args:
            keep_last_n: Number of most recent checkpoints to keep
        """
        checkpoints = self.metadata.get('checkpoints', [])

        if len(checkpoints) <= keep_last_n:
            return

        # Sort by timestamp
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda x: x['timestamp'],
            reverse=True
        )

        # Remove old ones
        to_remove = checkpoints_sorted[keep_last_n:]

        for ckpt in to_remove:
            path = Path(ckpt['path'])
            if path.exists():
                path.unlink()
                print(f"🗑️  Removed old checkpoint: {ckpt['name']}")

        # Update metadata
        self.metadata['checkpoints'] = checkpoints_sorted[:keep_last_n]
        self._save_metadata()


def with_checkpoint(phase_name: str, checkpoint_manager: CheckpointManager):
    """
    Decorator to wrap functions with automatic checkpointing.

    Usage:
        @with_checkpoint("initialization", checkpoint_mgr)
        def initialize_simulation(config):
            # ... simulation code ...
            return hierarchy, metrics
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                print(f"\n{'='*60}")
                print(f"🚀 Starting phase: {phase_name}")
                print(f"{'='*60}")

                result = func(*args, **kwargs)

                # Save checkpoint after successful completion
                checkpoint_manager.save_checkpoint(
                    phase=phase_name,
                    state={'result': result, 'args': args, 'kwargs': kwargs}
                )

                print(f"✅ Phase completed: {phase_name}\n")
                return result

            except Exception as e:
                print(f"\n❌ Phase failed: {phase_name}")
                print(f"Error: {e}")

                # Try to save partial state
                try:
                    checkpoint_manager.save_checkpoint(
                        phase=f"{phase_name}_FAILED",
                        state={'error': str(e), 'args': args, 'kwargs': kwargs}
                    )
                except:
                    pass

                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Test checkpoint manager
    mgr = CheckpointManager("/tmp/test_checkpoints")

    # Test save
    test_state = {
        'tensor': torch.randn(10, 10),
        'metrics': {'entropy': 4.5, 'step': 100},
        'config': {'dim': 64}
    }

    mgr.save_checkpoint("test_phase", test_state, step=100)

    # Test load
    loaded = mgr.load_checkpoint()

    if loaded:
        print(f"\nLoaded phase: {loaded['phase']}")
        print(f"State keys: {loaded['state'].keys()}")

    # List checkpoints
    print(f"\nAll checkpoints: {mgr.list_checkpoints()}")
