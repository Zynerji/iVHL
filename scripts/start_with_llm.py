#!/usr/bin/env python3
"""
iVHL Startup Script - LLM First, Then Simulations
==================================================

CRITICAL: This script ensures the LLM loads FIRST and reserves GPU memory
BEFORE simulations start. This prevents the simulation from saturating the H100
and leaving no room for the LLM.

Startup Sequence:
-----------------
1. Load LLM and reserve GPU memory (15-20% of H100)
2. Start LLM chat server
3. Calculate remaining GPU memory
4. Start simulation with memory budget
5. Connect LLM to simulation for monitoring

Usage:
------
# Start everything (recommended)
python scripts/start_with_llm.py --full

# Just LLM server
python scripts/start_with_llm.py --llm-only

# Custom memory allocation
python scripts/start_with_llm.py --llm-memory 0.2 --sim-memory 0.75

Author: iVHL Framework
Date: 2025-12-15
"""

import sys
import os
import time
import subprocess
import argparse
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manages GPU memory allocation between LLM and simulations"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()

        if self.has_gpu:
            self.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.total_memory_gb = 0
            self.gpu_name = "CPU"

    def get_available_memory(self):
        """Get currently available GPU memory in GB"""
        if not self.has_gpu:
            return 0

        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        return self.total_memory_gb - reserved

    def print_status(self):
        """Print GPU memory status"""
        logger.info("=" * 80)
        logger.info("GPU MEMORY STATUS")
        logger.info("=" * 80)
        logger.info(f"Device: {self.gpu_name}")

        if self.has_gpu:
            logger.info(f"Total Memory: {self.total_memory_gb:.2f} GB")
            logger.info(f"Available Memory: {self.get_available_memory():.2f} GB")
            logger.info(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            logger.info(f"Reserved Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        else:
            logger.info("No GPU detected - using CPU")

        logger.info("=" * 80)


class StartupCoordinator:
    """Coordinates startup of LLM and simulations"""

    def __init__(self, args):
        self.args = args
        self.gpu_manager = GPUMemoryManager()
        self.llm_process = None
        self.sim_process = None

    def start(self):
        """Main startup sequence"""
        logger.info("=" * 80)
        logger.info("iVHL STARTUP - LLM-First Architecture")
        logger.info("=" * 80)

        # Check GPU
        self.gpu_manager.print_status()

        if not self.gpu_manager.has_gpu:
            logger.warning("No GPU detected! LLM and simulations will run on CPU (slow)")

        # Step 1: Load LLM FIRST
        logger.info("\n[1/3] Loading LLM (this reserves GPU memory first)...")
        self.load_llm()

        # Wait for LLM to fully load
        time.sleep(5)

        # Check remaining memory
        remaining_memory = self.gpu_manager.get_available_memory()
        logger.info(f"\nRemaining GPU memory after LLM: {remaining_memory:.2f} GB")

        # Step 2: Start LLM chat server
        if not self.args.sim_only:
            logger.info("\n[2/3] Starting LLM chat server...")
            self.start_llm_server()

        # Step 3: Start simulations with remaining memory
        if not self.args.llm_only:
            logger.info("\n[3/3] Starting simulations with memory budget...")
            self.start_simulations(remaining_memory)

        logger.info("\n" + "=" * 80)
        logger.info("STARTUP COMPLETE")
        logger.info("=" * 80)

        if not self.args.llm_only and not self.args.sim_only:
            logger.info("\nServices running:")
            logger.info(f"  - LLM Chat: http://localhost:{self.args.llm_port}")
            logger.info(f"  - Simulation Dashboard: http://localhost:{self.args.sim_port}")
            logger.info("\nPress Ctrl+C to stop all services")

        # Keep running
        try:
            if self.llm_process or self.sim_process:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            self.cleanup()

    def load_llm(self):
        """Load LLM and reserve GPU memory"""
        try:
            from ivhl.llm.agent import iVHLAgent, LLMConfig

            logger.info(f"Loading LLM model: {self.args.model}")
            logger.info(f"GPU Memory Allocation: {self.args.llm_memory * 100:.0f}%")

            config = LLMConfig(
                model_name=self.args.model,
                gpu_memory_utilization=self.args.llm_memory,
                max_model_len=4096,
                tensor_parallel_size=1
            )

            # Initialize agent (this loads the model and reserves memory)
            agent = iVHLAgent(config, simulation_manager=None)

            logger.info("✓ LLM loaded successfully")
            logger.info(f"✓ GPU memory reserved for LLM")

            # Store agent globally for other processes to access
            global _global_agent
            _global_agent = agent

            return agent

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            logger.error("Continuing without LLM...")
            return None

    def start_llm_server(self):
        """Start LLM chat server in background"""
        try:
            cmd = [
                sys.executable,
                "dashboards/llm_chat.py",
                "--port", str(self.args.llm_port)
            ]

            logger.info(f"Starting LLM server on port {self.args.llm_port}...")

            self.llm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )

            logger.info("✓ LLM server started")
            logger.info(f"  Access at: http://localhost:{self.args.llm_port}")

        except Exception as e:
            logger.error(f"Failed to start LLM server: {e}")

    def start_simulations(self, available_memory_gb: float):
        """Start simulations with remaining GPU memory"""
        try:
            # Calculate appropriate simulation size based on available memory
            sim_memory = min(available_memory_gb, self.total_memory_gb * self.args.sim_memory)

            logger.info(f"Simulation memory budget: {sim_memory:.2f} GB")

            # Start simulation dashboard
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "dashboards/resonance_dashboard.py",
                "--server.port", str(self.args.sim_port),
                "--server.address", "0.0.0.0"
            ]

            logger.info(f"Starting simulation dashboard on port {self.args.sim_port}...")

            self.sim_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )

            logger.info("✓ Simulation dashboard started")
            logger.info(f"  Access at: http://localhost:{self.args.sim_port}")

        except Exception as e:
            logger.error(f"Failed to start simulations: {e}")

    def cleanup(self):
        """Clean up processes"""
        if self.llm_process:
            logger.info("Stopping LLM server...")
            self.llm_process.terminate()
            self.llm_process.wait()

        if self.sim_process:
            logger.info("Stopping simulations...")
            self.sim_process.terminate()
            self.sim_process.wait()

        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Start iVHL with LLM-first architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full system (LLM + simulations)
  python scripts/start_with_llm.py --full

  # LLM only (for testing)
  python scripts/start_with_llm.py --llm-only

  # Custom memory allocation
  python scripts/start_with_llm.py --llm-memory 0.15 --sim-memory 0.80

  # Different model
  python scripts/start_with_llm.py --model "microsoft/Phi-3-mini-4k-instruct"
        """
    )

    # Mode selection
    parser.add_argument("--full", action="store_true",
                        help="Start both LLM and simulations (default)")
    parser.add_argument("--llm-only", action="store_true",
                        help="Start only LLM server")
    parser.add_argument("--sim-only", action="store_true",
                        help="Start only simulations (no LLM)")

    # Model selection
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="LLM model to use")

    # Memory allocation
    parser.add_argument("--llm-memory", type=float, default=0.15,
                        help="GPU memory fraction for LLM (0.0-1.0, default: 0.15)")
    parser.add_argument("--sim-memory", type=float, default=0.80,
                        help="GPU memory fraction for simulations (0.0-1.0, default: 0.80)")

    # Port configuration
    parser.add_argument("--llm-port", type=int, default=7860,
                        help="LLM chat server port (default: 7860)")
    parser.add_argument("--sim-port", type=int, default=8501,
                        help="Simulation dashboard port (default: 8501)")

    args = parser.parse_args()

    # Default to full mode if nothing specified
    if not args.llm_only and not args.sim_only:
        args.full = True

    # Validate memory allocations
    if args.llm_memory + args.sim_memory > 1.0:
        logger.warning(f"Total memory allocation ({args.llm_memory + args.sim_memory:.2f}) exceeds 100%")
        logger.warning("This may cause OOM errors. Consider reducing allocations.")

    # Start coordinator
    coordinator = StartupCoordinator(args)
    coordinator.start()


if __name__ == "__main__":
    main()
