"""
Startup Validator for SPOF #6
==============================

Validates environment setup before simulation launch.
Provides clear error messages and graceful degradation.
"""

import sys
import subprocess
from typing import Dict, List, Tuple
from pathlib import Path


class StartupValidator:
    """
    Validates system setup and dependencies.

    Checks:
    - Python version
    - Critical imports (torch, numpy, etc.)
    - GPU availability
    - LaTeX installation
    - vLLM server connectivity
    - File system permissions
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, fail on any error. If False, warn and continue.
        """
        self.strict_mode = strict_mode
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

    def _check_python_version(self) -> Tuple[bool, str]:
        """Check Python version is 3.8+."""
        version_info = sys.version_info

        if version_info.major == 3 and version_info.minor >= 8:
            return True, f"Python {version_info.major}.{version_info.minor}.{version_info.micro}"
        else:
            return False, f"Python {version_info.major}.{version_info.minor} (requires 3.8+)"

    def _check_import(self, module_name: str, package_name: str = None) -> Tuple[bool, str]:
        """
        Check if module can be imported.

        Args:
            module_name: Name to import
            package_name: pip package name (if different from module)
        """
        if package_name is None:
            package_name = module_name

        try:
            __import__(module_name)
            return True, f"{package_name} available"
        except ImportError as e:
            return False, f"{package_name} not found: {e}"

    def _check_torch_gpu(self) -> Tuple[bool, str]:
        """Check PyTorch GPU availability."""
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return True, f"GPU available: {device_name} ({vram_gb:.1f} GB VRAM)"
            else:
                return False, "No GPU detected (will use CPU fallback)"

        except ImportError:
            return False, "PyTorch not installed"

    def _check_vllm_server(self, url: str = "http://localhost:8000/v1/models") -> Tuple[bool, str]:
        """Check if vLLM server is reachable."""
        try:
            import requests

            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True, "vLLM server online"
            else:
                return False, f"vLLM server returned status {response.status_code}"

        except ImportError:
            return False, "requests module not available"
        except Exception as e:
            return False, f"vLLM server unreachable: {e}"

    def _check_pdflatex(self) -> Tuple[bool, str]:
        """Check if pdflatex is installed."""
        try:
            result = subprocess.run(
                ['pdflatex', '--version'],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                return True, "pdflatex available"
            else:
                return False, "pdflatex not working"

        except FileNotFoundError:
            return False, "pdflatex not installed"
        except Exception as e:
            return False, f"pdflatex check failed: {e}"

    def _check_directory_writable(self, path: str) -> Tuple[bool, str]:
        """Check if directory is writable."""
        dir_path = Path(path)

        try:
            # Create directory if doesn't exist
            dir_path.mkdir(parents=True, exist_ok=True)

            # Try to write test file
            test_file = dir_path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()

            return True, f"{path} is writable"

        except Exception as e:
            return False, f"{path} not writable: {e}"

    def run_all_checks(self) -> Dict[str, bool]:
        """
        Run all validation checks.

        Returns:
            Dictionary of check results
        """
        print("="*60)
        print("iVHL Framework Startup Validation")
        print("="*60)

        results = {}

        # Critical checks (must pass)
        critical_checks = [
            ("Python Version", self._check_python_version),
            ("NumPy", lambda: self._check_import("numpy")),
            ("PyTorch", lambda: self._check_import("torch")),
            ("SciPy", lambda: self._check_import("scipy")),
            ("Results Directory", lambda: self._check_directory_writable("/results")),
        ]

        print("\n🔍 Critical Checks:")
        for name, check_func in critical_checks:
            passed, message = check_func()
            results[name] = passed

            if passed:
                print(f"  ✅ {name}: {message}")
                self.checks_passed.append(name)
            else:
                print(f"  ❌ {name}: {message}")
                self.checks_failed.append(name)

                if self.strict_mode:
                    print(f"\n❌ CRITICAL: {name} check failed in strict mode")
                    sys.exit(1)

        # Optional checks (can fail with warning)
        optional_checks = [
            ("GPU", self._check_torch_gpu),
            ("vLLM Server", self._check_vllm_server),
            ("pdflatex", self._check_pdflatex),
            ("FastAPI", lambda: self._check_import("fastapi")),
            ("PyVista", lambda: self._check_import("pyvista")),
            ("Matplotlib", lambda: self._check_import("matplotlib")),
        ]

        print("\n🔧 Optional Checks:")
        for name, check_func in optional_checks:
            passed, message = check_func()
            results[name] = passed

            if passed:
                print(f"  ✅ {name}: {message}")
                self.checks_passed.append(name)
            else:
                print(f"  ⚠️  {name}: {message}")
                self.warnings.append(f"{name}: {message}")

        # Summary
        print("\n" + "="*60)
        print("Validation Summary")
        print("="*60)

        print(f"✅ Passed: {len(self.checks_passed)}")
        print(f"❌ Failed: {len(self.checks_failed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")

        if self.checks_failed:
            print("\n❌ CRITICAL FAILURES:")
            for check in self.checks_failed:
                print(f"   - {check}")
            print("\nSimulation cannot start. Fix critical issues above.")
            return results

        if self.warnings:
            print("\n⚠️  WARNINGS (simulation will continue with reduced functionality):")
            for warning in self.warnings:
                print(f"   - {warning}")

        print("\n✅ Validation complete. Starting simulation...\n")
        return results

    def get_degraded_config(self, results: Dict[str, bool]) -> Dict:
        """
        Generate degraded configuration based on failed checks.

        Args:
            results: Check results from run_all_checks()

        Returns:
            Configuration dict with appropriate fallbacks
        """
        config = {
            'device': 'cuda' if results.get('GPU', False) else 'cpu',
            'llm_enabled': results.get('vLLM Server', False),
            'pdf_output': results.get('pdflatex', False),
            'web_monitor': results.get('FastAPI', False),
            'gpu_rendering': results.get('PyVista', False) and results.get('GPU', False),
        }

        # Adjust parameters based on device
        if config['device'] == 'cpu':
            config['base_dimension'] = 16  # Minimal for CPU
            config['bond_dimension'] = 4
            config['timesteps'] = 50
        else:
            config['base_dimension'] = 64  # Default for GPU
            config['bond_dimension'] = 16
            config['timesteps'] = 300

        return config


def validate_startup(strict: bool = False) -> Dict:
    """
    Convenience function to run startup validation.

    Args:
        strict: Whether to exit on any failure

    Returns:
        Configuration dict with appropriate fallbacks
    """
    validator = StartupValidator(strict_mode=strict)
    results = validator.run_all_checks()

    if validator.checks_failed and strict:
        print("\n❌ Startup validation failed in strict mode. Exiting.")
        sys.exit(1)

    config = validator.get_degraded_config(results)

    print("\n📋 Generated Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    return config


if __name__ == "__main__":
    # Test startup validator
    import argparse

    parser = argparse.ArgumentParser(description="iVHL Startup Validator")
    parser.add_argument(
        '--strict',
        action='store_true',
        help="Exit on any critical failure"
    )

    args = parser.parse_args()

    config = validate_startup(strict=args.strict)

    print("\n✅ Validation complete")
