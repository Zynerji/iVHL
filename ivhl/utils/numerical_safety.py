"""
Numerical Stability Checker for SPOF #4
========================================

Detects and handles NaN/Inf values in tensor computations.
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple
from functools import wraps


class NumericalInstabilityError(Exception):
    """Raised when numerical instability detected and cannot be recovered."""
    pass


class NumericalSafetyChecker:
    """
    Monitors tensor computations for numerical stability.

    Detects:
    - NaN (Not a Number) values
    - Inf (Infinity) values
    - Near-zero denominators
    - Exploding gradients
    """

    def __init__(
        self,
        enable_warnings: bool = True,
        auto_fix: bool = True,
        max_failures: int = 10
    ):
        self.enable_warnings = enable_warnings
        self.auto_fix = auto_fix
        self.max_failures = max_failures
        self.failure_count = 0
        self.failure_log = []

    def check_tensor(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        name: str = "tensor",
        raise_on_error: bool = False
    ) -> Tuple[bool, str]:
        """
        Check tensor for numerical issues.

        Args:
            tensor: Tensor to check
            name: Name for error reporting
            raise_on_error: Whether to raise exception on detection

        Returns:
            (is_valid, message)
        """
        # Convert numpy to torch for consistent checking
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        # Check for NaN
        has_nan = torch.isnan(tensor).any().item()
        if has_nan:
            msg = f"❌ NaN detected in {name}"
            self._log_failure(msg)

            if raise_on_error:
                raise NumericalInstabilityError(msg)
            return False, msg

        # Check for Inf
        has_inf = torch.isinf(tensor).any().item()
        if has_inf:
            msg = f"❌ Inf detected in {name}"
            self._log_failure(msg)

            if raise_on_error:
                raise NumericalInstabilityError(msg)
            return False, msg

        # Check for extremely large values (potential overflow)
        max_val = tensor.abs().max().item()
        if max_val > 1e10:
            msg = f"⚠️  Large values detected in {name}: max={max_val:.2e}"
            if self.enable_warnings:
                print(msg)
            self.failure_log.append(msg)
            # Don't count as failure, just warning

        # Check for extremely small values (potential underflow)
        nonzero_mask = tensor != 0
        if nonzero_mask.any():
            min_val = tensor[nonzero_mask].abs().min().item()
            if min_val < 1e-10:
                msg = f"⚠️  Small values detected in {name}: min={min_val:.2e}"
                if self.enable_warnings:
                    print(msg)

        return True, "OK"

    def _log_failure(self, message: str):
        """Log numerical failure."""
        self.failure_count += 1
        self.failure_log.append(message)

        if self.enable_warnings:
            print(message)

        if self.failure_count >= self.max_failures:
            raise NumericalInstabilityError(
                f"Too many numerical failures ({self.failure_count}). Aborting."
            )

    def fix_tensor(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        strategy: str = "zero"
    ) -> torch.Tensor:
        """
        Attempt to fix numerical issues in tensor.

        Args:
            tensor: Tensor with potential issues
            name: Name for logging
            strategy: Fix strategy ("zero", "clamp", "interpolate")

        Returns:
            Fixed tensor
        """
        fixed = tensor.clone()

        if strategy == "zero":
            # Replace NaN/Inf with zeros
            fixed = torch.nan_to_num(fixed, nan=0.0, posinf=0.0, neginf=0.0)

        elif strategy == "clamp":
            # Clamp to reasonable range
            fixed = torch.nan_to_num(fixed, nan=0.0, posinf=1e10, neginf=-1e10)
            fixed = torch.clamp(fixed, min=-1e10, max=1e10)

        elif strategy == "interpolate":
            # Try to interpolate from neighbors (simple nearest-neighbor)
            nan_mask = torch.isnan(fixed)
            inf_mask = torch.isinf(fixed)
            bad_mask = nan_mask | inf_mask

            if bad_mask.any():
                # Replace with local mean (simple approach)
                good_values = fixed[~bad_mask]
                if good_values.numel() > 0:
                    replacement_value = good_values.mean()
                else:
                    replacement_value = 0.0

                fixed[bad_mask] = replacement_value

        else:
            raise ValueError(f"Unknown fix strategy: {strategy}")

        if self.enable_warnings:
            num_fixed = (tensor != fixed).sum().item()
            if num_fixed > 0:
                print(f"🔧 Fixed {num_fixed} values in {name} using '{strategy}' strategy")

        return fixed

    def safe_divide(
        self,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        Safe division that avoids divide-by-zero.

        Args:
            numerator: Numerator tensor
            denominator: Denominator tensor
            epsilon: Small value to add to denominator

        Returns:
            Result of division
        """
        safe_denom = denominator.clone()

        # Add epsilon where denominator is near zero
        near_zero = safe_denom.abs() < epsilon
        safe_denom[near_zero] += epsilon

        return numerator / safe_denom

    def safe_log(
        self,
        tensor: torch.Tensor,
        epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        Safe logarithm that avoids log(0).

        Args:
            tensor: Input tensor
            epsilon: Small value to add before log

        Returns:
            Log of tensor
        """
        return torch.log(tensor + epsilon)

    def safe_sqrt(
        self,
        tensor: torch.Tensor,
        epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        Safe square root that avoids negative values.

        Args:
            tensor: Input tensor
            epsilon: Small value to add before sqrt

        Returns:
            Square root of tensor
        """
        # Clamp to non-negative
        clamped = torch.clamp(tensor, min=0.0)
        return torch.sqrt(clamped + epsilon)


def check_numerical_stability(name: str = "result", auto_fix: bool = True):
    """
    Decorator to automatically check function outputs for numerical stability.

    Usage:
        @check_numerical_stability("svd_result")
        def compute_svd(tensor):
            return torch.svd(tensor)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            checker = NumericalSafetyChecker(auto_fix=auto_fix)

            try:
                result = func(*args, **kwargs)

                # Check result
                if isinstance(result, (torch.Tensor, np.ndarray)):
                    is_valid, msg = checker.check_tensor(result, name=name)

                    if not is_valid and auto_fix:
                        print(f"🔧 Auto-fixing {name}")
                        result = checker.fix_tensor(result, name=name)

                elif isinstance(result, (list, tuple)):
                    # Check each element
                    fixed_results = []
                    for i, item in enumerate(result):
                        if isinstance(item, (torch.Tensor, np.ndarray)):
                            is_valid, msg = checker.check_tensor(item, name=f"{name}[{i}]")

                            if not is_valid and auto_fix:
                                item = checker.fix_tensor(item, name=f"{name}[{i}]")

                        fixed_results.append(item)

                    result = type(result)(fixed_results)  # Preserve tuple/list type

                return result

            except Exception as e:
                print(f"❌ Error in {func.__name__}: {e}")
                raise

        return wrapper
    return decorator


# Global instance for convenience
global_checker = NumericalSafetyChecker()


def check_tensor(tensor: Union[torch.Tensor, np.ndarray], name: str = "tensor") -> bool:
    """Convenience function using global checker."""
    is_valid, msg = global_checker.check_tensor(tensor, name=name)
    return is_valid


def fix_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Convenience function using global checker."""
    return global_checker.fix_tensor(tensor, name=name)


if __name__ == "__main__":
    # Test numerical safety checker
    print("="*60)
    print("Numerical Safety Checker Tests")
    print("="*60)

    checker = NumericalSafetyChecker()

    # Test 1: Normal tensor
    print("\nTest 1: Normal tensor")
    normal = torch.randn(10, 10)
    is_valid, msg = checker.check_tensor(normal, "normal")
    print(f"  Result: {msg}")

    # Test 2: Tensor with NaN
    print("\nTest 2: Tensor with NaN")
    with_nan = torch.randn(10, 10)
    with_nan[5, 5] = float('nan')
    is_valid, msg = checker.check_tensor(with_nan, "with_nan")
    print(f"  Result: {msg}")

    # Test 3: Fix NaN tensor
    print("\nTest 3: Fix NaN tensor")
    fixed = checker.fix_tensor(with_nan, "with_nan", strategy="zero")
    is_valid, msg = checker.check_tensor(fixed, "fixed")
    print(f"  Result: {msg}")

    # Test 4: Safe division
    print("\nTest 4: Safe division")
    numerator = torch.randn(5)
    denominator = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0])
    result = checker.safe_divide(numerator, denominator)
    print(f"  Result: {result}")
    print(f"  Valid: {checker.check_tensor(result, 'division_result')[0]}")

    # Test 5: Decorator
    print("\nTest 5: Decorator")

    @check_numerical_stability("test_output")
    def risky_computation():
        tensor = torch.randn(5, 5)
        tensor[2, 2] = float('inf')
        return tensor

    result = risky_computation()
    print(f"  Result shape: {result.shape}")
    print(f"  Has inf: {torch.isinf(result).any().item()}")
