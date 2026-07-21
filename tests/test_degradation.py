"""Tests for the degradation damping model.

All tests are pure-Python — no PyBullet required.
"""

from __future__ import annotations

import pytest

from src.simulation.degradation import compute_damping_at_cycle


class TestComputeDampingAtCycle:
    """Tests for the pure damping-curve function."""

    def test_returns_base_at_cycle_zero(self) -> None:
        """Damping at cycle 0 must equal base_damping exactly."""
        result = compute_damping_at_cycle(
            cycle=0, total_cycles=300,
            base_damping=0.05, max_damping=5.0, power=3,
        )
        assert result == pytest.approx(0.05)

    def test_returns_max_at_final_cycle(self) -> None:
        """Damping at cycle == total_cycles must equal max_damping exactly."""
        result = compute_damping_at_cycle(
            cycle=300, total_cycles=300,
            base_damping=0.05, max_damping=5.0, power=3,
        )
        assert result == pytest.approx(5.0)

    def test_monotonically_non_decreasing(self) -> None:
        """Damping must never decrease as cycle increases."""
        total = 300
        values = [
            compute_damping_at_cycle(c, total, 0.05, 5.0, power=3)
            for c in range(total + 1)
        ]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], (
                f"damping decreased at cycle {i}: {values[i-1]} → {values[i]}"
            )

    def test_curve_accelerates_with_power_gt_1(self) -> None:
        """With power > 1 the late-stage increase must exceed the early-stage."""
        total = 300
        base, max_d, power = 0.05, 5.0, 3.0

        early_gain = compute_damping_at_cycle(50, total, base, max_d, power) - base
        late_gain = (
            compute_damping_at_cycle(total, total, base, max_d, power)
            - compute_damping_at_cycle(total - 50, total, base, max_d, power)
        )
        assert late_gain > early_gain, (
            f"Expected late-stage gain ({late_gain:.4f}) > early-stage ({early_gain:.4f})"
        )
