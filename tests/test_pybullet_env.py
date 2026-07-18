"""Tests for the PyBullet environment module.

``test_compute_sine_target`` is pure-Python and needs no PyBullet
installation.  ``test_run_demo_headless`` uses ``p.DIRECT`` (no display)
and requires ``pybullet`` to be installed — skip gracefully if it is
not.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulation.pybullet_env import compute_sine_target

# ---------------------------------------------------------------------------
# Pure-Python tests (no PyBullet required)
# ---------------------------------------------------------------------------


class TestComputeSineTarget:
    """Tests for compute_sine_target — pure function, no PyBullet."""

    def test_returns_zero_at_step_zero(self) -> None:
        """sin(0) == 0, so the target must be 0 at step 0."""
        result = compute_sine_target(step=0, amplitude_rad=1.0, frequency_hz=0.5, sim_timestep=1 / 240)
        assert result == pytest.approx(0.0)

    def test_oscillates_within_amplitude(self) -> None:
        """Output must never exceed ±amplitude_rad."""
        amp = 0.6
        freq = 0.2
        dt = 1 / 240
        for step in range(10000):
            val = compute_sine_target(step, amp, freq, dt)
            assert -amp - 1e-10 <= val <= amp + 1e-10, (
                f"step {step}: {val} outside [-{amp}, {amp}]"
            )

    def test_period_is_correct(self) -> None:
        """The sine should complete one full cycle at t = 1/frequency."""
        amp = 1.0
        freq = 0.5  # period = 2.0 s
        dt = 0.01
        period_steps = int(1.0 / (freq * dt))

        # At t=0 and t=period, sin should be ~0
        assert compute_sine_target(0, amp, freq, dt) == pytest.approx(0.0, abs=1e-10)
        assert compute_sine_target(period_steps, amp, freq, dt) == pytest.approx(
            0.0, abs=1e-3
        )

        # At t=period/4, sin should be ~+amplitude
        quarter_steps = period_steps // 4
        assert compute_sine_target(quarter_steps, amp, freq, dt) == pytest.approx(
            amp, abs=1e-2
        )

    def test_symmetry(self) -> None:
        """sin(pi) should give approximately 0, sin(pi/2) should be +amp."""
        amp = 2.0
        freq = 1.0
        dt = 1 / 240
        quarter = int(1.0 / (4.0 * freq * dt))
        half = 2 * quarter
        assert compute_sine_target(quarter, amp, freq, dt) == pytest.approx(amp, abs=1e-2)
        assert compute_sine_target(half, amp, freq, dt) == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Headless PyBullet test (requires pybullet)
# ---------------------------------------------------------------------------

try:
    import pybullet as p  # noqa: F401

    _HAS_PYBULLET = True
except ImportError:
    _HAS_PYBULLET = False


@pytest.mark.skipif(not _HAS_PYBULLET, reason="pybullet not installed")
class TestRunDemoHeadless:
    """Integration test using p.DIRECT (no display)."""

    def test_returns_expected_shape(self, tmp_path) -> None:
        """DataFrame should have num_steps rows and the expected columns."""
        num_steps = 20
        log_path = tmp_path / "test_log.csv"
        config = {
            "joint_index": 3,
            "amplitude_rad": 0.6,
            "frequency_hz": 0.2,
            "sim_timestep": 0.0041666,
            "num_steps": num_steps,
            "log_path": str(log_path),
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from src.simulation.pybullet_env import run_demo

        df = run_demo(config_path=str(config_path), gui=False)

        assert len(df) == num_steps
        assert list(df.columns) == ["step", "angle", "velocity", "applied_torque"]
        assert not df.isna().any().any()
        assert (df["step"] == range(num_steps)).all()

    def test_angle_stays_in_bounds(self, tmp_path) -> None:
        """Angle should stay within ±amplitude during the demo."""
        config = {
            "joint_index": 3,
            "amplitude_rad": 0.5,
            "frequency_hz": 0.2,
            "sim_timestep": 0.0041666,
            "num_steps": 30,
            "log_path": str(tmp_path / "log.csv"),
        }
        config_path = tmp_path / "cfg.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from src.simulation.pybullet_env import run_demo

        df = run_demo(config_path=str(config_path), gui=False)

        # Joint angle should stay within ±amplitude (approximately)
        assert df["angle"].abs().max() <= 0.5 + 0.2  # allow some overshoot
