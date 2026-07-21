"""Tests for multi-run generation and cycle-level aggregation.

All tests that call ``generate_multi_run_dataset`` run a tiny config
(2 cycles, 2 runs) headless — fast enough for CI.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data.synthetic_joint_generator import (
    aggregate_cycle_features,
    generate_multi_run_dataset,
)


def _make_raw_df(cycles: int = 3, steps_per_cycle: int = 5) -> pd.DataFrame:
    """Build a minimal synthetic raw-step DataFrame for testing."""
    rows = []
    for c in range(cycles):
        for s in range(steps_per_cycle):
            rows.append({
                "step": c * steps_per_cycle + s,
                "cycle": c,
                "target_angle": 0.1,
                "actual_angle": 0.09,
                "position_error": 0.01,
                "velocity": 1.0 + c * 0.5,
                "applied_torque": 2.0 + c * 1.0,
                "current_damping": 0.05 + c * 0.1,
                "cycles_to_failure": cycles - c,
            })
    return pd.DataFrame(rows)


class TestAggregateCycleFeatures:
    """Tests for per-cycle aggregation."""

    def test_one_row_per_cycle(self) -> None:
        """Output must have exactly one row per distinct cycle value."""
        raw = _make_raw_df(cycles=5, steps_per_cycle=10)
        result = aggregate_cycle_features(raw, run_id=0)
        assert len(result) == 5
        assert list(result["cycle"]) == [0, 1, 2, 3, 4]

    def test_output_columns_exclude_damping(self) -> None:
        """The aggregated output must NOT contain a 'damping' column."""
        raw = _make_raw_df(cycles=3)
        result = aggregate_cycle_features(raw, run_id=0)
        assert "damping" not in result.columns
        assert "current_damping" not in result.columns

    def test_expected_columns_present(self) -> None:
        """Output must contain all required feature columns."""
        raw = _make_raw_df(cycles=3)
        result = aggregate_cycle_features(raw, run_id=7)
        expected = {
            "cycle", "torque_rms", "velocity_rms",
            "position_error_mean", "cycles_to_failure", "run_id",
        }
        assert set(result.columns) == expected
        assert (result["run_id"] == 7).all()


class TestMultiRunRandomization:
    """Verify that different runs actually use different parameters."""

    @pytest.fixture()
    def tiny_config(self, tmp_path: Path) -> str:
        """Create a minimal 2-run config for fast testing."""
        cfg = textwrap.dedent("""\
            joint_index: 3
            amplitude_rad: 0.6
            frequency_hz: 0.2
            sim_timestep: 0.0041666
            num_steps: 1000
            log_path: "outputs/logs/phase2_joint_readout.csv"
            degradation:
              base_damping: 0.05
              max_damping: 50.0
              degradation_power: 3
              total_cycles: 2
              output_csv: "outputs/logs/_test_raw.csv"
              num_runs: 2
              damping_variation_pct: 0.15
              power_variation: 0.5
              seed_base: 1000
              multirun_output_csv: "outputs/logs/_test_multirun.csv"
        """)
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(cfg)
        return str(config_path)

    def test_runs_produce_different_outputs(self, tiny_config: str) -> None:
        """Different run_ids must yield different torque_rms at final cycle.

        If the same config were used for every run, torque_rms values
        would be identical — this confirms randomization is applied.
        """
        df = generate_multi_run_dataset(tiny_config)
        final_cycle = df["cycle"].max()
        final = df[df["cycle"] == final_cycle]

        assert len(final) == 2
        rms_values = final["torque_rms"].values
        assert rms_values[0] != pytest.approx(rms_values[1], rel=1e-3), (
            f"Run outputs identical torque_rms ({rms_values[0]:.6f}) — "
            "randomization may not be applied"
        )
