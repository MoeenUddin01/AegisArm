"""Tests for Phase 4 joint-health training pipeline.

These tests verify structural invariants of the training configuration
and data split — not the trained model itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.train_joint_health import FEATURE_COLS, _split_by_run_id


class TestFeatureColumnSafety:
    """Guards against label leakage through feature columns."""

    def test_feature_cols_exclude_damping(self) -> None:
        """'damping' must never appear in FEATURE_COLS — it leaks the label."""
        assert "damping" not in FEATURE_COLS
        assert "current_damping" not in FEATURE_COLS

    def test_feature_cols_exclude_cycle(self) -> None:
        """'cycle' must never appear in FEATURE_COLS — it is a proxy for
        cycles_to_failure (total_cycles - cycle) and would leak the label."""
        assert "cycle" not in FEATURE_COLS

    def test_feature_cols_is_exact_allowlist(self) -> None:
        """FEATURE_COLS must be the three explicitly chosen features."""
        assert FEATURE_COLS == [
            "torque_rms", "velocity_rms", "position_error_mean",
        ]


class TestTrainValSplit:
    """Verify the run_id-based train/val split."""

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        """DataFrame with 10 runs, 5 cycles each."""
        rows = []
        for run in range(10):
            for cyc in range(5):
                rows.append({
                    "run_id": run,
                    "cycle": cyc,
                    "torque_rms": 1.0,
                    "velocity_rms": 0.5,
                    "position_error_mean": 0.01,
                    "cycles_to_failure": 5 - cyc,
                })
        return pd.DataFrame(rows)

    def test_train_val_run_ids_disjoint(self, sample_df: pd.DataFrame) -> None:
        """Train and validation run_id sets must have no overlap."""
        train_df, val_df = _split_by_run_id(sample_df, val_split_runs=3, seed=42)
        train_runs = set(train_df["run_id"].unique())
        val_runs = set(val_df["run_id"].unique())
        assert train_runs.isdisjoint(val_runs), (
            f"Overlap: {train_runs & val_runs}"
        )

    def test_all_runs_assigned(self, sample_df: pd.DataFrame) -> None:
        """Every run_id must appear in either train or val (no data lost)."""
        train_df, val_df = _split_by_run_id(sample_df, val_split_runs=3, seed=42)
        all_runs = set(sample_df["run_id"].unique())
        assigned = set(train_df["run_id"].unique()) | set(val_df["run_id"].unique())
        assert assigned == all_runs
