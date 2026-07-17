"""Tests for the C-MAPSS data loader.

These tests use small synthetic in-memory DataFrames so they run without
any real data files present.
"""

import pandas as pd
import pytest

from src.data.cmapss_loader import (
    compute_rul_labels,
    identify_constant_sensors,
    load_test_rul,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_df() -> pd.DataFrame:
    """Build a tiny synthetic training DataFrame (2 units, 3 cycles each)."""
    rows = []
    for unit in [1, 2]:
        for cycle in [1, 2, 3]:
            rows.append({
                "unit_number": unit,
                "time_in_cycles": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 0.0,
                "sensor_1": 100.0 + cycle,
                "sensor_2": 200.0 + cycle,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_rul_labels tests
# ---------------------------------------------------------------------------

class TestComputeRulLabels:
    """Tests for compute_rul_labels."""

    def test_rul_zero_at_final_cycle(self) -> None:
        """RUL must be 0 at each unit's last cycle."""
        df = _make_training_df()
        labelled = compute_rul_labels(df)

        for unit in [1, 2]:
            final_cycle = df.loc[df["unit_number"] == unit, "time_in_cycles"].max()
            rul = labelled.loc[
                (labelled["unit_number"] == unit)
                & (labelled["time_in_cycles"] == final_cycle),
                "RUL",
            ]
            assert rul.iloc[0] == 0, f"RUL at final cycle for unit {unit} should be 0"

    def test_rul_decreases_monotonically(self) -> None:
        """RUL should decrease as cycles advance within each unit."""
        df = _make_training_df()
        labelled = compute_rul_labels(df)

        for unit in [1, 2]:
            unit_ruls = labelled.loc[
                labelled["unit_number"] == unit, "RUL"
            ].tolist()
            assert unit_ruls == sorted(unit_ruls, reverse=True), (
                f"RUL for unit {unit} is not monotonically decreasing"
            )

    def test_preserves_original_columns(self) -> None:
        """The returned DataFrame must contain all original columns plus RUL."""
        df = _make_training_df()
        labelled = compute_rul_labels(df)
        assert "RUL" in labelled.columns
        for col in df.columns:
            assert col in labelled.columns


# ---------------------------------------------------------------------------
# identify_constant_sensors tests
# ---------------------------------------------------------------------------

class TestIdentifyConstantSensors:
    """Tests for identify_constant_sensors."""

    def test_detects_constant_sensor(self) -> None:
        """A sensor with zero variance should be flagged."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "time_in_cycles": [1, 2, 3],
            "sensor_1": [5.0, 5.0, 5.0],
            "sensor_2": [1.0, 2.0, 3.0],
        })
        const = identify_constant_sensors(df)
        assert "sensor_1" in const
        assert "sensor_2" not in const

    def test_no_constant_sensors(self) -> None:
        """When all sensors vary, the result should be empty."""
        df = pd.DataFrame({
            "sensor_1": [1.0, 2.0, 3.0],
            "sensor_2": [4.0, 5.0, 6.0],
        })
        assert identify_constant_sensors(df) == []


# ---------------------------------------------------------------------------
# load_test_rul tests (synthetic file)
# ---------------------------------------------------------------------------

class TestLoadTestRul:
    """Tests for load_test_rul using a synthetic temporary file."""

    def test_returns_one_row_per_unit(self, tmp_path) -> None:
        """RUL file with N lines should produce N rows."""
        rul_file = tmp_path / "RUL_FD001.txt"
        rul_file.write_text("12\n7\n3\n")

        result = load_test_rul(str(rul_file))

        assert len(result) == 3
        assert list(result.columns) == ["unit_number", "RUL"]
        assert list(result["unit_number"]) == [1, 2, 3]
        assert list(result["RUL"]) == [12, 7, 3]

    def test_single_unit(self, tmp_path) -> None:
        """A single-line RUL file should still return a proper DataFrame."""
        rul_file = tmp_path / "RUL_single.txt"
        rul_file.write_text("25\n")

        result = load_test_rul(str(rul_file))

        assert len(result) == 1
        assert result["RUL"].iloc[0] == 25
