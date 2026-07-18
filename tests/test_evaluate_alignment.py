"""Tests for the evaluate_cmapss alignment logic.

Specifically tests that ``y_true`` is correctly filtered and ordered
to match ``y_pred`` when skipped units are in the MIDDLE of the unit
range (not just at edges).  This is the regression test for the bug
where positional slicing caused y_true/y_pred to compare predictions
against the wrong engines.

The alignment logic is duplicated here (rather than imported from
``scripts/evaluate_cmapss``) to avoid pulling in ``torch`` as a test
dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — mirrors _last_window_per_unit + alignment from evaluate_cmapss.py
# ---------------------------------------------------------------------------

def _last_window_per_unit(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, list[int]]:
    """Extract the last *window_size* cycles per unit; skip short units.

    Returns ``(X, surviving_unit_ids)``.
    """
    windows: list[np.ndarray] = []
    surviving: list[int] = []
    for _, grp in df.groupby("unit_number", sort=True):
        unit_id = int(grp["unit_number"].iloc[0])
        tail = grp.tail(window_size)
        if len(tail) < window_size:
            continue
        windows.append(tail[feature_cols].values.astype(np.float32))
        surviving.append(unit_id)
    if not windows:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), []
    return np.stack(windows), surviving


def _align_y_true(
    rul_df: pd.DataFrame,
    surviving_units: list[int],
) -> np.ndarray:
    """Filter and order RUL ground truth to match prediction order.

    This is the corrected logic from evaluate_cmapss.py — join on
    unit_number, NOT positional slicing.
    """
    surviving_set = set(surviving_units)
    filtered = rul_df[rul_df["unit_number"].isin(surviving_set)].copy()
    filtered["order"] = filtered["unit_number"].map(
        {u: i for i, u in enumerate(surviving_units)}
    )
    filtered = filtered.sort_values("order").reset_index(drop=True)
    return filtered["RUL"].values.astype(np.float64)


def _make_test_df(
    units: list[int],
    rows_per_unit: dict[int, int],
) -> pd.DataFrame:
    """Build a synthetic test DataFrame with specified per-unit row counts."""
    feature_cols = ["feat_a", "feat_b"]
    rows = []
    for u in units:
        for t in range(1, rows_per_unit[u] + 1):
            row: dict[str, object] = {"unit_number": u, "time_in_cycles": t}
            for fi, fc in enumerate(feature_cols):
                row[fc] = float(u * 100 + t + fi)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_rul_df(units: list[int], rul_values: list[int]) -> pd.DataFrame:
    """Build a synthetic RUL ground-truth DataFrame."""
    return pd.DataFrame({"unit_number": units, "RUL": rul_values})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLastWindowPerUnit:
    """Tests for _last_window_per_unit."""

    def test_returns_surviving_unit_ids(self) -> None:
        """surviving_units must list only units that had enough rows."""
        df = _make_test_df(
            units=[1, 2, 3, 4, 5],
            rows_per_unit={1: 10, 2: 3, 3: 10, 4: 2, 5: 10},
        )
        X, unit_ids = _last_window_per_unit(df, ["feat_a", "feat_b"], window_size=5)

        assert unit_ids == [1, 3, 5]
        assert X.shape[0] == 3

    def test_skipped_units_in_middle(self) -> None:
        """Short units between long units must be skipped cleanly."""
        df = _make_test_df(
            units=[1, 2, 3, 4, 5],
            rows_per_unit={1: 20, 2: 3, 3: 20, 4: 4, 5: 20},
        )
        X, unit_ids = _last_window_per_unit(df, ["feat_a", "feat_b"], window_size=10)

        assert unit_ids == [1, 3, 5]
        assert len(unit_ids) == X.shape[0]

    def test_all_short_returns_empty(self) -> None:
        """When every unit is too short, return empty arrays and empty list."""
        df = _make_test_df(
            units=[1, 2, 3],
            rows_per_unit={1: 2, 2: 3, 3: 1},
        )
        X, unit_ids = _last_window_per_unit(df, ["feat_a", "feat_b"], window_size=10)

        assert unit_ids == []
        assert X.shape[0] == 0

    def test_no_skips(self) -> None:
        """When all units are long enough, all should survive."""
        df = _make_test_df(
            units=[10, 20, 30],
            rows_per_unit={10: 15, 20: 15, 30: 15},
        )
        X, unit_ids = _last_window_per_unit(df, ["feat_a", "feat_b"], window_size=5)

        assert unit_ids == [10, 20, 30]


class TestRulAlignment:
    """Tests that y_true is correctly aligned to y_pred via unit_number join."""

    def test_mid_range_skips_aligned_correctly(self) -> None:
        """Skipped units in the MIDDLE must not cause positional misalignment.

        This is the regression test: y_true had 100 entries (all units),
        y_pred had 93 (surviving units only), and the 7 skipped units
        were scattered throughout the range.  Positional slicing would
        silently compare predictions against the wrong engines.
        """
        all_units = [1, 2, 3, 4, 5, 6, 7, 8]
        # Short units (will be skipped): 2, 4, 6 — in the MIDDLE
        rows_per_unit = {
            1: 20, 2: 3, 3: 20, 4: 2, 5: 20, 6: 4, 7: 20, 8: 20,
        }
        rul_values = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60, 6: 50, 7: 40, 8: 30}

        df = _make_test_df(units=all_units, rows_per_unit=rows_per_unit)
        rul_df = _make_rul_df(all_units, [rul_values[u] for u in all_units])

        X, surviving_units = _last_window_per_unit(
            df, ["feat_a", "feat_b"], window_size=10
        )
        y_true = _align_y_true(rul_df, surviving_units)

        # Lengths match
        assert len(y_true) == X.shape[0]

        # Values match the correct units (1, 3, 5, 7, 8) in order
        expected_rul = [rul_values[u] for u in surviving_units]
        np.testing.assert_array_equal(y_true, expected_rul)

    def test_positional_slicing_would_fail(self) -> None:
        """Demonstrate that the old positional approach gives wrong results.

        If we naively took the first N rows from RUL_FD001.txt (or the
        last N), the values would NOT match the surviving units.
        """
        all_units = [1, 2, 3, 4, 5]
        rows_per_unit = {1: 20, 2: 3, 3: 20, 4: 2, 5: 20}
        rul_values = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60}

        df = _make_test_df(units=all_units, rows_per_unit=rows_per_unit)
        rul_df = _make_rul_df(all_units, [rul_values[u] for u in all_units])

        X, surviving_units = _last_window_per_unit(
            df, ["feat_a", "feat_b"], window_size=10
        )
        y_true_correct = _align_y_true(rul_df, surviving_units)

        # The WRONG way: positional slice of all RUL values
        y_true_wrong = rul_df.sort_values("unit_number")["RUL"].values.astype(
            np.float64
        )[: len(surviving_units)]

        # They must differ — proving positional slicing is incorrect
        assert not np.array_equal(y_true_correct, y_true_wrong), (
            "Positional slicing happened to match — bad test design"
        )
        # Correct alignment: [100, 80, 60] (units 1, 3, 5)
        np.testing.assert_array_equal(y_true_correct, [100.0, 80.0, 60.0])
        # Wrong positional slice: [100, 90, 80] (units 1, 2, 3)
        np.testing.assert_array_equal(y_true_wrong, [100.0, 90.0, 80.0])

    def test_surviving_units_order_matches_predictions(self) -> None:
        """The order of y_true must follow surviving_units order exactly."""
        all_units = [10, 20, 30, 40, 50]
        rows_per_unit = {10: 15, 20: 2, 30: 15, 40: 3, 50: 15}
        rul_values = {10: 10, 20: 20, 30: 30, 40: 40, 50: 50}

        df = _make_test_df(units=all_units, rows_per_unit=rows_per_unit)
        rul_df = _make_rul_df(all_units, [rul_values[u] for u in all_units])

        _, surviving_units = _last_window_per_unit(
            df, ["feat_a", "feat_b"], window_size=10
        )
        y_true = _align_y_true(rul_df, surviving_units)

        assert surviving_units == [10, 30, 50]
        np.testing.assert_array_equal(y_true, [10.0, 30.0, 50.0])
