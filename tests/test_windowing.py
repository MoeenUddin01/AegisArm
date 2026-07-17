"""Tests for the windowing module.

All tests use small synthetic in-memory DataFrames so they run without
any data files present.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.data.windowing import create_sliding_windows, normalize_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_grouped_df(
    num_groups: int = 3,
    rows_per_group: int = 10,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic DataFrame with numbered groups.

    Each group gets monotonically increasing feature values so windows
    are easy to reason about.
    """
    if feature_cols is None:
        feature_cols = ["feat_a", "feat_b"]

    rows = []
    for g in range(1, num_groups + 1):
        for t in range(1, rows_per_group + 1):
            row: dict[str, object] = {
                "group_id": g,
                "time": t,
                "label": float(t * g),
            }
            for fi, fc in enumerate(feature_cols):
                row[fc] = float(g * 100 + t + fi)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# create_sliding_windows tests
# ---------------------------------------------------------------------------

class TestCreateSlidingWindows:
    """Tests for create_sliding_windows."""

    def test_output_shapes(self) -> None:
        """X should be [num_windows, window_size, num_features], y [num_windows]."""
        df = _make_grouped_df(num_groups=3, rows_per_group=10)
        feature_cols = ["feat_a", "feat_b"]

        X, y = create_sliding_windows(
            df,
            feature_cols=feature_cols,
            label_col="label",
            group_col="group_id",
            window_size=4,
        )

        # Each group of 10 rows yields 10 - 4 + 1 = 7 windows
        expected_windows = 3 * 7
        assert X.shape == (expected_windows, 4, 2)
        assert y.shape == (expected_windows,)

    def test_label_is_last_row(self) -> None:
        """The label value must correspond to the last row of each window."""
        df = _make_grouped_df(num_groups=1, rows_per_group=5)
        feature_cols = ["feat_a"]

        X, y = create_sliding_windows(
            df,
            feature_cols=feature_cols,
            label_col="label",
            group_col="group_id",
            window_size=3,
        )

        # Group 1, label = time * 1
        # Windows: [0:3] -> label at row 3 -> 3.0, [1:4] -> 4.0, [2:5] -> 5.0
        expected_labels = np.array([3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(y, expected_labels)

    def test_window_content_matches_source(self) -> None:
        """Window rows should be consecutive slices of the source data."""
        df = _make_grouped_df(num_groups=1, rows_per_group=5)
        feature_cols = ["feat_a", "feat_b"]

        X, _ = create_sliding_windows(
            df,
            feature_cols=feature_cols,
            label_col="label",
            group_col="group_id",
            window_size=3,
        )

        # First window should be rows 0, 1, 2 of the group
        expected = df[feature_cols].iloc[:3].values.astype(np.float64)
        np.testing.assert_array_almost_equal(X[0], expected)

    def test_short_groups_skipped(self) -> None:
        """Groups with fewer rows than window_size must be skipped."""
        df = _make_grouped_df(num_groups=3, rows_per_group=10)
        # Add a short group
        short = pd.DataFrame({
            "group_id": [4, 4],
            "time": [1, 2],
            "label": [1.0, 2.0],
            "feat_a": [10.0, 20.0],
            "feat_b": [11.0, 21.0],
        })
        df = pd.concat([df, short], ignore_index=True)

        X, y = create_sliding_windows(
            df,
            feature_cols=["feat_a", "feat_b"],
            label_col="label",
            group_col="group_id",
            window_size=4,
        )

        # Only the 3 original groups contribute; short group is skipped
        assert X.shape[0] == 3 * 7

    def test_short_group_warning_logged(self, caplog) -> None:
        """A warning should be logged when groups are skipped."""
        df = pd.DataFrame({
            "group_id": [1, 1, 2],
            "time": [1, 2, 1],
            "label": [1.0, 2.0, 10.0],
            "feat_a": [1.0, 2.0, 10.0],
        })

        with caplog.at_level(logging.WARNING, logger="src.data.windowing"):
            create_sliding_windows(
                df,
                feature_cols=["feat_a"],
                label_col="label",
                group_col="group_id",
                window_size=5,
            )

        assert "Skipped" in caplog.text

    def test_all_groups_short_returns_empty(self) -> None:
        """When every group is too short, return empty arrays."""
        df = pd.DataFrame({
            "group_id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "label": [1.0, 2.0, 3.0, 4.0],
            "feat_a": [1.0, 2.0, 3.0, 4.0],
        })

        X, y = create_sliding_windows(
            df,
            feature_cols=["feat_a"],
            label_col="label",
            group_col="group_id",
            window_size=5,
        )

        assert X.shape == (0, 5, 1)
        assert y.shape == (0,)

    def test_invalid_window_size_raises(self) -> None:
        """window_size < 1 should raise ValueError."""
        df = _make_grouped_df()
        with pytest.raises(ValueError, match="window_size"):
            create_sliding_windows(df, ["feat_a"], "label", "group_id", 0)

    def test_missing_column_raises(self) -> None:
        """Referencing a non-existent column should raise ValueError."""
        df = _make_grouped_df()
        with pytest.raises(ValueError, match="Columns not found"):
            create_sliding_windows(df, ["nonexistent"], "label", "group_id", 3)


# ---------------------------------------------------------------------------
# normalize_features tests
# ---------------------------------------------------------------------------

class TestNormalizeFeatures:
    """Tests for normalize_features."""

    def test_train_scaled_to_unit_range(self) -> None:
        """Each feature in the training output should be in [0, 1]."""
        df = _make_grouped_df()
        feature_cols = ["feat_a", "feat_b"]

        train_scaled, _, _ = normalize_features(df, [], feature_cols)

        for col in feature_cols:
            assert train_scaled[col].min() == pytest.approx(0.0)
            assert train_scaled[col].max() == pytest.approx(1.0)

    def test_other_dfs_uses_train_minmax(self) -> None:
        """Other DataFrames must be scaled using training min/max, not their own."""
        train = pd.DataFrame({
            "feat_a": [10.0, 20.0, 30.0],
            "feat_b": [100.0, 200.0, 300.0],
        })
        test = pd.DataFrame({
            "feat_a": [50.0],  # outside train range
            "feat_b": [50.0],  # inside train range
        })

        _, test_scaled, params = normalize_features(train, [test], ["feat_a", "feat_b"])

        # feat_a: train min=10, max=30, range=20
        # test value 50 -> (50-10)/20 = 2.0 — outside [0,1], proving train params were used
        assert test_scaled[0].iloc[0]["feat_a"] == pytest.approx(2.0)

        # feat_b: train min=100, max=300, range=200
        # test value 50 -> (50-100)/200 = -0.25
        assert test_scaled[0].iloc[0]["feat_b"] == pytest.approx(-0.25)

    def test_params_dict_matches_train_stats(self) -> None:
        """The returned params dict should contain train min/max."""
        train = pd.DataFrame({
            "feat_a": [5.0, 15.0, 25.0],
            "feat_b": [100.0, 200.0, 300.0],
        })

        _, _, params = normalize_features(train, [], ["feat_a", "feat_b"])

        assert params["feat_a"]["min"] == 5.0
        assert params["feat_a"]["max"] == 25.0
        assert params["feat_b"]["min"] == 100.0
        assert params["feat_b"]["max"] == 300.0

    def test_constant_feature_defaults_to_zero(self) -> None:
        """A feature constant in train should produce 0.0 everywhere."""
        train = pd.DataFrame({"feat_a": [7.0, 7.0, 7.0], "feat_b": [1.0, 2.0, 3.0]})
        test = pd.DataFrame({"feat_a": [7.0, 7.0], "feat_b": [4.0, 5.0]})

        train_scaled, test_scaled, _ = normalize_features(
            train, [test], ["feat_a", "feat_b"]
        )

        assert (train_scaled["feat_a"] == 0.0).all()
        assert (test_scaled[0]["feat_a"] == 0.0).all()

    def test_does_not_mutate_input(self) -> None:
        """Original DataFrames must not be modified."""
        train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0]})
        other = pd.DataFrame({"feat_a": [4.0, 5.0]})

        train_original = train.copy()
        other_original = other.copy()

        normalize_features(train, [other], ["feat_a"])

        pd.testing.assert_frame_equal(train, train_original)
        pd.testing.assert_frame_equal(other, other_original)

    def test_multiple_other_dfs(self) -> None:
        """Should handle a list of multiple other DataFrames."""
        train = pd.DataFrame({"feat_a": [0.0, 10.0]})
        val = pd.DataFrame({"feat_a": [5.0]})
        test = pd.DataFrame({"feat_a": [15.0]})

        _, (val_scaled, test_scaled), params = normalize_features(
            train, [val, test], ["feat_a"]
        )

        assert val_scaled["feat_a"].iloc[0] == pytest.approx(0.5)
        # 15 is outside train range; still scaled with train min/max
        assert test_scaled["feat_a"].iloc[0] == pytest.approx(1.5)
