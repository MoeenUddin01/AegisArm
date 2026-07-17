"""Sliding-window and normalization utilities for time-series features.

This module is shared between Phase 1 (C-MAPSS, 21+ features) and
Phase 3/4 (synthetic joint data, 3-4 features).  It contains **no**
column-count or column-name assumptions — callers specify which columns
are features, which is the label, and how groups are identified.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    group_col: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows from time-ordered grouped rows.

    For each group (identified by *group_col*), the function slides a
    window of length *window_size* over the rows and produces one sample
    per window.  The label for each window is the value of *label_col*
    at the **last** row of that window (i.e. predicting the label at the
    most recent timestep).

    Groups with fewer rows than *window_size* are silently skipped and a
    warning is logged with the count.

    Args:
        df: Input DataFrame.  Rows within each group must already be
            sorted in chronological (or logical) order.
        feature_cols: Column names to use as features.  These become the
            last axis of *X*.
        label_col: Column whose value at the final timestep of each
            window becomes the regression target.
        group_col: Column that identifies each independent time series
            (e.g. ``unit_number``).
        window_size: Number of consecutive rows per window.

    Returns:
        ``(X, y)`` where
        - *X* has shape ``[num_windows, window_size, len(feature_cols)]``
        - *y* has shape ``[num_windows]``

    Raises:
        ValueError: If *window_size* < 1 or *feature_cols* / *label_col*
            / *group_col* reference columns absent from *df*.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    missing = set(feature_cols + [label_col, group_col]) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in df: {sorted(missing)}")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    skipped = 0

    for _, group_df in df.groupby(group_col, sort=False):
        if len(group_df) < window_size:
            skipped += 1
            continue

        values = group_df[feature_cols].values.astype(np.float64)
        labels = group_df[label_col].values.astype(np.float64)

        num_windows = len(group_df) - window_size + 1
        for i in range(num_windows):
            all_X.append(values[i : i + window_size])
            all_y.append(labels[i + window_size - 1])

    if skipped:
        logger.warning(
            "Skipped %d group(s) shorter than window_size=%d",
            skipped,
            window_size,
        )

    if not all_X:
        return (
            np.empty((0, window_size, len(feature_cols)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    X = np.stack(all_X)
    y = np.stack(all_y)
    return X, y


def normalize_features(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, dict[str, float]]]:
    """Fit min-max normalization on *train_df* and apply to all datasets.

    **Why fit only on train?**
    Normalization parameters must come exclusively from the training set.
    If validation or test data leaked into the min/max computation, the
    model could receive artificially optimistic feature scales during
    evaluation, inflating reported performance.  Fitting on train alone
    guarantees the test set is truly "unseen" in every respect —
    including feature range.

    Each feature is scaled to [0, 1] via: ``(x - min) / (max - min)``.
    If a feature is constant in *train_df* (max == min), the scaled
    value is set to 0.0 for that feature across all datasets to avoid
    division by zero.

    Args:
        train_df: Training set — min/max are derived from this frame
            only.
        other_dfs: One or more additional DataFrames (validation, test,
            etc.) to which the training-set min/max will be applied.
        feature_cols: Column names to normalize.

    Returns:
        ``(train_scaled, [other_scaled_1, ...], params)`` where:
        - Each DataFrame has the same columns as its input with feature
          columns overwritten in-place.
        - *params* is a dict mapping each feature name to its fitted
          ``{"min": ..., "max": ...}``.
    """
    params: dict[str, dict[str, float]] = {}

    train_scaled = train_df.copy()
    other_scaled = [df.copy() for df in other_dfs]

    for col in feature_cols:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())
        params[col] = {"min": col_min, "max": col_max}

        range_val = col_max - col_min
        if range_val == 0.0:
            train_scaled[col] = 0.0
            for df in other_scaled:
                df[col] = 0.0
        else:
            train_scaled[col] = (train_df[col] - col_min) / range_val
            for df in other_scaled:
                df[col] = (df[col] - col_min) / range_val

    return train_scaled, other_scaled, params
