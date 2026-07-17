"""Evaluation metrics for RUL prediction.

Provides RMSE, the NASA PHM08 asymmetric scoring function, and a
predicted-vs-actual scatter plot used as the Phase 1 verification
checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Scalar RMSE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM08 asymmetric scoring function.

    This is the scoring function used in the C-MAPSS / PHM08 data
    challenge.  It penalises *late* predictions (model predicted the
    engine had more life than it actually did) more heavily than *early*
    predictions (model was pessimistic).

    **Why the asymmetry matters:**  In a real maintenance setting a late
    prediction is far more dangerous — the engine fails before the
    scheduled intervention, potentially causing catastrophic damage or
    loss of life.  An early prediction wastes some useful life but is
    still safe.  The exponential penalty curves encode this: a late
    error of *d* cycles incurs ``exp(d / 10) − 1`` while an early
    error of the same magnitude incurs only ``exp(d / 13) − 1``.

    Per-sample score::

        s_i = exp(-d_i / 13) − 1   if d_i < 0   (early)
              exp( d_i / 10) − 1   if d_i >= 0   (late)

    where ``d_i = predicted_i − actual_i``.

    The total score is the **sum** across all samples (lower is better).

    Reference:
        Saxena, A. et al., "Damage Propagation Modeling for Aircraft
        Engine Run-to-Failure Simulation", PHM08.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Summed asymmetric score (scalar, >= 0).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true  # positive = late, negative = early

    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,   # early: gentler penalty
        np.exp(d / 10.0) - 1.0,     # late:  harsher penalty
    )
    return float(np.sum(score))


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
) -> None:
    """Scatter plot of predicted vs actual RUL with a perfect-prediction line.

    Saves the figure to *save_path* (parent directories are created
    automatically).

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.
        save_path: Destination file path (e.g.
            ``outputs/plots/phase1_pred_vs_actual.png``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Phase 1 — Predicted vs Actual RUL")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
