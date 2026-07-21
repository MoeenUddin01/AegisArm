"""Phase 3 CLI entrypoint — synthetic degradation data generation.

Runs the KUKA joint through hundreds of sine cycles with increasing
damping, logs the resulting telemetry, and produces the verification
checkpoint plot.

Usage::

    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --config configs/sim_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.synthetic_joint_generator import generate_degradation_run
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "sim_config.yaml"
_PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"


def _get_cycle_stats(df, cycles: list[int]) -> dict[int, dict[str, float]]:
    """Return summary stats for specific cycles.

    Args:
        df: DataFrame with at least ``cycle``, ``applied_torque``,
            ``position_error`` columns.
        cycles: Cycle numbers to extract stats for.

    Returns:
        Dict mapping cycle → {torque_rms, position_error_mean}.
    """
    stats: dict[int, dict[str, float]] = {}
    for c in cycles:
        subset = df[df["cycle"] == c]
        stats[c] = {
            "torque_rms": float(np.sqrt(np.mean(subset["applied_torque"] ** 2))),
            "position_error_mean": float(subset["position_error"].abs().mean()),
        }
    return stats


def _save_verification_plot(df, path: Path) -> None:
    """Save the Phase 3 verification checkpoint plot.

    Two stacked subplots: torque RMS vs cycle and
    position_error vs cycle.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute per-cycle RMS torque
    cycle_groups = df.groupby("cycle")["applied_torque"]
    torque_rms = cycle_groups.apply(lambda s: float(np.sqrt(np.mean(s ** 2))))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(torque_rms.index, torque_rms.values, linewidth=0.8)
    ax1.set_ylabel("Torque RMS (N·m)")
    ax1.set_title("Phase 3 — Raw degradation signal (verification checkpoint)")
    ax1.grid(True, alpha=0.3)

    # Mean absolute position error per cycle
    cycle_err = df.groupby("cycle")["position_error"].apply(lambda s: s.abs().mean())
    ax2.plot(cycle_err.index, cycle_err.values, linewidth=0.8, color="C1")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("|position error| mean (rad)")
    ax2.grid(True, alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description="Phase 3 synthetic degradation generator")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to sim_config.yaml",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting Phase 3 synthetic degradation run")

    df = generate_degradation_run(args.config)
    logger.info("Generated %d rows across %d cycles", len(df), df["cycle"].nunique())

    # --- verification checkpoint: summary stats at key cycles ---
    sample_cycles = [0, 150, df["cycle"].max()]
    stats = _get_cycle_stats(df, sample_cycles)

    logger.info("Phase 3 verification checkpoint — key cycle stats:")
    for c in sample_cycles:
        s = stats[c]
        logger.info(
            "  cycle %3d:  torque_rms=%10.4f N·m  |error|_mean=%10.4f rad",
            c,
            s["torque_rms"],
            s["position_error_mean"],
        )

    # --- sanity check: trend must be visible ---
    torque_first = stats[0]["torque_rms"]
    torque_last = stats[sample_cycles[-1]]["torque_rms"]
    error_first = stats[0]["position_error_mean"]
    error_last = stats[sample_cycles[-1]]["position_error_mean"]

    if torque_last <= torque_first:
        logger.error(
            "FAIL — torque RMS did NOT increase over time (%.4f → %.4f). "
            "Tuning degradation_power or max_damping is needed.",
            torque_first,
            torque_last,
        )
        sys.exit(1)

    # Position error may not grow much if the motor compensates — warn
    # but don't fail, since torque RMS is the primary degradation signal.
    if error_last <= error_first:
        logger.warning(
            "Position error did NOT increase (%.4f → %.4f) — motor may be "
            "compensating. Torque RMS is the primary signal; proceeding.",
            error_first,
            error_last,
        )

    logger.info(
        "Trend confirmed: torque RMS %.4f → %.4f, |error| %.4f → %.4f",
        torque_first,
        torque_last,
        error_first,
        error_last,
    )

    # --- save verification plot ---
    plot_path = _PLOTS_DIR / "phase3_raw_degradation_signal.png"
    _save_verification_plot(df, plot_path)
    logger.info("Verification plot saved to %s", plot_path)
    logger.info("Phase 3 verification checkpoint complete.")


if __name__ == "__main__":
    main()
