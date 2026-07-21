"""Phase 3 CLI — multi-run degradation data generation.

.. warning::

    This script runs 25 independent degradation simulations (configurable).
    Each takes ~15 seconds headless, so the full run takes roughly 6–7
    minutes.  It is headless by design and must **not** be run with
    ``gui=True``.

Usage::

    python scripts/generate_multirun_data.py
    python scripts/generate_multirun_data.py --config configs/sim_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.synthetic_joint_generator import generate_multi_run_dataset
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "sim_config.yaml"
_PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"


def _save_overlay_plot(df, path: Path) -> None:
    """Save torque_rms vs cycle, one line per run overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    for run_id, group in df.groupby("run_id"):
        ax.plot(
            group["cycle"], group["torque_rms"],
            linewidth=0.6, alpha=0.5, label=f"run {run_id}",
        )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Torque RMS (N·m)")
    ax.set_title("Phase 3 — Multi-run overlay (torque RMS vs cycle)")
    ax.grid(True, alpha=0.3)

    # Compact legend: only show a few runs to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        step = max(1, len(handles) // 8)
        ax.legend(
            handles[::step], labels[::step],
            fontsize="small", loc="upper left",
        )
    else:
        ax.legend(fontsize="small", loc="upper left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Phase 3 multi-run degradation generator",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to sim_config.yaml",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting Phase 3 multi-run degradation generation")

    df = generate_multi_run_dataset(args.config)
    logger.info(
        "Generated %d rows across %d runs, %d cycles each",
        len(df),
        df["run_id"].nunique(),
        df["cycle"].nunique(),
    )

    # --- verification: check run-to-run variation ---
    final_cycle = df["cycle"].max()
    final_torques = df[df["cycle"] == final_cycle]["torque_rms"]
    logger.info(
        "Final-cycle torque RMS: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        final_torques.mean(),
        final_torques.std(),
        final_torques.min(),
        final_torques.max(),
    )

    if final_torques.std() < 0.01:
        logger.error(
            "FAIL — run-to-run variation at final cycle is negligible "
            "(std=%.4f). Check damping_variation_pct and power_variation.",
            final_torques.std(),
        )
        sys.exit(1)

    logger.info(
        "Run-to-run variation confirmed (std=%.4f at final cycle)",
        final_torques.std(),
    )

    # --- save overlay plot ---
    plot_path = _PLOTS_DIR / "phase3_multirun_overlay.png"
    _save_overlay_plot(df, plot_path)
    logger.info("Overlay plot saved to %s", plot_path)
    logger.info("Phase 3 multi-run generation complete.")


if __name__ == "__main__":
    main()
