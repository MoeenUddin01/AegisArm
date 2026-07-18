"""Phase 2 CLI entrypoint — sine-wave joint demo.

Runs the KUKA arm with a sine-wave-driven joint, logs raw state, prints
summary stats, and saves a torque-vs-step plot.  This plot **is** the
Phase 2 verification checkpoint: values must be stable (no NaN/inf, no
unexplained drift) across all steps with zero degradation applied.

Usage::

    # Interactive window (default)
    python scripts/run_phase2_demo.py

    # Headless, quick test
    python scripts/run_phase2_demo.py --no-gui --steps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulation.pybullet_env import run_demo
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "sim_config.yaml"
_PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"


def _print_summary(df, label: str) -> None:
    """Log mean / std / min / max for a column."""
    vals = df[label].values
    logger.info(
        "  %18s  mean=%10.4f  std=%10.4f  min=%10.4f  max=%10.4f",
        label,
        np.mean(vals),
        np.std(vals),
        np.min(vals),
        np.max(vals),
    )


def _save_torque_plot(df, path: Path) -> None:
    """Save a plot of applied_torque vs step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["step"], df["applied_torque"], linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Applied torque (N·m)")
    ax.set_title("Phase 2 — Joint torque over time (no degradation)")
    ax.grid(True, alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description="Phase 2 sine-wave demo")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to sim_config.yaml",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Open PyBullet GUI window (default: True)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless (p.DIRECT)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override num_steps from config",
    )
    args = parser.parse_args()

    setup_logging()

    # Handle --no-gui
    gui = not args.no_gui

    # Override num_steps if provided
    if args.steps is not None:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["num_steps"] = args.steps
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(_PROJECT_ROOT / "configs")
        )
        yaml.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name
    else:
        config_path = args.config

    logger.info("Running Phase 2 demo (gui=%s)", gui)
    df = run_demo(config_path=config_path, gui=gui)

    # --- summary stats (Phase 2 verification checkpoint) -----------------
    logger.info("Phase 2 summary stats:")
    _print_summary(df, "angle")
    _print_summary(df, "velocity")
    _print_summary(df, "applied_torque")

    # Sanity checks
    if df.isna().any().any():
        logger.error("FAIL — NaN values detected in joint readout!")
        sys.exit(1)
    if np.isinf(df[["angle", "velocity", "applied_torque"]].values).any():
        logger.error("FAIL — Inf values detected in joint readout!")
        sys.exit(1)

    # Save plot
    plot_path = _PLOTS_DIR / "phase2_joint_readout.png"
    _save_torque_plot(df, plot_path)
    logger.info("Torque plot saved to %s", plot_path)
    logger.info("Phase 2 verification checkpoint complete.")


if __name__ == "__main__":
    main()
