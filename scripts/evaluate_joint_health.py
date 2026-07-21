"""Phase 4 evaluation script — Joint-health LSTM.

Loads the validation run_ids, applies the *saved* feature scaler (no
refit), runs inference with the trained model, and reports RMSE alongside
a predicted-vs-actual scatter plot.

**Alignment pattern:** predictions are joined to ground truth by ``run_id``
+ ``cycle``, never by positional slicing — the same class of bug fixed in
Phase 1's ``evaluate_cmapss.py``.

Usage::

    python scripts/evaluate_joint_health.py
    python scripts/evaluate_joint_health.py --config configs/joint_health_config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import plot_predicted_vs_actual, rmse
from src.models.lstm_rul import LSTMRegressor
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

_DATA_DIR = _PROJECT_ROOT / "data" / "synthetic"
_MODELS_DIR = _PROJECT_ROOT / "models"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_PLOTS_DIR = _OUTPUTS_DIR / "plots"

FEATURE_COLS: list[str] = [
    "torque_rms",
    "velocity_rms",
    "position_error_mean",
]

LABEL_COL = "cycles_to_failure"
GROUP_COL = "run_id"


def _load_config(config_path: str | Path) -> dict:
    """Load the joint-health YAML config.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_scaler(path: Path) -> dict[str, dict[str, float]]:
    """Load the saved min-max scaler JSON."""
    with open(path) as f:
        return json.load(f)


def _apply_scaler(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply a pre-fitted min-max scaler to the given columns."""
    df = df.copy()
    for col in feature_cols:
        p = scaler[col]
        rng = p["max"] - p["min"]
        if rng == 0.0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - p["min"]) / rng
    return df


def _last_window_per_run(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, list[int]]:
    """Extract the last *window_size* cycles for each run as a 3-D array.

    Runs with fewer than *window_size* rows are skipped.

    Returns:
        ``(X, run_ids)`` where *X* has shape
        ``[num_surviving_runs, window_size, num_features]`` and
        *run_ids* is the list of ``run_id`` values that survived,
        in the same order as the rows of *X*.
    """
    windows: list[np.ndarray] = []
    surviving_runs: list[int] = []
    for _, grp in df.groupby(GROUP_COL, sort=True):
        run_id = int(grp[GROUP_COL].iloc[0])
        tail = grp.tail(window_size)
        if len(tail) < window_size:
            logger.warning(
                "Run %d has only %d rows (< window_size %d) — skipping",
                run_id, len(tail), window_size,
            )
            continue
        windows.append(tail[feature_cols].values.astype(np.float32))
        surviving_runs.append(run_id)
    return np.stack(windows), surviving_runs


def main(config_path: Path | str | None = None) -> None:
    """Run the Phase 4 evaluation pipeline.

    Args:
        config_path: Optional override for the YAML config file.
    """
    setup_logging()

    cfg_path = Path(config_path) if config_path else (
        _PROJECT_ROOT / "configs" / "joint_health_config.yaml"
    )
    cfg = _load_config(cfg_path)
    window_size = cfg["window_size"]
    val_split_runs = cfg["val_split_runs"]
    seed = cfg.get("seed", 42)
    logger.info("Config loaded from %s", cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- load data --------------------------------------------------------
    csv_path = _DATA_DIR / "joint_degradation_multirun.csv"
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Reproduce the same train/val split used during training
    run_ids = df[GROUP_COL].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(run_ids)
    val_runs = set(run_ids[:val_split_runs])
    val_df = df[df[GROUP_COL].isin(val_runs)].reset_index(drop=True)
    logger.info("Evaluation: %d val runs", len(val_runs))

    # --- apply saved scaler (no refit) ------------------------------------
    scaler_path = _MODELS_DIR / "joint_health_feature_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Run train_joint_health.py first."
        )
    scaler = _load_scaler(scaler_path)
    logger.info("Scaler loaded from %s", scaler_path)

    feature_cols = list(scaler.keys())
    input_size = len(feature_cols)
    logger.info("Using %d features from scaler: %s", input_size, feature_cols)

    val_df = _apply_scaler(val_df, feature_cols, scaler)

    # --- build one window per val run -------------------------------------
    X_val, surviving_runs = _last_window_per_run(val_df, feature_cols, window_size)
    logger.info("Val windows shape: %s", X_val.shape)
    logger.info("Surviving runs (%d): %s", len(surviving_runs), surviving_runs)

    # --- load model & run inference ---------------------------------------
    weights_path = _MODELS_DIR / "lstm_joint_health.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Run train_joint_health.py first."
        )
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    logger.info("Model loaded from %s", weights_path)

    with torch.no_grad():
        x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_pred = model(x_tensor).cpu().numpy()

    # --- align true labels to surviving runs only -------------------------
    # Join by run_id, NOT positional slicing — skipped runs can appear
    # anywhere in the run range.
    surviving_set = set(surviving_runs)
    last_cycles = (
        val_df[val_df[GROUP_COL].isin(surviving_set)]
        .groupby(GROUP_COL)
        .last()
        .reset_index()
    )
    # Reorder to match surviving_runs order (which matches y_pred order)
    last_cycles["order"] = last_cycles[GROUP_COL].map(
        {r: i for i, r in enumerate(surviving_runs)}
    )
    last_cycles = last_cycles.sort_values("order").reset_index(drop=True)
    y_true = last_cycles[LABEL_COL].values.astype(np.float64)

    # --- sanity check before metrics --------------------------------------
    assert len(y_true) == len(y_pred), (
        f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch."
    )

    # --- compute metrics --------------------------------------------------
    rmse_val = rmse(y_true, y_pred)
    logger.info("RMSE: %.4f cycles", rmse_val)

    # --- scatter plot (Phase 4 verification checkpoint) -------------------
    plot_path = _PLOTS_DIR / "phase4_pred_vs_actual.png"
    plot_predicted_vs_actual(y_true, y_pred, plot_path)
    logger.info("Scatter plot saved to %s", plot_path)
    logger.info("Phase 4 evaluation complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate joint-health LSTM on validation runs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/joint_health_config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
