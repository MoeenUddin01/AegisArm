"""Phase 1 evaluation script — C-MAPSS LSTM RUL regressor.

Loads the test set, applies the *saved* feature scaler (no refit),
builds one window per test unit from its last *window_size* cycles,
runs inference with the trained model, and reports RMSE + NASA score
alongside a predicted-vs-actual scatter plot.

**This is the Phase 1 verification checkpoint from CLAUDE.md.**
After running this script, the scatter plot must show predictions
clustering near the diagonal.  If the checkpoint fails, the fix
belongs in ``train_cmapss.py`` or ``windowing.py`` — do **not** modify
Phase 3/4 files based on this result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Resolve project root so imports work from any cwd.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.cmapss_loader import load_cmapss_raw, load_test_rul
from src.evaluation.metrics import nasa_score, plot_predicted_vs_actual, rmse
from src.models.lstm_rul import LSTMRegressor, load_config
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
_MODELS_DIR = _PROJECT_ROOT / "models"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_PLOTS_DIR = _OUTPUTS_DIR / "plots"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sensor + op_setting columns."""
    return [
        c for c in df.columns
        if c.startswith("sensor_") or c.startswith("op_setting_")
    ]


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


def _last_window_per_unit(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> np.ndarray:
    """Extract the last *window_size* cycles for each unit as a 3-D array.

    Returns:
        Array of shape ``[num_units, window_size, num_features]``.
    """
    windows: list[np.ndarray] = []
    for _, grp in df.groupby("unit_number", sort=True):
        tail = grp.tail(window_size)
        if len(tail) < window_size:
            logger.warning(
                "Unit %s has only %d rows (< window_size %d) — skipping",
                tail["unit_number"].iloc[0],
                len(tail),
                window_size,
            )
            continue
        windows.append(tail[feature_cols].values.astype(np.float32))
    return np.stack(windows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Path | str | None = None) -> None:
    """Run the Phase 1 evaluation pipeline.

    Args:
        config_path: Optional override for the YAML config file.
    """
    setup_logging()

    # --- config -----------------------------------------------------------
    cfg_path = Path(config_path) if config_path else (
        _PROJECT_ROOT / "configs" / "cmapss_config.yaml"
    )
    cfg = load_config(cfg_path)
    window_size = cfg["window_size"]
    rul_cap = cfg.get("rul_cap")
    logger.info("Config loaded from %s", cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- load test data ---------------------------------------------------
    test_path = _DATA_DIR / "test_FD001.txt"
    rul_path = _DATA_DIR / "RUL_FD001.txt"
    logger.info("Loading test data from %s", test_path)

    test_df = load_cmapss_raw(str(test_path))
    true_rul_df = load_test_rul(str(rul_path))

    feature_cols = _feature_columns(test_df)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # --- apply saved scaler (no refit) ------------------------------------
    scaler_path = _MODELS_DIR / "cmapss_feature_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Run train_cmapss.py first to generate it."
        )
    scaler = _load_scaler(scaler_path)
    logger.info("Scaler loaded from %s", scaler_path)
    test_df = _apply_scaler(test_df, feature_cols, scaler)

    # --- build one window per test unit -----------------------------------
    X_test = _last_window_per_unit(test_df, feature_cols, window_size)
    logger.info("Test windows shape: %s", X_test.shape)

    # --- load model & run inference ---------------------------------------
    weights_path = _MODELS_DIR / "lstm_pdm.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Run train_cmapss.py first."
        )
    model = LSTMRegressor.from_config(cfg_path).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    logger.info("Model loaded from %s", weights_path)

    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(x_tensor).cpu().numpy()

    # --- align true RUL to the same unit order as X_test -----------------
    # _last_window_per_unit groups with sort=True, so units are 1..N.
    y_true = true_rul_df.sort_values("unit_number")["RUL"].values.astype(np.float64)

    # Optionally cap the true RUL to match training convention
    if rul_cap is not None:
        y_true = np.clip(y_true, a_max=rul_cap, a_min=None)

    # --- compute metrics --------------------------------------------------
    rmse_val = rmse(y_true, y_pred)
    nasa_val = nasa_score(y_true, y_pred)

    logger.info("RMSE:       %.4f", rmse_val)
    logger.info("NASA score: %.4f", nasa_val)

    # --- scatter plot (Phase 1 verification checkpoint) -------------------
    plot_path = _PLOTS_DIR / "phase1_pred_vs_actual.png"
    plot_predicted_vs_actual(y_true, y_pred, plot_path)
    logger.info("Scatter plot saved to %s", plot_path)
    logger.info("Phase 1 verification checkpoint complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate C-MAPSS LSTM RUL model on test_FD001"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/cmapss_config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
