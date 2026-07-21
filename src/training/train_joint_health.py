"""Phase 4 training script — Joint-health LSTM.

Loads the multi-run aggregated dataset, preprocesses it (normalisation,
windowing by run_id), trains an ``LSTMRegressor`` on cycles-to-failure,
and saves weights + loss history.

**Input:** ``data/synthetic/joint_degradation_multirun.csv``
**Output:** ``models/lstm_joint_health.pth``, scaler JSON, loss history JSON.

Runnable as a standalone script or importable as a library function.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    TensorDataset = None  # type: ignore[assignment,misc]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.cmapss_loader import identify_constant_sensors
from src.data.windowing import create_sliding_windows, normalize_features
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)

_DATA_DIR = _PROJECT_ROOT / "data" / "synthetic"
_MODELS_DIR = _PROJECT_ROOT / "models"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_PLOTS_DIR = _OUTPUTS_DIR / "plots"

# ---------------------------------------------------------------------------
# Hardcoded allowlist — do NOT derive by excluding a blocklist.
# Adding a new column to the CSV will not silently leak it into the model.
# ---------------------------------------------------------------------------
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


def _load_data() -> pd.DataFrame:
    """Load the multi-run aggregated dataset.

    Returns:
        DataFrame with one row per cycle per run.
    """
    csv_path = _DATA_DIR / "joint_degradation_multirun.csv"
    logger.info("Loading multi-run data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d runs, %d cycles per run",
                len(df), df[GROUP_COL].nunique(), df["cycle"].nunique())
    return df


def _warn_near_zero_variance(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Log a warning if any feature column has near-zero variance."""
    const = identify_constant_sensors(df, threshold=1e-5, columns=feature_cols)
    if const:
        logger.warning(
            "Near-zero-variance feature(s) detected (not dropping): %s", const,
        )
    else:
        logger.info("All %d features have non-zero variance.", len(feature_cols))


def _split_by_run_id(
    df: pd.DataFrame,
    val_split_runs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/validation by run_id — no cycle leakage.

    Args:
        df: Full multi-run DataFrame.
        val_split_runs: Number of run_ids to hold out for validation.
        seed: Random seed for shuffling.

    Returns:
        ``(train_df, val_df)`` with disjoint run_id sets.
    """
    run_ids = df[GROUP_COL].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(run_ids)

    val_runs = set(run_ids[:val_split_runs])
    train_runs = set(run_ids[val_split_runs:])

    train_df = df[df[GROUP_COL].isin(train_runs)].reset_index(drop=True)
    val_df = df[df[GROUP_COL].isin(val_runs)].reset_index(drop=True)

    logger.info(
        "Split: %d train runs, %d val runs",
        len(train_runs),
        len(val_runs),
    )
    return train_df, val_df


def _train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
) -> float:
    """Train for a single epoch; return average loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        n += len(y_batch)
    return total_loss / n


def _validate(
    model,
    loader,
    criterion,
    device,
) -> float:
    """Evaluate on validation set; return average loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
            n += len(y_batch)
    return total_loss / n


def _save_checkpoint(model, path: Path) -> None:
    """Save model state_dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _save_loss_history(history: dict[str, list[float]], path: Path) -> None:
    """Persist per-epoch losses as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Loss history saved to %s", path)


def main(config_path: Path | str | None = None) -> None:
    """Run the full Phase 4 training pipeline.

    Args:
        config_path: Optional override for the YAML config file.
    """
    setup_logging()

    cfg_path = Path(config_path) if config_path else (
        _PROJECT_ROOT / "configs" / "joint_health_config.yaml"
    )
    cfg = _load_config(cfg_path)
    logger.info("Config loaded from %s", cfg_path)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- data pipeline ----------------------------------------------------
    df = _load_data()

    # Sanity check: feature columns are the hardcoded allowlist
    assert FEATURE_COLS == [
        "torque_rms", "velocity_rms", "position_error_mean",
    ], "FEATURE_COLS must be the explicit allowlist — do not derive from a blocklist"
    logger.info("Feature columns (%d): %s", len(FEATURE_COLS), FEATURE_COLS)

    _warn_near_zero_variance(df, FEATURE_COLS)

    train_df, val_df = _split_by_run_id(
        df, cfg["val_split_runs"], seed,
    )

    # Normalisation — fit on train only, apply to val
    train_df, (val_df,), scaler_params = normalize_features(
        train_df, [val_df], FEATURE_COLS,
    )
    scaler_path = _MODELS_DIR / "joint_health_feature_scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    logger.info("Scaler saved to %s", scaler_path)

    # Windowing — group by run_id, window_size is in cycles
    window_size = cfg["window_size"]
    X_train, y_train = create_sliding_windows(
        train_df,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        group_col=GROUP_COL,
        window_size=window_size,
    )
    X_val, y_val = create_sliding_windows(
        val_df,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        group_col=GROUP_COL,
        window_size=window_size,
    )
    logger.info("Windows — train: X=%s y=%s, val: X=%s y=%s",
                X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # PyTorch datasets / loaders
    batch_size = cfg["batch_size"]
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # --- model / optimiser ------------------------------------------------
    from src.models.lstm_rul import LSTMRegressor

    input_size = len(FEATURE_COLS)
    logger.info("Building model with input_size=%d", input_size)

    model_cfg_path = str(cfg_path)
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    logger.info(
        "Model params: %d",
        sum(p.numel() for p in model.parameters()),
    )

    # --- training loop ----------------------------------------------------
    num_epochs = cfg["num_epochs"]
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch %3d/%d — train_loss: %.6f  val_loss: %.6f",
            epoch, num_epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, _MODELS_DIR / "lstm_joint_health.pth")

    # --- persist artefacts ------------------------------------------------
    loss_path = _PLOTS_DIR / "joint_health_loss_history.json"
    _save_loss_history(history, loss_path)

    logger.info("Training complete. Best val_loss: %.6f", best_val_loss)
    logger.info("Model weights: %s", _MODELS_DIR / "lstm_joint_health.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train joint-health LSTM")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/joint_health_config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
