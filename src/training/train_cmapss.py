"""Phase 1 training script — C-MAPSS LSTM RUL regressor.

Loads train_FD001.txt, preprocesses it (RUL labels, capping, normalisation,
windowing), trains an ``LSTMRegressor``, and saves weights + loss history.

Runnable as a standalone script **or** copy-pasted cell-by-cell into a
Kaggle notebook — all paths are relative and configurable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Resolve project root so imports work when run as a script from any cwd.
# When running in Kaggle notebooks that import from src/, the sys.path
# insertion is harmless.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.cmapss_loader import (
    compute_rul_labels,
    identify_constant_sensors,
    load_cmapss_raw,
)
from src.data.windowing import create_sliding_windows, normalize_features
from src.models.lstm_rul import LSTMRegressor, load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root — no absolute hardcoded paths)
# ---------------------------------------------------------------------------
_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
_MODELS_DIR = _PROJECT_ROOT / "models"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_PLOTS_DIR = _OUTPUTS_DIR / "plots"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sensor + op_setting columns (everything except meta + label)."""
    return [
        c for c in df.columns
        if c.startswith("sensor_") or c.startswith("op_setting_")
    ]


def _load_and_prepare(cfg: dict) -> pd.DataFrame:
    """Load train_FD001.txt, compute RUL labels, and apply rul_cap."""
    train_path = _DATA_DIR / "train_FD001.txt"
    logger.info("Loading training data from %s", train_path)
    df = load_cmapss_raw(str(train_path))

    df = compute_rul_labels(df)

    rul_cap = cfg.get("rul_cap")
    if rul_cap is not None:
        before = int(df["RUL"].max())
        df["RUL"] = df["RUL"].clip(upper=rul_cap)
        logger.info("RUL capped at %d (was %d)", rul_cap, before)

    return df


def _split_by_unit(
    df: pd.DataFrame,
    val_fraction: float,
    rng: np.random.RandomState,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/validation by unit_number (no row leakage)."""
    units = df["unit_number"].unique()
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])

    train_df = df[df["unit_number"].isin(train_units)].reset_index(drop=True)
    val_df = df[df["unit_number"].isin(val_units)].reset_index(drop=True)

    logger.info(
        "Split: %d train units, %d val units",
        len(train_units),
        len(val_units),
    )
    return train_df, val_df


def _build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows and return numpy arrays."""
    X, y = create_sliding_windows(
        df,
        feature_cols=feature_cols,
        label_col="RUL",
        group_col="unit_number",
        window_size=window_size,
    )
    logger.info("Windows — X: %s, y: %s", X.shape, y.shape)
    return X, y


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate on validation set; return average loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        total_loss += loss.item() * len(y_batch)
        n += len(y_batch)
    return total_loss / n


def _save_loss_history(history: dict[str, list[float]], path: Path) -> None:
    """Persist per-epoch losses as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Loss history saved to %s", path)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main(config_path: Path | str | None = None) -> None:
    """Run the full Phase 1 training pipeline.

    Args:
        config_path: Optional override for the YAML config file.
    """
    setup_logging()

    # --- config -----------------------------------------------------------
    cfg_path = Path(config_path) if config_path else (
        _PROJECT_ROOT / "configs" / "cmapss_config.yaml"
    )
    cfg = load_config(cfg_path)
    logger.info("Config loaded from %s", cfg_path)

    # --- reproducibility --------------------------------------------------
    seed = cfg.get("seed", 42)
    set_seed(seed)
    rng = np.random.RandomState(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- data pipeline ----------------------------------------------------
    df = _load_and_prepare(cfg)
    feature_cols = _feature_columns(df)

    # Drop zero-variance sensors — they carry no degradation signal and
    # would cause divide-by-zero in min-max normalisation.
    const_sensors = identify_constant_sensors(df)
    if const_sensors:
        feature_cols = [c for c in feature_cols if c not in const_sensors]
        logger.info(
            "Dropped %d constant sensor(s) (zero-variance, would cause "
            "divide-by-zero in min-max normalisation): %s",
            len(const_sensors),
            const_sensors,
        )

    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    train_df, val_df = _split_by_unit(df, cfg.get("val_split", 0.15), rng)

    # Normalisation — fit on train only, apply to val
    train_df, (val_df,), scaler_params = normalize_features(
        train_df, [val_df], feature_cols,
    )
    scaler_path = _MODELS_DIR / "cmapss_feature_scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    logger.info("Scaler saved to %s", scaler_path)

    # Windowing
    window_size = cfg["window_size"]
    X_train, y_train = _build_windows(train_df, feature_cols, window_size)
    X_val, y_val = _build_windows(val_df, feature_cols, window_size)

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
    # input_size is derived from feature_cols, NOT read from YAML — it
    # depends on which sensors survived constant-sensor filtering.
    input_size = len(feature_cols)
    logger.info("Building model with input_size=%d (derived from feature_cols)", input_size)
    model = LSTMRegressor.from_config(cfg_path, input_size=input_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    logger.info(
        "Model — params: %d",
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
            epoch,
            num_epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, _MODELS_DIR / "lstm_pdm.pth")

    # --- persist artefacts ------------------------------------------------
    loss_path = _PLOTS_DIR / "train_val_loss.json"
    _save_loss_history(history, loss_path)

    logger.info("Training complete. Best val_loss: %.6f", best_val_loss)
    logger.info("Model weights: %s", _MODELS_DIR / "lstm_pdm.pth")


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save model state_dict, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train C-MAPSS LSTM RUL model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/cmapss_config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
