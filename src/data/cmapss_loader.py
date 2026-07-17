"""C-MAPSS data loader module.

Handles loading and cleaning raw NASA C-MAPSS dataset files.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# Default column names for C-MAPSS files
DEFAULT_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cmapss_config.yaml"


def _load_column_names() -> list[str]:
    """Load column names from config if available, otherwise use defaults.

    Returns:
        List of column name strings.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        # Config may define columns explicitly; fall back to defaults
        if "column_names" in cfg:
            return list(cfg["column_names"])
    return list(DEFAULT_COLUMNS)


def load_cmapss_raw(filepath: str) -> pd.DataFrame:
    """Read a C-MAPSS train or test file into a DataFrame.

    The source files are space-delimited with no header row and may contain
    trailing whitespace that produces empty trailing columns. This function
    assigns proper column names and drops any such empty columns.

    Args:
        filepath: Path to a C-MAPSS text file (e.g. train_FD001.txt).

    Returns:
        DataFrame with named columns and no trailing empty columns.
    """
    columns = _load_column_names()
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=columns,
    )
    # Drop trailing unnamed columns caused by trailing whitespace
    df = df.drop(columns=[c for c in df.columns if c is None or (isinstance(c, float) and pd.isna(c))])
    return df


def compute_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Remaining Useful Life labels for training data.

    For each unit_number, RUL at a given cycle equals the maximum cycle
    count for that unit minus the current cycle count.  RUL is *not* capped
    here — capping is a modeling decision left to the training script.

    Args:
        df: Training DataFrame with columns ``unit_number`` and
            ``time_in_cycles`` (plus any other columns).

    Returns:
        A copy of *df* with an added ``RUL`` column.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_in_cycles"]
    return df


def load_test_rul(filepath: str) -> pd.DataFrame:
    """Read the per-unit RUL file for the C-MAPSS test set.

    The file contains one integer RUL value per line, one line per test
    unit, in unit-number order.

    Args:
        filepath: Path to the RUL file (e.g. RUL_FD001.txt).

    Returns:
        DataFrame with columns ``unit_number`` (1-indexed) and ``RUL``.
    """
    rul_values = pd.read_csv(filepath, sep=r"\s+", header=None).squeeze("columns")
    df = pd.DataFrame({
        "unit_number": range(1, len(rul_values) + 1),
        "RUL": rul_values.values,
    })
    return df


def identify_constant_sensors(
    df: pd.DataFrame,
    threshold: float = 1e-5,
) -> list[str]:
    """Identify sensor columns with near-zero variance.

    Sensors whose standard deviation falls below *threshold* across the
    entire dataset carry no degradation signal and are commonly dropped in
    preprocessing.  This function only *identifies* them — the caller
    decides whether to drop.

    Args:
        df: DataFrame containing ``sensor_*`` columns.
        threshold: Standard-deviation threshold below which a sensor is
            considered constant.

    Returns:
        List of sensor column names that are (near-)constant.
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    stds = df[sensor_cols].std()
    return stds[stds < threshold].index.tolist()
