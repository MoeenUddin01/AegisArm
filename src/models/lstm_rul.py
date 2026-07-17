"""Phase 1 LSTM regressor for C-MAPSS Remaining Useful Life prediction.

.. warning::

    This model is trained exclusively on C-MAPSS data (21 aircraft-engine
    sensor channels).  It is **not** compatible with the PyBullet joint-
    simulation data produced in Phases 3–4, which has 3–4 features with
    entirely different physical meaning.  See CLAUDE.md "Critical
    constraint" section for details.  Never load these weights into
    ``lstm_joint_health.py``.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from torch import Tensor
import torch.nn as nn

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cmapss_config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load the C-MAPSS YAML config.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


class LSTMRegressor(nn.Module):
    """LSTM-based RUL regressor for C-MAPSS data.

    Architecture: stacked LSTM layers → linear projection of the final
    hidden state to a single scalar (predicted RUL).

    **This model must only be trained and evaluated on C-MAPSS features.**
    The 21-channel input dimension is incompatible with the 3–4 channel
    joint-simulation feature space.  Loading Phase 1 weights into the
    Phase 4 ``LSTMJointHealth`` model is a bug — see CLAUDE.md.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialise the model.

        Args:
            input_size: Number of input features per timestep (21 for
                C-MAPSS).
            hidden_size: Number of features in the LSTM hidden state.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability applied between LSTM layers
                (ignored when ``num_layers == 1``).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``[batch, window_size, input_size]``.

        Returns:
            Predicted RUL tensor of shape ``[batch]``.
        """
        # lstm_out: [batch, window_size, hidden_size]
        lstm_out, _ = self.lstm(x)
        # Take the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]
        out = self.fc(last_hidden)  # [batch, 1]
        return out.squeeze(-1)  # [batch]

    @classmethod
    def from_config(
        cls,
        path: Path = _CONFIG_PATH,
        input_size: int | None = None,
    ) -> "LSTMRegressor":
        """Construct an ``LSTMRegressor`` from ``configs/cmapss_config.yaml``.

        ``input_size`` is **not** stored in the YAML — it is a derived
        value that depends on which sensors survived constant-sensor
        filtering.  The caller must compute it (typically
        ``len(feature_cols)``) and pass it here.

        Args:
            path: Path to the YAML configuration file.
            input_size: Number of input features per timestep.  **Required**
                — ``input_size`` is not read from the config because it
                varies by dataset after zero-variance sensor removal.

        Returns:
            A new ``LSTMRegressor`` instance with hyperparameters read
            from the config.

        Raises:
            ValueError: If *input_size* is not provided.
        """
        if input_size is None:
            raise ValueError(
                "input_size must be provided explicitly — it is derived "
                "from feature_cols after constant-sensor removal and is "
                "not stored in the YAML config."
            )
        cfg = load_config(path)
        return cls(
            input_size=input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
