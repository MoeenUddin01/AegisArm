"""Phase 4 joint-health LSTM — predicts cycles-to-failure from joint telemetry.

This model is trained **exclusively** on aggregated features from
``data/synthetic/joint_degradation_multirun.csv``:

- ``torque_rms``
- ``velocity_rms``
- ``position_error_mean``

It must **never** receive ``damping`` or ``cycle`` as inputs — both leak
the ``cycles_to_failure`` label.  ``damping`` is the hidden ground-truth
variable that generates the degradation curve; ``cycle`` is a direct
proxy for ``cycles_to_failure = total_cycles - cycle``.

The model class is ``LSTMRegressor`` imported from
``src.models.lstm_rul`` — this is the same architecture used in Phase 1,
instantiated with a different ``input_size`` (3 instead of ~15).  We
reuse, not duplicate.
"""

from __future__ import annotations

from src.models.lstm_rul import LSTMRegressor

__all__ = ["LSTMRegressor"]
