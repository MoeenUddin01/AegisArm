# JointGuardian

Predictive-maintenance project: a simulated robot arm (PyBullet) performs a
repetitive task while one joint degrades over time. An LSTM model watches
live joint telemetry and predicts a health score; when health drops below a
threshold the system triggers a safety shutdown before the joint fails
physically.

> **Trace the formula, don't trust the plausibility.** Every derived number
> (health %, RUL, degradation curve) must be traceable back to an explicit
> formula or labeled ground truth.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| ML | PyTorch (LSTM models, training loops) |
| Simulation | PyBullet (physics) |
| Video | OpenCV (frame capture, overlay, mp4 export) |
| Data | pandas / numpy |
| Config | YAML (`configs/`) — no hardcoded values in code |
| Testing | pytest |

## Project Structure

```
.
├── configs/
│   ├── cmapss_config.yaml          # windowing + LSTM hyperparams (Phase 1)
│   ├── joint_health_config.yaml    # joint-health LSTM hyperparams (Phase 4)
│   └── sim_config.yaml             # PyBullet, degradation, health thresholds
├── data/
│   ├── raw/                        # C-MAPSS .txt files (gitignored)
│   ├── processed/                  # windowed tensors
│   └── synthetic/                  # generated joint degradation logs
├── src/
│   ├── data/
│   │   ├── cmapss_loader.py        # load + clean raw C-MAPSS
│   │   ├── windowing.py            # shared sliding-window logic
│   │   └── synthetic_joint_generator.py
│   ├── models/
│   │   ├── lstm_rul.py             # Phase 1: 21-feature LSTM
│   │   └── lstm_joint_health.py    # Phase 4: re-exports LSTMRegressor (3 features)
│   ├── training/
│   │   ├── train_cmapss.py
│   │   └── train_joint_health.py   # Phase 4: train on multirun data
│   ├── evaluation/
│   │   └── metrics.py              # RMSE, NASA scoring, pred-vs-actual plot
│   ├── simulation/
│   │   ├── pybullet_env.py         # arm load, sine-wave motion, joint readout
│   │   ├── degradation.py          # damping power-curve model
│   │   └── health_monitor.py       # live inference loop + shutdown decision
│   ├── video/
│   │   └── recorder.py             # camera capture + overlay + mp4 writer
│   └── utils/
│       ├── logging_utils.py
│       └── seed.py
├── models/                         # saved .pth weights + scaler JSON
├── outputs/
│   ├── logs/                       # CSV joint state logs
│   ├── plots/                      # verification and evaluation plots
│   └── videos/
├── scripts/
│   ├── evaluate_cmapss.py          # Phase 1 eval: scatter plot + metrics
│   ├── evaluate_joint_health.py    # Phase 4 eval: scatter plot + RMSE
│   ├── generate_multirun_data.py   # Phase 3 multi-run generation + overlay plot
│   ├── generate_synthetic_data.py  # Phase 3 single-run generation + verification
│   ├── run_phase2_demo.py          # Phase 2 CLI entrypoint
│   └── run_simulation.py           # full pipeline CLI entrypoint
├── tests/
│   ├── test_cmapss_loader.py
│   ├── test_degradation.py
│   ├── test_evaluate_alignment.py
│   ├── test_health_monitor.py
│   ├── test_multirun_generation.py
│   ├── test_pybullet_env.py
│   ├── test_train_joint_health.py
│   └── test_windowing.py
└── notebooks/
    └── 01_cmapss_lstm_kaggle.ipynb # thin wrapper for Kaggle GPU runs
```

## Installation

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Development Phases

### Phase 1 — C-MAPSS LSTM (Kaggle-capable) ✅

Load NASA C-MAPSS FD001 data, build sliding windows, train an LSTM RUL
regressor, evaluate with RMSE + pred-vs-actual scatter.

**Checkpoint:** scatter-plot dots must cluster near the diagonal. Do not
proceed if they don't — report the metric and stop for review.

**Critical:** the 21-feature C-MAPSS model and the 3–4-feature simulation
model are **incompatible**. Never load Phase 1 weights into the Phase 4
model.

### Phase 2 — PyBullet Basics ✅

Load a KUKA arm, drive one joint with a sine wave, log raw joint state to
CSV. No degradation, no ML.

**Checkpoint:** raw torque/velocity stable for 200+ steps with no
degradation applied.

**Real-time pacing:** the simulation loop includes `time.sleep(timestep)`
so motion is visible and Phase 5 video output won't be a blur.

### Phase 3 — Synthetic Degradation ✅

Inject accelerating damping increase into the joint, log labeled
cycles-to-failure data. Supports both single-run raw-step logs (for
Phase 5 video) and multi-run aggregated datasets (for Phase 4 training)
with configurable parameter variation across runs.

**Checkpoint:** torque RMS must show a visible accelerating trend by eye.
The multi-run overlay plot must show run-to-run variation.

**Outputs:**
- `data/synthetic/joint_degradation_log.csv` — single raw-step run
- `data/synthetic/joint_degradation_multirun.csv` — 25 aggregated runs
- `outputs/plots/phase3_raw_degradation_signal.png` — verification plot
- `outputs/plots/phase3_multirun_overlay.png` — overlay plot

### Phase 4 — Joint Health Model ✅

Train a small LSTM on aggregated multi-run synthetic data (torque_rms,
velocity_rms, position_error_mean), evaluate with RMSE + scatter plot.

**Checkpoint:** predicted cycles-to-failure must be tight near failure
(MAE < 5 cycles in the 1–10 range) — that's the region that determines
correct shutdown timing.

**Results:** eval RMSE 3.66 cycles on 5 held-out runs. Near-failure
MAE 1.85 cycles. Model is tightest where it matters most.

**Outputs:**
- `models/lstm_joint_health.pth` — trained weights
- `models/joint_health_feature_scaler.json` — min-max scaler
- `outputs/plots/joint_health_loss_history.json` — train/val loss
- `outputs/plots/phase4_pred_vs_actual.png` — scatter plot

### Phase 5 — Video Export

Frame capture + cycle-count / health-score overlay + mp4 via
`src/video/recorder.py`.

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use synthetic in-memory data — no real data files required.

## Configuration

All thresholds, window sizes, learning rates, and degradation constants live
in `configs/*.yaml`. Nothing is hardcoded in source. Edit the YAML files, not
the Python, to tune the system.

## Coding Standards

- Type hints on all function signatures.
- Google-style docstrings on every public function and class.
- No magic numbers in code.
- Every module under `src/` is independently importable and testable.
- Logging via `src/utils/logging_utils.py`, not bare `print()`.
- Tests for any function that computes a number used downstream.
