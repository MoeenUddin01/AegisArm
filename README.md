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
| Language | Python 3.11 |
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
│   │   └── lstm_joint_health.py    # Phase 4: 3–4 joint-feature LSTM
│   ├── training/
│   │   ├── train_cmapss.py
│   │   └── train_joint_health.py
│   ├── evaluation/
│   │   └── metrics.py              # RMSE, NASA scoring, pred-vs-actual plot
│   ├── simulation/
│   │   ├── pybullet_env.py         # arm load, sine-wave motion, joint readout
│   │   ├── degradation.py          # friction / torque-noise injection
│   │   └── health_monitor.py       # live inference loop + shutdown decision
│   ├── video/
│   │   └── recorder.py             # camera capture + overlay + mp4 writer
│   └── utils/
│       ├── logging_utils.py
│       └── seed.py
├── models/                         # saved .pth weights
├── outputs/
│   ├── plots/
│   └── videos/
├── scripts/
│   ├── generate_synthetic_data.py  # Phase 3 data generation
│   └── run_simulation.py           # full pipeline CLI entrypoint
├── tests/
│   ├── test_cmapss_loader.py
│   ├── test_windowing.py
│   ├── test_degradation.py
│   └── test_health_monitor.py
└── notebooks/
    └── 01_cmapss_lstm_kaggle.ipynb # thin wrapper for Kaggle GPU runs
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development Phases

### Phase 1 — C-MAPSS LSTM (Kaggle-capable)

Load NASA C-MAPSS FD001 data, build sliding windows, train an LSTM RUL
regressor, evaluate with RMSE + pred-vs-actual scatter.

**Checkpoint:** scatter-plot dots must cluster near the diagonal. Do not
proceed if they don't — report the metric and stop for review.

**Critical:** the 21-feature C-MAPSS model and the 3–4-feature simulation
model are **incompatible**. Never load Phase 1 weights into the Phase 4
model.

### Phase 2 — PyBullet Basics

Load a KUKA arm, drive one joint with a sine wave, log raw joint state to
CSV. No degradation, no ML.

**Checkpoint:** raw torque/velocity stable for 200+ steps with no
degradation applied.

### Phase 3 — Synthetic Degradation

Inject friction increase into the joint, log labeled cycles-to-failure
data. Verify the raw signal visibly trends before training anything.

**Checkpoint:** plot raw degraded signal — trend must be visible by eye.

### Phase 4 — Joint Health Model

Train `lstm_joint_health.py` on synthetic data, wire it into the live
PyBullet loop, convert predicted RUL into 0–100 % health score, trigger
shutdown below threshold (default 20 %, configurable in YAML).

**Checkpoint:** predicted health must cross the shutdown threshold *before*
the synthetic failure point, not after.

### Phase 5 — Video Export

Frame capture + cycle-count / health-score overlay + mp4 via
`src/video/recorder.py`.

## Running Tests

```bash
pytest tests/ -v
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
