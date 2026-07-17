# CLAUDE.md — JointGuardian

## Project overview

JointGuardian is a predictive-maintenance learning project. A simulated robot
arm (PyBullet) performs a repetitive task while one joint degrades over time.
An LSTM model, trained separately, watches live joint telemetry and predicts
a health score. When health drops below a threshold, the system triggers a
safety shutdown before the joint would physically fail. The final deliverable
includes a recorded video of the arm running and shutting down, with an
overlay showing cycle count and health score.

This is the author's first robotics/simulation project, following prior
full-stack ML projects (sports prediction, autonomous trading). Those
projects established a working principle that governs this one too:

> **Trace the formula, don't trust the plausibility.** Every derived number
> (health %, RUL, degradation curve) must be traceable back to an explicit
> formula or labeled ground truth. Never accept a result because it "looks
> about right" — prove it against the number that produced it.

## Critical constraint — read before writing any model code

C-MAPSS (Phase 1) and the PyBullet simulation (Phase 3+) are **different
feature spaces**. C-MAPSS has 21 aircraft-engine sensor channels; the
simulated joint produces 3-4 features (torque, velocity, position error).

**Never load Phase 1 weights (`lstm_pdm.pth`) into the Phase 3/4 simulation
model.** They have incompatible input dimensions and were trained on
unrelated physical processes. Phase 1's job is to prove the windowing +
LSTM + RUL regression *technique* works. Phase 4 trains a **new, separate**
model on synthetic joint data using that same technique. If any code
attempts to load `lstm_pdm.pth` into `lstm_joint_health.py`'s model class,
that is a bug — stop and flag it rather than reshaping inputs to force a fit.

## Tech stack

- Python 3.11
- PyTorch (LSTM models, training loops)
- PyBullet (physics simulation)
- OpenCV (frame capture, overlay, video writing)
- pandas / numpy (data handling)
- pytest (tests)
- YAML configs (no hardcoded hyperparameters or thresholds in code)

## Development phases

1. **Phase 1 (Kaggle-capable, code lives in `src/`)** — C-MAPSS LSTM: load
   data, build sliding windows, train RUL regressor, evaluate with RMSE +
   pred-vs-actual scatter. Output: `models/lstm_pdm.pth` (reference artifact
   only — not reused downstream).
2. **Phase 2 (local)** — PyBullet basics: load KUKA arm, drive one joint with
   a sine wave, log raw joint state to CSV. No degradation, no ML yet.
3. **Phase 3 (local)** — Inject synthetic degradation (friction increase)
   into the joint. Log labeled cycles-to-failure data. Verify the raw signal
   visibly trends before training anything on it.
4. **Phase 4 (local)** — Train `lstm_joint_health.py` on synthetic data.
   Wire it into the live PyBullet loop. Convert predicted RUL into a 0-100%
   health score. Trigger shutdown below threshold (default 20%, configurable).
5. **Phase 5 (local)** — Frame capture + overlay + mp4 export via
   `src/video/recorder.py`.

Work through phases in order. Do not start a phase's model code until the
previous phase's verification checkpoint (below) has passed.

## Verification checkpoints (do not skip)

- **After Phase 1 training**: produce and show the pred-vs-actual RUL
  scatter plot. Dots should cluster near the diagonal. If not, do not
  proceed — report the metric and stop for review.
- **After Phase 2**: log and display raw joint torque/velocity for at least
  200 steps with no degradation applied. Confirm values are stable (proves
  the sim readout itself isn't already noisy/wrong before degradation is added).
- **After Phase 3**: plot the raw degraded signal (torque or position error)
  before training anything. The trend must be visible by eye. If it's flat
  or dominated by noise, do not proceed to model training — the degradation
  formula needs adjustment first.
- **After Phase 4**: plot predicted health score vs. cycle for a full run
  and confirm it crosses the shutdown threshold before the "failure" point
  defined in the synthetic label, not after.

## Coding standards

- Type hints on all function signatures.
- Docstrings (Google style) on every public function and class.
- No magic numbers in code — thresholds, window sizes, learning rates, and
  degradation constants belong in `configs/*.yaml`.
- Every module under `src/` is independently importable and testable — no
  hidden dependency on notebook execution order.
- Logging via `src/utils/logging_utils.py`, not bare `print()`, except in
  `scripts/` CLI entrypoints where console output is the intended UX.
- Tests in `tests/` for any function that computes a number used downstream
  (windowing math, degradation formula, health-score normalization).

## What not to do

- Don't build a FastAPI/Next.js layer for this project unless explicitly
  requested — this is intentionally a local-script project for now.
- Don't add dashboard, database, or deployment code speculatively.
- Don't silently reshape or truncate feature vectors to make incompatible
  data "fit" a model — flag the mismatch instead.
- Don't fabricate evaluation numbers or skip a verification checkpoint to
  keep moving — a failed checkpoint is useful information, not a blocker
  to paper over.