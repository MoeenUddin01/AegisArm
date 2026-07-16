# Joint Guardian

A predictive maintenance system for robotic arms using LSTM-based RUL prediction and real-time health monitoring.

## Project Structure
- `src/` - Core source code
- `configs/` - Configuration files
- `data/` - Data directories (raw, processed, synthetic)
- `scripts/` - CLI entrypoints
- `tests/` - Unit tests
- `models/` - Saved model weights
- `outputs/` - Plots and videos

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare C-MAPSS data in `data/raw/`
2. Run training scripts
3. Execute simulation with `scripts/run_simulation.py`