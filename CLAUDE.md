# Claude Instructions for Joint Guardian Project

This document provides guidance for Claude when working on the Joint Guardian project.

## Project Overview
Joint Guardian is a predictive maintenance system for robotic arms that combines:
1. LSTM-based RUL prediction using C-MAPSS data
2. Synthetic joint degradation simulation
3. Real-time health monitoring in PyBullet simulation

## Code Conventions
- Use Python 3.8+
- Follow PEP 8 style guidelines
- Include type hints for all functions
- Use docstrings for public APIs

## Testing
- Run pytest for unit tests
- Ensure all tests pass before committing
- Add tests for new functionality

## Data Flow
1. Phase 1: Train LSTM on C-MAPSS data
2. Phase 3: Generate synthetic degradation data
3. Phase 4: Train joint health model
4. Phase 5: Run simulation with live monitoring