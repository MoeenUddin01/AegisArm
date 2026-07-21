"""Synthetic joint degradation data generator.

.. warning::

    This script takes several minutes to run depending on ``total_cycles``
    and ``steps_per_cycle``.  It is headless by design and must **not**
    be run with ``gui=True``.

Phase 3 drives the KUKA joint through hundreds of sine cycles while
damping increases along an accelerating curve.  The resulting CSV
(``joint_degradation_log.csv``) is the raw per-step dataset.
``generate_multi_run_dataset`` produces an aggregated multi-run dataset
(``joint_degradation_multirun.csv``) with per-cycle features and run IDs
for Phase 4 training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.simulation.degradation import apply_damping, compute_damping_at_cycle
from src.simulation.pybullet_env import (
    compute_sine_target,
    create_environment,
    get_joint_state,
)


def generate_degradation_run(
    config_path: str,
    max_damping_override: float | None = None,
    power_override: float | None = None,
) -> pd.DataFrame:
    """Run a full degradation cycle and persist the joint-state log.

    Reads the simulation config, creates a headless PyBullet environment,
    drives the configured joint through ``total_cycles`` sine periods with
    increasing damping, and records joint telemetry at every physics step.

    Args:
        config_path: Path to the YAML simulation configuration file.
        max_damping_override: If provided, overrides ``max_damping`` from
            the config.  Used by multi-run generation to vary parameters.
        power_override: If provided, overrides ``degradation_power`` from
            the config.

    Returns:
        DataFrame with columns ``step``, ``cycle``, ``target_angle``,
        ``actual_angle``, ``position_error``, ``velocity``,
        ``applied_torque``, ``current_damping``, ``cycles_to_failure``.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    joint_index: int = cfg["joint_index"]
    amplitude: float = cfg["amplitude_rad"]
    frequency: float = cfg["frequency_hz"]
    timestep: float = cfg["sim_timestep"]

    deg = cfg["degradation"]
    base_damping: float = deg["base_damping"]
    max_damping: float = max_damping_override if max_damping_override is not None else deg["max_damping"]
    power: float = power_override if power_override is not None else deg["degradation_power"]
    total_cycles: int = deg["total_cycles"]

    steps_per_cycle = round(1 / (frequency * timestep))
    total_steps = total_cycles * steps_per_cycle

    client, robot_id = create_environment(gui=False)

    records: list[dict] = []
    current_damping = base_damping

    for step in range(total_steps):
        current_cycle = step // steps_per_cycle

        # Apply new damping at the start of each cycle
        if step % steps_per_cycle == 0:
            current_damping = compute_damping_at_cycle(
                current_cycle, total_cycles, base_damping, max_damping, power,
            )
            apply_damping(robot_id, joint_index, current_damping)

        target = compute_sine_target(step, amplitude, frequency, timestep)

        import pybullet as p
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            physicsClientId=client,
        )
        p.stepSimulation(physicsClientId=client)

        state = get_joint_state(robot_id, joint_index)
        actual_angle = state["angle"]
        records.append({
            "step": step,
            "cycle": current_cycle,
            "target_angle": target,
            "actual_angle": actual_angle,
            "position_error": target - actual_angle,
            "velocity": state["velocity"],
            "applied_torque": state["applied_torque"],
            "current_damping": current_damping,
            "cycles_to_failure": total_cycles - current_cycle,
        })

    p.disconnect(physicsClientId=client)

    df = pd.DataFrame(records)
    output_path = Path(deg["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def aggregate_cycle_features(raw_df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Aggregate raw per-step data into per-cycle features.

    Groups the raw log by ``cycle`` and computes RMS torque, RMS velocity,
    and mean absolute position error for each cycle.  The raw ``damping``
    column is intentionally excluded — it is a hidden ground-truth variable
    used only to generate the label, never a model input.

    Args:
        raw_df: Raw per-step DataFrame from ``generate_degradation_run``.
        run_id: Integer ID for this run (included in output).

    Returns:
        DataFrame with one row per cycle, columns: ``cycle``,
        ``torque_rms``, ``velocity_rms``, ``position_error_mean``,
        ``cycles_to_failure``, ``run_id``.
    """
    groups = raw_df.groupby("cycle")

    aggregated = pd.DataFrame({
        "cycle": groups["cycle"].first(),
        "torque_rms": groups["applied_torque"].apply(
            lambda s: float(np.sqrt(np.mean(s ** 2))),
        ),
        "velocity_rms": groups["velocity"].apply(
            lambda s: float(np.sqrt(np.mean(s ** 2))),
        ),
        "position_error_mean": groups["position_error"].apply(
            lambda s: float(s.abs().mean()),
        ),
        "cycles_to_failure": groups["cycles_to_failure"].first(),
    })

    aggregated["run_id"] = run_id
    aggregated.reset_index(drop=True, inplace=True)
    return aggregated


def generate_multi_run_dataset(config_path: str) -> pd.DataFrame:
    """Generate an aggregated multi-run degradation dataset.

    Runs ``num_runs`` independent degradation simulations, each with
    slightly varied ``max_damping`` and ``degradation_power`` sampled
    within the configured variation ranges.  Each run is seeded with
    ``seed_base + i`` for reproducibility.  Results are aggregated to
    per-cycle features and concatenated into a single DataFrame.

    Args:
        config_path: Path to the YAML simulation configuration file.

    Returns:
        DataFrame with columns ``cycle``, ``torque_rms``, ``velocity_rms``,
        ``position_error_mean``, ``cycles_to_failure``, ``run_id``.
        One row per cycle per run (~300 × num_runs rows).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    deg = cfg["degradation"]
    num_runs: int = deg["num_runs"]
    damping_var_pct: float = deg["damping_variation_pct"]
    power_var: float = deg["power_variation"]
    seed_base: int = deg["seed_base"]

    base_max_damping: float = deg["max_damping"]
    base_power: float = deg["degradation_power"]

    all_runs: list[pd.DataFrame] = []

    for i in range(num_runs):
        # Lazy import — torch is not needed for single-run tests
        from src.utils.seed import set_seed
        set_seed(seed_base + i)

        # Sample this run's parameters within variation ranges
        rng = np.random.RandomState(seed_base + i)
        run_max_damping = base_max_damping * (1.0 + rng.uniform(-damping_var_pct, damping_var_pct))
        run_power = base_power + rng.uniform(-power_var, power_var)

        raw_df = generate_degradation_run(
            config_path,
            max_damping_override=run_max_damping,
            power_override=run_power,
        )
        cycle_df = aggregate_cycle_features(raw_df, run_id=i)
        all_runs.append(cycle_df)

    multi_run_df = pd.concat(all_runs, ignore_index=True)

    output_path = Path(deg.get(
        "multirun_output_csv",
        "data/synthetic/joint_degradation_multirun.csv",
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    multi_run_df.to_csv(output_path, index=False)
    return multi_run_df
