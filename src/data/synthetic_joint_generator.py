"""Synthetic joint degradation data generator.

.. warning::

    This script takes several minutes to run depending on ``total_cycles``
    and ``steps_per_cycle``.  It is headless by design and must **not**
    be run with ``gui=True``.

Phase 3 drives the KUKA joint through hundreds of sine cycles while
damping increases along an accelerating curve.  The resulting CSV
(``joint_degradation_log.csv``) is the training/validation dataset for
Phase 4's joint-health LSTM.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.simulation.degradation import apply_damping, compute_damping_at_cycle
from src.simulation.pybullet_env import (
    compute_sine_target,
    create_environment,
    get_joint_state,
)


def generate_degradation_run(config_path: str) -> pd.DataFrame:
    """Run a full degradation cycle and persist the joint-state log.

    Reads the simulation config, creates a headless PyBullet environment,
    drives the configured joint through ``total_cycles`` sine periods with
    increasing damping, and records joint telemetry at every physics step.

    Args:
        config_path: Path to the YAML simulation configuration file.

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
    max_damping: float = deg["max_damping"]
    power: float = deg["degradation_power"]
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
