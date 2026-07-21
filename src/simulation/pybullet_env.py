"""PyBullet simulation environment for JointGuardian.

.. note::

    PyBullet must be installed locally (``pip install pybullet``).  This
    module does **not** run on Kaggle — it requires a local physics
    server.

Phase 2 loads a KUKA IIWA arm, drives one joint with a sine wave, and
logs raw joint state.  No degradation, no ML — only proving the
simulation environment is reliable.

``compute_sine_target`` is a pure function (no PyBullet dependency) and
can be imported/tested without pybullet installed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Pure function — no PyBullet import required
# ---------------------------------------------------------------------------

def compute_sine_target(
    step: int,
    amplitude_rad: float,
    frequency_hz: float,
    sim_timestep: float,
) -> float:
    """Compute the sine-wave joint target for a given simulation step.

    This is a pure function with no PyBullet calls — independently
    testable.

    Formula:
        ``amplitude_rad * sin(2 * pi * frequency_hz * step * sim_timestep)``

    Args:
        step: Current simulation step index (0-based).
        amplitude_rad: Half-range of the oscillation in radians.
        frequency_hz: Oscillation frequency in Hz.
        sim_timestep: Duration of one simulation step in seconds.

    Returns:
        Target angle in radians.
    """
    t = step * sim_timestep
    return amplitude_rad * math.sin(2.0 * math.pi * frequency_hz * t)


# ---------------------------------------------------------------------------
# PyBullet-dependent functions — lazy import so compute_sine_target is
# always importable.
# ---------------------------------------------------------------------------

def create_environment(gui: bool = True) -> tuple[int, int]:
    """Connect to PyBullet, set gravity, and load the KUKA IIWA arm.

    Args:
        gui: If ``True``, open an interactive GUI window (``p.GUI``).
            If ``False``, run headless (``p.DIRECT``).

    Returns:
        ``(client_id, robot_id)`` — the physics client and body IDs.
    """
    import pybullet as p
    import pybullet_data

    if gui:
        client = p.connect(p.GUI)
    else:
        client = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.8, physicsClientId=client)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=client)
    robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client,
    )

    return client, robot_id


def get_joint_state(robot_id: int, joint_index: int) -> dict[str, float]:
    """Read the current state of a single joint.

    Args:
        robot_id: PyBullet body ID of the robot.
        joint_index: Index of the joint to read.

    Returns:
        Dictionary with keys ``angle`` (rad), ``velocity`` (rad/s), and
        ``applied_torque`` (N·m).
    """
    import pybullet as p

    pos, vel, _, torques = p.getJointState(robot_id, joint_index)
    return {
        "angle": float(pos),
        "velocity": float(vel),
        "applied_torque": float(torques),
    }


def run_demo(config_path: str, gui: bool = True) -> pd.DataFrame:
    """Run the Phase 2 sine-wave demo and log joint state.

    Reads ``configs/sim_config.yaml`` (or the path provided), creates a
    PyBullet environment, drives the configured joint with a sine wave for
    ``num_steps`` steps, records joint state at every step, writes the log
    to ``log_path``, and disconnects.

    Args:
        config_path: Path to the YAML configuration file.
        gui: Open an interactive window (``True``) or run headless.

    Returns:
        DataFrame with columns ``step``, ``angle``, ``velocity``,
        ``applied_torque``.
    """
    import pybullet as p

    cfg_path = Path(config_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    joint_index: int = cfg["joint_index"]
    amplitude: float = cfg["amplitude_rad"]
    frequency: float = cfg["frequency_hz"]
    timestep: float = cfg["sim_timestep"]
    num_steps: int = cfg["num_steps"]
    log_path = Path(cfg["log_path"])

    client, robot_id = create_environment(gui=gui)

    records: list[dict[str, Any]] = []

    for step in range(num_steps):
        target = compute_sine_target(step, amplitude, frequency, timestep)

        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            physicsClientId=client,
        )
        p.stepSimulation(physicsClientId=client)
        time.sleep(timestep)  # real-time pacing so motion is visible

        state = get_joint_state(robot_id, joint_index)
        records.append({
            "step": step,
            **state,
        })

    # Persist log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(log_path, index=False)

    p.disconnect(physicsClientId=client)
    return df
