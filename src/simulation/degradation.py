"""Degradation model for JointGuardian.

Provides a pure function that maps a cycle number to a damping value
(``compute_damping_at_cycle``) and a thin PyBullet wrapper that applies
that damping to a joint (``apply_damping``).

``compute_damping_at_cycle`` has no PyBullet dependency and is
independently testable.

Formula::

    damping = base + (max - base) * (cycle / total_cycles) ** power

At cycle 0 → base.  At cycle == total_cycles → max.
``power`` > 1 produces an accelerating (concave-up) wear curve.
"""

from __future__ import annotations


def compute_damping_at_cycle(
    cycle: int,
    total_cycles: int,
    base_damping: float,
    max_damping: float,
    power: float,
) -> float:
    """Return the joint damping value for a given cycle.

    Args:
        cycle: Current cycle index (0-based).
        total_cycles: Total number of cycles in the run.
        base_damping: Damping at cycle 0 (N·m·s/rad).
        max_damping: Damping at ``cycle == total_cycles`` (N·m·s/rad).
        power: Exponent controlling curve shape.  ``power == 1`` is
            linear; ``power > 1`` is accelerating wear.

    Returns:
        Damping value in N·m·s/rad.
    """
    progress = cycle / total_cycles
    return base_damping + (max_damping - base_damping) * progress ** power


def apply_damping(robot_id: int, joint_index: int, damping: float) -> None:
    """Set the joint damping on a PyBullet body.

    Thin wrapper around ``p.changeDynamics`` — requires pybullet to be
    installed.

    Args:
        robot_id: PyBullet body ID.
        joint_index: Index of the joint to modify.
        damping: Damping value in N·m·s/rad.
    """
    import pybullet as p

    p.changeDynamics(robot_id, joint_index, jointDamping=damping)
