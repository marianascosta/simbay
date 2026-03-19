from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryProfile:
    total_steps: int
    move_steps: int
    settle_steps: int
    control_dt_seconds: float
    settle_time_seconds: float
    control_duration_seconds: float
    max_joint_delta: float
    max_velocity: float


def describe_linear_trajectory(
    start_pos,
    target_pos,
    max_velocity,
    control_dt,
    settle_time=0.2,
) -> TrajectoryProfile:
    start_pos = np.array(start_pos)
    target_pos = np.array(target_pos)
    if control_dt <= 0:
        raise ValueError("control_dt must be greater than zero")

    distance = np.abs(target_pos - start_pos)
    control_duration = float(np.max(distance / max_velocity))
    move_steps = int(np.ceil(control_duration / control_dt))
    settle_steps = int(np.ceil(settle_time / control_dt)) if settle_time > 0 else 0
    total_steps = move_steps + settle_steps

    return TrajectoryProfile(
        total_steps=total_steps,
        move_steps=move_steps,
        settle_steps=settle_steps,
        control_dt_seconds=float(control_dt),
        settle_time_seconds=float(settle_time),
        control_duration_seconds=control_duration,
        max_joint_delta=float(np.max(distance)) if distance.size else 0.0,
        max_velocity=float(max_velocity),
    )


def plan_linear_trajectory(start_pos, target_pos, max_velocity, control_dt, settle_time=0.2):
    """
    Calculates a straight-line trajectory in joint space.
    Returns a NumPy array of intermediate steps.
    
    Args:
        settle_time (float): Time in seconds to hold the final position 
                             so the physical PD controllers can stabilize.
    """
    start_pos = np.array(start_pos)
    target_pos = np.array(target_pos)
    profile = describe_linear_trajectory(
        start_pos,
        target_pos,
        max_velocity=max_velocity,
        control_dt=control_dt,
        settle_time=settle_time,
    )

    # Generate move_steps + 1 points, and slice off the start_pos
    moving_path = np.linspace(start_pos, target_pos, profile.move_steps + 1)[1:]

    # --- NEW: Settle Phase ---
    if profile.settle_steps > 0:
        # Create a matrix of just the final position
        settling_path = np.tile(target_pos, (profile.settle_steps, 1))
        # Stack the moving path and the settling path together
        return np.vstack((moving_path, settling_path))

    return moving_path
