import numpy as np

from src.utils.tracing import set_span_attributes
from src.utils.tracing import trace_call


@trace_call("simbay.trajectory", "planning.plan_linear_trajectory")
def plan_linear_trajectory(start_pos, target_pos, max_velocity, dt, settle_time=0.2):
    """
    Calculates a straight-line trajectory in joint space.
    Returns a NumPy array of intermediate steps.
    
    Args:
        settle_time (float): Time in seconds to hold the final position 
                             so the physical PD controllers can stabilize.
    """
    set_span_attributes(
        {
            "planning.max_velocity": float(max_velocity),
            "planning.control_dt": float(dt),
            "planning.settle_time": float(settle_time),
            "planning.start_dim": int(np.asarray(start_pos).shape[0]),
            "planning.target_dim": int(np.asarray(target_pos).shape[0]),
        }
    )
    start_pos = np.array(start_pos)
    target_pos = np.array(target_pos)
    
    # Calculate distance and bottleneck time
    distance = np.abs(target_pos - start_pos)
    duration = np.max(distance / max_velocity)
    
    # Use CEIL to ensure any fractional step rounds UP to a full step
    total_steps = int(np.ceil(duration / dt))
    set_span_attributes(
        {
            "planning.duration": float(duration),
            "planning.total_steps": total_steps,
        }
    )

    # Generate total_steps + 1 points, and slice off the start_pos
    moving_path = np.linspace(start_pos, target_pos, total_steps + 1)[1:]

    # --- NEW: Settle Phase ---
    if settle_time > 0:
        settle_steps = int(np.ceil(settle_time / dt))
        # Create a matrix of just the final position
        settling_path = np.tile(target_pos, (settle_steps, 1))
        # Stack the moving path and the settling path together
        return np.vstack((moving_path, settling_path))

    return moving_path
