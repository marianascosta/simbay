import logging

import numpy as np

from src.utils.tracing import set_span_attributes
from src.utils.tracing import trace_call

from .base import IKProblem


logger = logging.getLogger("simbay.ik_solver")


@trace_call("simbay.ik_solver", "ik.solve")
def solve_IKProblem(problem: IKProblem, theta: np.ndarray, t: np.ndarray,
                       tol: float = 1e-6, max_iter: int = 500):
    """
    Generic Iterative Inverse Kinematics Solver.

    This function solves the IK problem for any robot implementing the IKProblem interface.
    It uses a robust loop with safety checks for convergence and iteration limits.

    Args:
        problem (IKProblem): The robot-specific problem instance.
        theta (np.ndarray): The initial joint guess.
        t (np.ndarray): The target position/pose.
        tol (float): The tolerance for the error (distance to target).
        max_iter (int): The maximum number of iterations to prevent infinite loops.

    Returns:
        tuple: (final_theta (np.ndarray), final_error (float))
    """
    # CRITICAL FIX: Make a copy so we don't overwrite the original 'theta' outside this function
    set_span_attributes(
        {
            "ik.problem_class": problem.__class__.__name__,
            "ik.theta_dim": int(theta.shape[0]),
            "ik.target_dim": int(t.shape[0]),
            "ik.tolerance": float(tol),
            "ik.max_iterations": int(max_iter),
        }
    )
    theta = theta.copy()

    for i in range(max_iter):
        set_span_attributes({"ik.iteration": i})
        # 1. Compute Forward Kinematics
        s = problem.get_s(theta)
        
        # 2. Compute Error
        e = problem.get_e(s, t)
        error = np.linalg.norm(e)
        
        # 3. Check Convergence
        if error < tol:
            set_span_attributes(
                {
                    "ik.converged": True,
                    "ik.iterations": i,
                    "ik.residual_norm": float(error),
                }
            )
            logger.info(
                {
                    "event": "ik_converged",
                    "msg": f"IK converged in {i} iterations.",
                    "iterations": i,
                    "residual_norm": error,
                }
            )
            return theta

        # 4. Compute Jacobian & Step
        J = problem.get_jacobian(s, theta)
        delta_theta = problem.step(J, e)
        theta += delta_theta
        
        # 5. Enforce Limits
        theta = problem.clamp_to_limits(theta)
    
    # NOTE: If we reach max_iter, the returned 'error' corresponds to the 
    # second-to-last position. This is acceptable for failed/non-converged states.
    logger.info(
        {
            "event": "ik_not_converged",
            "msg": f"IK did not converge after {i} iterations.",
            "iterations": i,
            "residual_norm": error,
        }
    )
    set_span_attributes(
        {
            "ik.converged": False,
            "ik.iterations": i,
            "ik.residual_norm": float(error),
        }
    )
    return theta
