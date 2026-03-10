import numpy as np

from .base import IKProblem


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
    theta = theta.copy()

    for i in range(max_iter):
        # 1. Compute Forward Kinematics
        s = problem.get_s(theta)
        
        # 2. Compute Error
        e = problem.get_e(s, t)
        error = np.linalg.norm(e)
        
        # 3. Check Convergence
        if error < tol:
            print(f"[IK Info] Converged in {i} iterations with error: {error:.2e}.")
            return theta

        # 4. Compute Jacobian & Step
        J = problem.get_jacobian(s, theta)
        delta_theta = problem.step(J, e)
        theta += delta_theta
        
        # 5. Enforce Limits
        theta = problem.clamp_to_limits(theta)
    
    # NOTE: If we reach max_iter, the returned 'error' corresponds to the 
    # second-to-last position. This is acceptable for failed/non-converged states.
    print(f"[IK Info] Did not converge in {i} iterations with 2nd to last error: {error:.2e}.")
    return theta