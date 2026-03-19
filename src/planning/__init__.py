"""
Planning module providing trajectory generation and 
high-level IK solvers for Franka robot motions.
"""

from .franka_solver import FrankaSmartSolver
from .trajectory import describe_linear_trajectory
from .trajectory import plan_linear_trajectory

# Defines the public API for the planning module
__all__ = [
    "FrankaSmartSolver",
    "describe_linear_trajectory",
    "plan_linear_trajectory",
]
