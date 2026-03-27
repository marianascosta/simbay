"""
Kinematics module providing Inverse Kinematics (IK) solvers,
numerical stepping engines, and MuJoCo-specific implementations.
"""

from .base import IKProblem
from .engines import StepMethods
from .ik_solver import solve_IKProblem
from .mujoco_ik import MujocoPoseIK

# Defines the public API for the kinematics module
__all__ = [
    "IKProblem",
    "StepMethods",
    "solve_IKProblem",
    "MujocoPoseIK",
]
