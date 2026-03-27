"""
Robots module providing a unified interface for Franka FR3 control
across different simulation engines and real-world hardware.
"""

from .base import BaseRobot
from .mujoco_robot import MujocoRobot
from .real_robot import RealRobot

# Defines the public API for the robots module
__all__ = [
    "BaseRobot",
    "MujocoRobot",
    "RealRobot",
]
