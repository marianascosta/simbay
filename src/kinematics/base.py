from abc import ABC
from abc import abstractmethod

import numpy as np


class IKProblem(ABC):
    """
    Abstract Base Class defining the interface for a generic IK problem.
    
    Any specific robot implementation (MuJoCo, PyBullet, etc.) must inherit 
    from this class and implement these methods.
    """
    
    @abstractmethod
    def get_s(self, theta: np.ndarray) -> np.ndarray:
        """
        Computes the forward kinematics to get the current end-effector position/pose.

        Args:
            theta (np.ndarray): The current joint configuration.

        Returns:
            np.ndarray: The current position (or pose) of the end-effector.
        """
        ...

    @abstractmethod
    def get_e(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Computes the error vector between the current state and the target.

        This method abstracts the error calculation logic. 
        - For Position IK: It might simply be (target - current).
        - For Orientation IK: It must handle quaternion differences or axis-angle 
          errors to ensure the result is a rotation vector (radians), not just 
          subtracted components.

        Args:
            s (np.ndarray): The current end-effector state (from get_s).
            t (np.ndarray): The target state.

        Returns:
            np.ndarray: The error vector 'e' to be minimized.
                        Shape must match the rows of the Jacobian (3, 6, etc.).
        """
        ...

    @abstractmethod
    def get_jacobian(self, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix for the current configuration.

        Args:
            s (np.ndarray): The current end-effector position (optional optimization).
            theta (np.ndarray): The current joint configuration.

        Returns:
            np.ndarray: The Jacobian matrix.
        """
        ...

    @abstractmethod
    def step(self, J: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Calculates the joint delta for a single iteration.
        """
        ...

    @abstractmethod
    def clamp_to_limits(self, theta: np.ndarray) -> np.ndarray:
        """
        Enforces physical joint limits on the configuration.

        Args:
            theta (np.ndarray): The joint configuration to check.

        Returns:
            np.ndarray: The clamped joint configuration.
        """
        ...