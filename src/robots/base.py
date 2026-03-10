from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseRobot(ABC):
    """
    Abstract Interface that both Sim and Real robots must implement.
    """
    
    @abstractmethod
    def move_joints(self, pos: np.ndarray):
        """
        Moves the robot to the target joint configuration.
        """
        ...

    @abstractmethod
    def get_pos(self) -> np.ndarray:
        """
        Returns the current joint positions.
        """
        ...

    @abstractmethod
    def get_sensor_reads(self) -> np.ndarray:
        """
        Returns sensory reads.
        """
        ...

    @abstractmethod
    def wait_seconds(self, duration: int):
        """
        Holds the current simulation state for duration (seconds).
        """
        ...

