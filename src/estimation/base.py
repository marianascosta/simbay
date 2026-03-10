from abc import ABC
from abc import abstractmethod

import numpy as np


class ParticleEnvironment(ABC):
    """
    Abstract blueprint for any environment plugged into the Particle Filter.
    It isolates the domain-specific physics and math from the pure statistics engine.
    """

    @property
    @abstractmethod
    def num_particles(self) -> int:
        """
        The number of particles (N) this environment is managing.
        Concrete implementations should typically set this during __init__.
        """
        pass

    @abstractmethod
    def initialize_particles(self) -> np.ndarray:
        """
        Generates the initial guesses for the particles.
        
        Returns:
            np.ndarray: An array of shape (N, state_dimension) containing initial states.
        """
        pass

    @abstractmethod
    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """
        Steps the simulation forward and applies random process noise.
        (The 'Prediction' step)
        
        Args:
            particles (np.ndarray): The current states of all N particles.
            control_input (np.ndarray): The action applied to the system (e.g., joint commands).
            
        Returns:
            np.ndarray: The new predicted states of the particles.
        """
        pass

    @abstractmethod
    def compute_likelihoods(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Compares simulated sensor readings against the real observation.
        (The 'Measurement Update' step)
        
        Args:
            particles (np.ndarray): The current states of all N particles.
            observation (np.ndarray): The actual sensor reading from the real world/PyBullet.
            
        Returns:
            np.ndarray: A 1D array of length N containing the likelihood weights.
        """
        pass           