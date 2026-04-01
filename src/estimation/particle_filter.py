import logging

import numpy as np

from .base import ParticleEnvironment
from src.utils.logging_utils import get_process_memory_bytes
from src.utils.tracing import trace_public_methods


@trace_public_methods(
    "simbay.particle_filter",
    exclude={"estimate", "effective_sample_size", "memory_profile"},
)
class ParticleFilter:
    """
    A universal Particle Filter (Sequential Monte Carlo) implementation.

    This class handles the core Bayesian statistics: prediction, likelihood
    weighting, normalization, and resampling. It relies on an external
    ParticleEnvironment to handle domain-specific physics and sensor models.
    """

    def __init__(self, env: ParticleEnvironment, logging_data: dict[str, object] | None = None):
        """
        Initializes the particle filter with a uniform weight distribution.

        Args:
            env (ParticleEnvironment): The domain-specific model (motion and measurement logic).
        """
        self.logger = logging.getLogger("simbay.particle_filter")
        self.env = env
        self.logging_data = dict(logging_data or {})

        # Ask the environment for the initial states first
        init_memory_before = get_process_memory_bytes()
        self.particles = self.env.initialize_particles()
        init_memory_after = get_process_memory_bytes()
        self.N = len(self.particles)

        # Before receiving any sensor data, we assume a uniform prior where
        # every random guess has an exact equal probability of being the true state.
        self.weights = np.ones(self.N) / self.N
        self._last_likelihoods = np.ones(self.N, dtype=np.float64)
        self.state_bytes_total = int(self.particles.nbytes + self.weights.nbytes)
        self.state_bytes_per_particle = self.state_bytes_total / self.N if self.N else 0.0
        self.process_memory_per_particle_estimate = (
            max(init_memory_after - init_memory_before, 0) / self.N if self.N else 0.0
        )

        self.logger.info(
            {
                **self.logging_data,
                "event": "particle_filter_initialized",
                "msg": f"Initialised the particle filter with {self.N} particles.",
                "particles": self.N,
            }
        )

    def predict(self, control_input: np.ndarray):
        """
        Propagates all particles forward in time using the environment's motion model.

        Args:
            control_input: The action or command applied to the system at the current step.
        """
        # Move the system forward in time. The environment is expected to inject
        # random process noise here to represent physical uncertainty and ensure
        # the particle cloud spreads out.
        self.particles = self.env.propagate(self.particles, control_input)

    def update(self, observation: np.ndarray):
        """
        Updates particle weights based on the likelihood of the new observation,
        then normalizes the weights so they sum to 1.

        Args:
            observation: The actual sensor reading from the real world or target system.
        """
        likelihoods = self.env.compute_likelihoods(self.particles, observation)
        self._last_likelihoods = np.asarray(likelihoods, dtype=np.float64).copy()

        # Apply Bayes' Rule: Update our belief by multiplying the current
        # weights by the likelihood of the new observation.
        self.weights *= likelihoods

        # Add a microscopic constant to all weights. If the sensor reading is
        # extreme and assigns 0.0 likelihood to all particles, this prevents a
        # fatal ZeroDivisionError during the normalization step below.
        self.weights += 1.0e-300

        # Normalize the weights so they represent a valid probability distribution (summing to 1)
        sum_weights = self.weights.sum()
        self.weights /= sum_weights

    def resample(self):
        """
        Refocuses the particle cloud by eliminating low-weight particles and
        cloning high-weight particles if the Effective Sample Size (ESS) drops
        below a critical threshold.
        """
        # Compute the Effective Sample Size (ESS) to measure how many particles
        # are actually contributing useful information.
        Neff = 1.0 / np.sum(self.weights**2)

        # Only resample if ESS drops below 50%. This prevents degeneracy while
        # saving computational overhead when the particle cloud is still healthy.
        if Neff < self.N / 2:

            # Systematic Resampling: Instead of spinning a roulette wheel N times,
            # we spin a wheel with N equally spaced pointers exactly once (offset 'u').
            # This is significantly faster and mathematically more stable.
            u = np.random.rand()
            positions = (np.arange(self.N) + u) / self.N

            cumulative_sum = np.cumsum(self.weights)

            # Guard against floating-point rounding errors that could cause index out-of-bounds
            cumulative_sum[-1] = 1.0

            indexes = np.searchsorted(cumulative_sum, positions, side="right")
            self.particles = self.particles[indexes]
            self.env.resample_states(indexes)

            # Reset weights back to uniform for the surviving clones
            self.weights.fill(1.0 / self.N)

    def effective_sample_size(self) -> float:
        """
        Return the current effective sample size (ESS).
        """
        return float(1.0 / np.sum(self.weights**2))

    def memory_profile(self) -> dict[str, float | int]:
        """
        Return memory metrics for the particle state and an RSS-based estimate.
        """
        return {
            "particles": self.N,
            "state_bytes_total": self.state_bytes_total,
            "state_bytes_per_particle": self.state_bytes_per_particle,
            "process_memory_per_particle_estimate_bytes": self.process_memory_per_particle_estimate,
        }

    def estimate(self):
        """
        Computes the current state estimation by calculating the weighted average
        of the entire particle cloud.

        Returns:
            np.ndarray: The estimated state vector.
        """
        # axis=0 ensures we average across the particle rows, returning a proper state vector
        return np.average(self.particles, weights=self.weights, axis=0)
